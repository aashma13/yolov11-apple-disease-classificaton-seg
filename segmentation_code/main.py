# %%writefile /content/train_yolo11_seg_fixed.py
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import yaml
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_coco(json_path: Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_poly(poly: List[float], width: int, height: int) -> List[float]:
    if len(poly) < 6 or len(poly) % 2 != 0:
        return []
    out = []
    for i in range(0, len(poly), 2):
        x = min(max(poly[i] / width, 0.0), 1.0)
        y = min(max(poly[i + 1] / height, 0.0), 1.0)
        out.extend([x, y])
    return out


def prepare_split(split_dir: Path) -> tuple[Path, Path]:
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    ensure_dir(images_dir)
    ensure_dir(labels_dir)

    # Move image files from split root into images/
    for p in list(split_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            target = images_dir / p.name
            if p.resolve() != target.resolve():
                p.rename(target)

    return images_dir, labels_dir


def coco_to_yolo_seg_for_split(
    split_dir: Path,
    json_name: str = "_annotations.coco.json",
) -> Tuple[Dict[int, int], List[str], Dict[str, Any]]:
    json_path = split_dir / json_name
    if not json_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {json_path}")

    images_dir, labels_dir = prepare_split(split_dir)
    coco = load_coco(json_path)

    images = {img["id"]: img for img in coco["images"]}
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    used_cat_ids = sorted({ann["category_id"] for ann in annotations})
    kept_categories = [c for c in categories if c["id"] in used_cat_ids]
    kept_categories = sorted(kept_categories, key=lambda c: c["id"])

    if not kept_categories:
        raise ValueError(f"No used categories found in {json_path}")

    cat_id_to_yolo_idx = {cat["id"]: i for i, cat in enumerate(kept_categories)}
    class_names = [cat["name"] for cat in kept_categories]

    anns_by_image: Dict[int, List[dict]] = defaultdict(list)
    for ann in annotations:
        anns_by_image[ann["image_id"]].append(ann)

    label_counts = Counter()
    skipped_anns = 0
    empty_images = 0

    for image_id, img in images.items():
        file_name = img["file_name"]
        width = int(img["width"])
        height = int(img["height"])

        image_path = images_dir / file_name
        if not image_path.exists():
            raise FileNotFoundError(f"Image listed in JSON not found: {image_path}")

        label_path = labels_dir / f"{Path(file_name).stem}.txt"
        lines: List[str] = []

        for ann in anns_by_image.get(image_id, []):
            cat_id = ann["category_id"]
            if cat_id not in cat_id_to_yolo_idx:
                skipped_anns += 1
                continue

            seg = ann.get("segmentation", [])
            if not isinstance(seg, list) or not seg:
                skipped_anns += 1
                continue

            polygons = [p for p in seg if isinstance(p, list) and len(p) >= 6 and len(p) % 2 == 0]
            if not polygons:
                skipped_anns += 1
                continue

            polygon = max(polygons, key=len)
            norm = normalize_poly(polygon, width, height)
            if len(norm) < 6:
                skipped_anns += 1
                continue

            cls_idx = cat_id_to_yolo_idx[cat_id]
            lines.append(str(cls_idx) + " " + " ".join(f"{v:.6f}" for v in norm))
            label_counts[class_names[cls_idx]] += 1

        with open(label_path, "w", encoding="utf-8") as f:
            if lines:
                f.write("\n".join(lines) + "\n")
            else:
                empty_images += 1

    stats = {
        "split": split_dir.name,
        "num_images": len(images),
        "num_annotations": len(annotations),
        "num_empty_images": empty_images,
        "num_skipped_annotations": skipped_anns,
        "label_counts": dict(label_counts),
    }
    return cat_id_to_yolo_idx, class_names, stats


def validate_split_dirs(dataset_root: Path) -> None:
    for s in ("train", "valid", "test"):
        if not (dataset_root / s).exists():
            raise FileNotFoundError(f"Missing split: {dataset_root / s}")


def build_data_yaml(dataset_root: Path, class_names: List[str]) -> Path:
    data = {
        "path": str(dataset_root.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": {i: n for i, n in enumerate(class_names)},
    }
    yaml_path = dataset_root / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return yaml_path


def summarize_metrics(metrics_obj: Any) -> Dict[str, Any]:
    out = {}
    for head_name in ("box", "seg"):
        head = getattr(metrics_obj, head_name, None)
        if head is None:
            continue
        d = {}
        try:
            r = head.mean_results()
            if len(r) >= 4:
                d["precision"] = float(r[0])
                d["recall"] = float(r[1])
                d["mAP50"] = float(r[2])
                d["mAP50_95"] = float(r[3])
        except Exception:
            pass
        try:
            d["per_class_mAP50_95"] = [float(x) for x in head.maps]
        except Exception:
            pass
        out[head_name] = d
    try:
        out["speed_ms_per_image"] = dict(metrics_obj.speed)
    except Exception:
        pass
    return out


def main():
    os.environ["WANDB_MODE"] = "disabled"

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    # parser.add_argument("--model", type=str, default="yolo11x-seg.pt")
    parser.add_argument("--model", default="yolo11l-seg")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--project", type=str, default="/content/apple_disease_seg")
    parser.add_argument("--name", type=str, default="yolo11l_seg_v5_fixed")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    validate_split_dirs(dataset_root)

    all_stats = {}
    ref_class_names = None
    for split in ("train", "valid", "test"):
        _, class_names, stats = coco_to_yolo_seg_for_split(dataset_root / split)
        all_stats[split] = stats
        if ref_class_names is None:
            ref_class_names = class_names
        elif ref_class_names != class_names:
            raise ValueError(f"Class mismatch in {split}")
        print(f"[INFO] Converted {split}: {stats}")

    yaml_path = build_data_yaml(dataset_root, ref_class_names)
    print(f"[INFO] data.yaml: {yaml_path}")
    print(f"[INFO] Classes: {ref_class_names}")

    model = YOLO(args.model)
    model.train(
        data=str(yaml_path),
        task="segment",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        pretrained=True,
        project=args.project,
        name=args.name,
        exist_ok=True,
        patience=args.patience,
        # optimizer='AdamW',
        optimizer="auto",
        # lr0=0.001,
        # weight_decay=0.0005,


        amp=False,
        cos_lr=True,
        overlap_mask=True,
        mask_ratio=4,
        plots=True,
        save=True,
        save_period=10,
        val=True,
        verbose=False,

        # disable extra online augmentation since Roboflow already augmented the export
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        fliplr=0.0,
        flipud=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
    )

    run_dir = Path(model.trainer.save_dir)
    best_weights = run_dir / "weights" / "best.pt"
    best_model = YOLO(str(best_weights))

    val_metrics = best_model.val(
        data=str(yaml_path),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        plots=True,
        save_json=True,
        verbose=False,
    )

    test_metrics = best_model.val(
        data=str(yaml_path),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        plots=True,
        save_json=True,
        verbose=False,
    )

    summary = {
        "dataset_root": str(dataset_root),
        "data_yaml": str(yaml_path),
        "model": args.model,
        "classes": ref_class_names,
        "conversion_stats": all_stats,
        "best_weights": str(best_weights),
        "val_metrics": summarize_metrics(val_metrics),
        "test_metrics": summarize_metrics(test_metrics),
    }

    with open(run_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[INFO] Run dir: {run_dir}")
    print(f"[INFO] Best weights: {best_weights}")


if __name__ == "__main__":
    main()