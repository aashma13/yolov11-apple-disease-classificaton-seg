#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from sklearn.metrics import confusion_matrix
from ultralytics import YOLO


# ----------------------------
# Utilities
# ----------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_split_images(data_cfg: dict, split: str) -> Path:
    root = Path(data_cfg["path"])
    rel = data_cfg[split]
    return (root / rel).resolve()


def names_from_data(data_cfg: dict) -> List[str]:
    names = data_cfg["names"]
    if isinstance(names, list):
        return names
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=lambda x: int(x))]
    raise ValueError("Unsupported names format in data.yaml")


def list_images(images_dir: Path) -> List[Path]:
    return sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])


def image_to_label_path(img_path: Path) -> Path:
    # expects dataset split like .../test/images/*.jpg -> .../test/labels/*.txt
    return img_path.parent.parent / "labels" / f"{img_path.stem}.txt"


def read_gt_instances(label_path: Path, image_shape: Tuple[int, int]) -> List[dict]:
    h, w = image_shape[:2]
    instances = []
    if not label_path.exists():
        return instances

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f.readlines() if x.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) < 7:
            continue
        cls = int(float(parts[0]))
        coords = np.array([float(x) for x in parts[1:]], dtype=np.float32).reshape(-1, 2)
        coords[:, 0] = np.clip(coords[:, 0] * w, 0, w - 1)
        coords[:, 1] = np.clip(coords[:, 1] * h, 0, h - 1)

        mask = np.zeros((h, w), dtype=np.uint8)
        pts = coords.astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 1)
        area = int(mask.sum())
        if area == 0:
            continue

        instances.append(
            {
                "class_id": cls,
                "mask": mask.astype(bool),
                "area": area,
            }
        )
    return instances


def extract_pred_instances(result, conf: float = 0.25) -> List[dict]:
    instances = []
    boxes = result.boxes
    masks = result.masks

    if boxes is None or masks is None or masks.data is None:
        return instances

    cls_ids = boxes.cls.detach().cpu().numpy().astype(int)
    confs = boxes.conf.detach().cpu().numpy()
    masks_np = masks.data.detach().cpu().numpy()  # [N, H, W]

    for cls_id, score, mask in zip(cls_ids, confs, masks_np):
        if score < conf:
            continue
        bin_mask = mask > 0.5
        area = int(bin_mask.sum())
        if area == 0:
            continue
        instances.append(
            {
                "class_id": int(cls_id),
                "score": float(score),
                "mask": bin_mask,
                "area": area,
            }
        )

    return instances


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def mask_dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return float((2.0 * inter) / denom) if denom > 0 else 0.0


def greedy_match_instances(
    gt_instances: List[dict],
    pred_instances: List[dict],
    iou_thr: float,
) -> List[Tuple[Optional[int], Optional[int], float]]:
    """
    Returns list of tuples:
        (gt_index or None, pred_index or None, iou)
    Matched pairs first, then unmatched GT, then unmatched predictions.
    """
    pairs = []
    candidates = []

    for gi, gt in enumerate(gt_instances):
        for pi, pred in enumerate(pred_instances):
            iou = mask_iou(gt["mask"], pred["mask"])
            if iou >= iou_thr:
                candidates.append((iou, gi, pi))

    candidates.sort(key=lambda x: x[0], reverse=True)

    used_gt = set()
    used_pred = set()

    for iou, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        pairs.append((gi, pi, iou))

    for gi in range(len(gt_instances)):
        if gi not in used_gt:
            pairs.append((gi, None, 0.0))

    for pi in range(len(pred_instances)):
        if pi not in used_pred:
            pairs.append((None, pi, 0.0))

    return pairs


# ----------------------------
# Metrics aggregation
# ----------------------------

class SegmentationEvaluator:
    def __init__(self, class_names: List[str], background_name: str = "background") -> None:
        self.class_names = class_names
        self.background_name = background_name
        self.num_classes = len(class_names)

        self.gt_distribution = Counter()
        self.pred_distribution = Counter()

        self.cm_true = []
        self.cm_pred = []

        self.per_class_intersection = np.zeros(self.num_classes, dtype=np.float64)
        self.per_class_union = np.zeros(self.num_classes, dtype=np.float64)
        self.per_class_gt_pixels = np.zeros(self.num_classes, dtype=np.float64)
        self.per_class_pred_pixels = np.zeros(self.num_classes, dtype=np.float64)

        self.instance_iou_scores = defaultdict(list)
        self.instance_dice_scores = defaultdict(list)

    def update(
        self,
        gt_instances: List[dict],
        pred_instances: List[dict],
        iou_thr: float,
    ) -> None:
        for gt in gt_instances:
            self.gt_distribution[gt["class_id"]] += 1
            cls = gt["class_id"]
            self.per_class_gt_pixels[cls] += gt["mask"].sum()

        for pred in pred_instances:
            self.pred_distribution[pred["class_id"]] += 1
            cls = pred["class_id"]
            self.per_class_pred_pixels[cls] += pred["mask"].sum()

        matches = greedy_match_instances(gt_instances, pred_instances, iou_thr=iou_thr)

        for gi, pi, _ in matches:
            if gi is not None and pi is not None:
                gt_cls = gt_instances[gi]["class_id"]
                pred_cls = pred_instances[pi]["class_id"]
                self.cm_true.append(gt_cls)
                self.cm_pred.append(pred_cls)

                if gt_cls == pred_cls:
                    gt_mask = gt_instances[gi]["mask"]
                    pred_mask = pred_instances[pi]["mask"]
                    inter = np.logical_and(gt_mask, pred_mask).sum()
                    union = np.logical_or(gt_mask, pred_mask).sum()

                    self.per_class_intersection[gt_cls] += inter
                    self.per_class_union[gt_cls] += union

                    self.instance_iou_scores[gt_cls].append(mask_iou(gt_mask, pred_mask))
                    self.instance_dice_scores[gt_cls].append(mask_dice(gt_mask, pred_mask))

            elif gi is not None and pi is None:
                gt_cls = gt_instances[gi]["class_id"]
                self.cm_true.append(gt_cls)
                self.cm_pred.append(self.num_classes)  # background FN

            elif gi is None and pi is not None:
                pred_cls = pred_instances[pi]["class_id"]
                self.cm_true.append(self.num_classes)  # background FP
                self.cm_pred.append(pred_cls)

    def finalize_tables(self) -> Tuple[np.ndarray, List[str], List[dict]]:
        labels = list(range(self.num_classes + 1))
        cm = confusion_matrix(self.cm_true, self.cm_pred, labels=labels)
        display_names = self.class_names + [self.background_name]

        rows = []
        for cls_id, cls_name in enumerate(self.class_names):
            inter = self.per_class_intersection[cls_id]
            union = self.per_class_union[cls_id]
            gt_px = self.per_class_gt_pixels[cls_id]
            pred_px = self.per_class_pred_pixels[cls_id]

            iou = inter / union if union > 0 else 0.0
            dice = (2 * inter) / (gt_px + pred_px) if (gt_px + pred_px) > 0 else 0.0

            rows.append(
                {
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "test_instances": int(self.gt_distribution[cls_id]),
                    "pred_instances": int(self.pred_distribution[cls_id]),
                    "pixel_iou": float(iou),
                    "pixel_dice": float(dice),
                    "instance_mean_iou": float(np.mean(self.instance_iou_scores[cls_id])) if self.instance_iou_scores[cls_id] else 0.0,
                    "instance_mean_dice": float(np.mean(self.instance_dice_scores[cls_id])) if self.instance_dice_scores[cls_id] else 0.0,
                }
            )

        return cm, display_names, rows


# ----------------------------
# Plotting
# ----------------------------

def style_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.titlesize": 18,
            "savefig.dpi": 300,
            "figure.dpi": 140,
        }
    )
    sns.set_theme(style="whitegrid", context="paper")


def plot_confusion_matrix(
    cm: np.ndarray,
    display_names: List[str],
    save_path: Path,
    normalize: bool = True,
) -> None:
    cm_plot = cm.astype(np.float64)
    if normalize:
        row_sums = cm_plot.sum(axis=1, keepdims=True)
        cm_plot = np.divide(cm_plot, row_sums, out=np.zeros_like(cm_plot), where=row_sums > 0)

    fig_w = max(9, 1.1 * len(display_names))
    fig_h = max(7, 0.9 * len(display_names))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=".2f" if normalize else ".0f",
        cmap="mako",
        square=True,
        cbar=True,
        linewidths=0.6,
        linecolor="white",
        xticklabels=display_names,
        yticklabels=display_names,
        ax=ax,
    )

    ax.set_title("Test-set Confusion Matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground-truth class")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(class_names: List[str], counts: List[int], save_path: Path, title: str) -> None:
    order = np.argsort(counts)[::-1]
    names_sorted = [class_names[i] for i in order]
    counts_sorted = [counts[i] for i in order]

    fig_w = max(10, 0.75 * len(class_names))
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    bars = ax.bar(names_sorted, counts_sorted)

    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of instances")
    ax.set_xticklabels(names_sorted, rotation=35, ha="right")

    for rect, value in zip(bars, counts_sorted):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height(),
            str(value),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_iou_dice_bars(rows: List[dict], save_path: Path) -> None:
    names = [r["class_name"] for r in rows]
    ious = [r["pixel_iou"] for r in rows]
    dices = [r["pixel_dice"] for r in rows]

    x = np.arange(len(names))
    width = 0.38

    fig_w = max(10, 0.75 * len(names))
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    ax.bar(x - width / 2, ious, width, label="IoU")
    ax.bar(x + width / 2, dices, width, label="Dice")

    ax.set_title("Per-class Segmentation Scores")
    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.legend(frameon=True)

    for i, (iou, dice) in enumerate(zip(ious, dices)):
        ax.text(i - width / 2, iou + 0.015, f"{iou:.3f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + width / 2, dice + 0.015, f"{dice:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-style evaluation for YOLO segmentation on the test split.")
    parser.add_argument("--weights", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--conf", type=float, default=0.25, help="Prediction confidence threshold")
    parser.add_argument("--iou-match", type=float, default=0.50, help="IoU threshold used for GT/pred instance matching")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--outdir", type=str, default="paper_eval_outputs")
    args = parser.parse_args()

    style_matplotlib()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    data_cfg = load_yaml(Path(args.data))
    class_names = names_from_data(data_cfg)
    images_dir = find_split_images(data_cfg, args.split)
    image_paths = list_images(images_dir)

    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    model = YOLO(args.weights)
    evaluator = SegmentationEvaluator(class_names=class_names)

    for idx, img_path in enumerate(image_paths, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Could not read image: {img_path}")
            continue

        result = model.predict(
            source=str(img_path),
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
            retina_masks=True,
            max_det=args.max_det,
        )[0]

        gt_instances = read_gt_instances(image_to_label_path(img_path), image_shape=img.shape)
        pred_instances = extract_pred_instances(result, conf=args.conf)
        evaluator.update(gt_instances, pred_instances, iou_thr=args.iou_match)

        if idx % 25 == 0 or idx == len(image_paths):
            print(f"[INFO] Processed {idx}/{len(image_paths)} images")

    cm, cm_names, rows = evaluator.finalize_tables()

    # Save figures
    plot_confusion_matrix(cm, cm_names, outdir / "confusion_matrix_normalized.png", normalize=True)
    plot_confusion_matrix(cm, cm_names, outdir / "confusion_matrix_counts.png", normalize=False)

    gt_counts = [int(evaluator.gt_distribution[i]) for i in range(len(class_names))]
    pred_counts = [int(evaluator.pred_distribution[i]) for i in range(len(class_names))]
    plot_class_distribution(class_names, gt_counts, outdir / "test_class_distribution.png", "Test-set Class Distribution")
    plot_class_distribution(class_names, pred_counts, outdir / "predicted_class_distribution.png", "Predicted Class Distribution")
    plot_iou_dice_bars(rows, outdir / "per_class_iou_dice.png")

    # Save metrics
    macro_iou = float(np.mean([r["pixel_iou"] for r in rows])) if rows else 0.0
    macro_dice = float(np.mean([r["pixel_dice"] for r in rows])) if rows else 0.0
    weighted_iou = float(
        np.average(
            [r["pixel_iou"] for r in rows],
            weights=[max(r["test_instances"], 1) for r in rows],
        )
    ) if rows else 0.0
    weighted_dice = float(
        np.average(
            [r["pixel_dice"] for r in rows],
            weights=[max(r["test_instances"], 1) for r in rows],
        )
    ) if rows else 0.0

    summary = {
        "weights": str(Path(args.weights).resolve()),
        "data": str(Path(args.data).resolve()),
        "split": args.split,
        "num_images": len(image_paths),
        "conf_threshold": args.conf,
        "iou_match_threshold": args.iou_match,
        "class_names": class_names,
        "macro_pixel_iou": macro_iou,
        "macro_pixel_dice": macro_dice,
        "weighted_pixel_iou": weighted_iou,
        "weighted_pixel_dice": weighted_dice,
        "per_class": rows,
        "outputs": {
            "confusion_matrix_normalized": str((outdir / "confusion_matrix_normalized.png").resolve()),
            "confusion_matrix_counts": str((outdir / "confusion_matrix_counts.png").resolve()),
            "test_class_distribution": str((outdir / "test_class_distribution.png").resolve()),
            "predicted_class_distribution": str((outdir / "predicted_class_distribution.png").resolve()),
            "per_class_iou_dice": str((outdir / "per_class_iou_dice.png").resolve()),
        },
    }

    with open(outdir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save CSV-like table
    csv_path = outdir / "per_class_metrics.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        header = [
            "class_id",
            "class_name",
            "test_instances",
            "pred_instances",
            "pixel_iou",
            "pixel_dice",
            "instance_mean_iou",
            "instance_mean_dice",
        ]
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(
                ",".join(
                    [
                        str(r["class_id"]),
                        str(r["class_name"]),
                        str(r["test_instances"]),
                        str(r["pred_instances"]),
                        f"{r['pixel_iou']:.6f}",
                        f"{r['pixel_dice']:.6f}",
                        f"{r['instance_mean_iou']:.6f}",
                        f"{r['instance_mean_dice']:.6f}",
                    ]
                ) + "\n"
            )

    print("\n[INFO] Evaluation complete.")
    print(f"[INFO] Outputs saved to: {outdir.resolve()}")
    print(f"[INFO] Macro IoU  : {macro_iou:.4f}")
    print(f"[INFO] Macro Dice : {macro_dice:.4f}")
    print(f"[INFO] Weighted IoU  : {weighted_iou:.4f}")
    print(f"[INFO] Weighted Dice : {weighted_dice:.4f}")


if __name__ == "__main__":
    main()
