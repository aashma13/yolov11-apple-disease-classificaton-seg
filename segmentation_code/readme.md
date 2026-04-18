# YOLOv11 for Robust Apple Disease Classification and Lesion Segmentation

## Project Summary

This project is about detecting apple diseases from fruit images and also marking the infected area on the fruit.  
For this work, I used **YOLOv11 segmentation**.

The dataset was first collected from real apple orchards in Indonesia. Then the images were uploaded to **Roboflow** for annotation, preprocessing, augmentation, and export. Part of the annotation work was helped by **SAM3**, and the remaining annotation and correction work was done manually.

After that, the dataset was exported in **COCO segmentation** format and trained using **Ultralytics YOLOv11**.

This README explains the full process in simple words.

---

## Why This Project Matters

Apple diseases reduce fruit quality and production. In real orchards, it takes time and effort to check fruits manually. A computer vision model can help identify disease faster.

This project does two things:

1. **Classification** – predict which disease class is present
2. **Segmentation** – show the diseased region on the apple image

This is useful because it not only says the disease name, but also shows where the disease is.

---

## Data Source

The original image source is:

**Apple Disease Dataset**  
**Version 1**  
**Published on:** November 3, 2025

### Source Information

- **Collection place:** Orchards in **Poncokusumo, Malang Regency, Indonesia**
- **Capture condition:** Natural lighting
- **Devices used:**
  - DSLR cameras
  - Realme 9 Pro
  - iPhone 13
  - Samsung Galaxy S series
- **Image format:** JPG
- **License:** CC BY 4.0

### Citation

If you want to cite the source dataset:

**FEBRIANTONO, ALDIKI (2025), "Apple Disease Dataset," Mendeley Data, V1**  
**DOI:** `10.17632/9zgkwwv9j8.1`

---

## Original Classes in the Source Dataset

The original source dataset contains 6 categories:

- Anthracnose
- Black Pox
- Black Rot
- Codling Moth
- Healthy
- Powdery Mildew

---

## Classes Used in This Project

For this project, I used 5 classes:

- Anthracnose
- Black Pox
- Black Rot
- Healthy
- Powdery Mildew

> Note: The original source dataset includes **Codling Moth**, but it was not used in the final training version described here.

---

## Dataset Preparation Workflow

The full dataset preparation process was:

1. Collect apple fruit images from orchards
2. Upload images to Roboflow
3. Annotate disease regions
4. Use **SAM3** to help generate some segmentation masks
5. Manually correct and complete the remaining annotations
6. Apply Roboflow preprocessing
7. Apply Roboflow augmentation
8. Export the final dataset in **COCO segmentation** format
9. Convert COCO format to YOLO segmentation format
10. Train YOLOv11 segmentation model

So this dataset is **not only auto-generated**. It includes **manual human work** to improve annotation quality.

---

## Annotation Method

The annotation process used a mixed approach:

### 1. Manual Annotation
Some images were annotated manually from the beginning.

### 2. SAM3-Assisted Annotation
For some images, **SAM3** was used to help create segmentation masks faster.

### 3. Manual Correction
After SAM3, the masks were checked and corrected manually. This was important because automatic masks are not always perfect.

Manual correction included:

- fixing wrong boundaries
- improving object shape
- correcting class labels
- removing poor masks
- keeping annotation quality more consistent

This means the final annotations are **human-reviewed**.

---

## Current Manual Annotation Counts

Below are the counts from the manually annotated disease portion you shared:

| Class Name | Annotation Count |
|------------|------------------:|
| Black Pox | 125 |
| Anthracnose | 84 |
| Powdery Mildew | 32 |
| Black Rot | 19 |
| **Total** | **260** |

---

## Roboflow Dataset Version

The dataset version used for training is:

**SegmentAppledisease - v5 New All Dataset**

This version was prepared in Roboflow and exported later for model training.

---

## Dataset Before Augmentation

Before Roboflow preprocessing and augmentation, the uploaded dataset had:

- **Source images:** 1,405
- **Classes:** 5
- **Unannotated images:** 0

### Initial Split

- **Training:** about 1,040 images
- **Validation:** 183 images
- **Testing:** 182 images

---

## Roboflow Preprocessing

The following preprocessing was applied in Roboflow:

- **Tile:** 2 rows × 2 columns
- **Auto-Orient:** Applied
- **Isolate Objects:** Applied
- **Static Crop:**
  - Horizontal region: 25% to 75%
  - Vertical region: 28% to 77%
- **Resize:** Fit within 640 × 640
- **Auto-Adjust Contrast:** Histogram Equalization

### Simple Meaning

These steps help make the images more suitable for training:

- **Tile** splits larger images into smaller parts
- **Auto-Orient** fixes image direction
- **Isolate Objects** helps focus on important object area
- **Static Crop** keeps only a selected region
- **Resize** makes image size consistent
- **Contrast adjustment** makes lesions easier to see

---

## Roboflow Augmentation

The following augmentation methods were used:

- Horizontal flip
- Vertical flip
- 90° rotation:
  - clockwise
  - counter-clockwise
  - upside down
- Random rotation between **-12° and +12°**
- Random saturation change between **-22% and +22%**
- Random brightness change between **-12% and +12%**
- Random blur up to **1.7 px**
- Bounding box noise up to **0.1% of pixels**

### Output Per Training Example

- **3 outputs per training image**

### Simple Meaning

Augmentation increases the variety of the training data.  
This helps the model learn better from different image directions, brightness levels, and small variations.

---

## Final Dataset Size After Preprocessing and Augmentation

After Roboflow preprocessing and augmentation, the final dataset became:

- **Total images:** 15,304
- **Train:** 13,500 images
- **Valid:** 940 images
- **Test:** 864 images

So the final dataset is much larger than the original uploaded dataset.

---

## Final Class Distribution After Export

From the export and conversion logs, the class counts are:

### Train
- Anthracnose: 2,580
- Black Pox: 2,772
- Black Rot: 2,760
- Healthy: 2,532
- Powdery Mildew: 2,856

### Valid
- Anthracnose: 232
- Black Pox: 232
- Black Rot: 184
- Healthy: 112
- Powdery Mildew: 176

### Test
- Anthracnose: 207
- Black Pox: 188
- Black Rot: 144
- Healthy: 132
- Powdery Mildew: 192

---

## Important Note About Missing Distribution After Augmentation

After preprocessing and augmentation, a small number of images in validation and test no longer had any labeled object.

This happened because operations like:

- tiling
- static crop
- isolate objects

can create some image pieces where the disease area is no longer visible.

### Export Summary

- **Train:** 13,500 images, 13,500 annotations, 0 empty images
- **Valid:** 940 images, 936 annotations, 4 empty images
- **Test:** 864 images, 863 annotations, 2 empty images

This means:

- **4 validation images** became background-only
- **2 test images** became background-only

So the distribution is not actually lost. It is just that a few augmented images do not contain any annotation after cropping/tiling.

---

## Why This Dataset Is Strong

This dataset is useful because it was collected in **real orchard conditions**, not only in clean laboratory settings.

That means the images contain real-world variation such as:

- different lighting
- different mobile phones and cameras
- different angles
- different fruit sizes
- different disease appearance
- background clutter
- natural orchard environment

Because of this, the trained model can be more useful in practical field conditions.

---

## Model Used

This project uses:

- **YOLOv11 segmentation**
- Pretrained model: `yolo11l-seg.pt`

YOLOv11 is used because it is fast and strong for detection and segmentation tasks.

---

## What the Training Script Does

The training script does the following:

1. Reads COCO annotation files from:
   - `train/_annotations.coco.json`
   - `valid/_annotations.coco.json`
   - `test/_annotations.coco.json`
2. Converts COCO polygons into YOLO segmentation labels
3. Organizes images and labels into YOLO folder format
4. Creates `data.yaml`
5. Trains the YOLOv11 segmentation model
6. Validates the best model on validation set
7. Tests the best model on test set
8. Saves summary metrics into a JSON file

---

## Folder Structure

Expected project structure:

```text
.
├── main.py
├── README.md
├── SegmentAppledisease-5/
│   ├── train/
│   │   ├── _annotations.coco.json
│   │   └── ...
│   ├── valid/
│   │   ├── _annotations.coco.json
│   │   └── ...
│   └── test/
│       ├── _annotations.coco.json
│       └── ...
└── runs/
````

After conversion, the structure becomes:

```text
SegmentAppledisease-5/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

---

## Requirements

You need the following:

* Python 3.10 or higher
* `uv`
* PyTorch
* Ultralytics
* PyYAML

---

## Install `uv`

### Linux / macOS

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Or install with pip

```bash
pip install uv
```

Check version:

```bash
uv --version
```

---

## Create Environment Using `uv`

Create a virtual environment:

```bash
uv venv
```

Activate on Linux/macOS:

```bash
source .venv/bin/activate
```

Activate on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install main packages:

```bash
uv pip install ultralytics pyyaml
```

---

## Install PyTorch

For CUDA 12.1:

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For CPU only:

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Choose the correct version based on your machine.

---

## Training Command

Run the training with:

```bash
uv run main.py \
  --dataset-root ./SegmentAppledisease-5 \
  --model yolo11l-seg.pt \
  --imgsz 640 \
  --epochs 30 \
  --batch 8 \
  --device 5 \
  --project ./runs/segment/apple_disease_seg \
  --name yolo11l_seg_30 \
  --workers 2 \
  --patience 5
```

---

## Meaning of Training Arguments

* `--dataset-root` → dataset folder
* `--model` → pretrained YOLOv11 segmentation model
* `--imgsz` → image input size
* `--epochs` → total training epochs
* `--batch` → batch size
* `--device` → GPU number
* `--project` → output folder
* `--name` → experiment name
* `--workers` → number of dataloader workers
* `--patience` → early stopping value

---

## Training Output

Training results will be saved in:

```text
./runs/segment/apple_disease_seg/yolo11l_seg_30
```

Important files include:

* `weights/best.pt`
* `weights/last.pt`
* `metrics_summary.json`
* training plots
* validation outputs
* prediction results

---

## Training Notes

* The dataset is exported from Roboflow in **COCO segmentation** format.
* The script converts it into **YOLO segmentation** format automatically.
* Extra online augmentation in Ultralytics is turned off because Roboflow augmentation was already applied.
* Early stopping is used with `--patience 5`.

---

## Short Result Summary

From the shared training logs:

* Best validation performance was reached around **epoch 9**
* Training stopped early at **epoch 14**
* Best model was saved as `best.pt`

### Best Validation Result

* Box mAP50: about **0.751**
* Box mAP50-95: about **0.713**
* Mask mAP50: about **0.751**
* Mask mAP50-95: about **0.717**

### Test Result

* Box mAP50: about **0.804**
* Box mAP50-95: about **0.758**
* Mask mAP50: about **0.804**
* Mask mAP50-95: about **0.758**

These values show that the model learned useful lesion segmentation patterns from the dataset.

---

## Conclusion

This project presents a full practical pipeline for apple disease analysis using **YOLOv11 segmentation**.

The work includes:

* real orchard image collection
* Roboflow dataset management
* SAM3-assisted annotation
* manual annotation and correction
* preprocessing and augmentation in Roboflow
* COCO export
* conversion to YOLO format
* training and evaluation with YOLOv11

Because the dataset was built from real field images and carefully reviewed annotations, it is suitable for research on robust disease detection and lesion segmentation in agriculture.

---

## Acknowledgment

The source data comes from the Apple Disease Dataset published in Mendeley Data.
The processed training version was prepared using Roboflow and trained using Ultralytics YOLOv11.
