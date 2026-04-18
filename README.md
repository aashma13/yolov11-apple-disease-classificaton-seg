## Integrating Bootstrap Reliability and Grad-CAM Interpretability into Apple Disease Classification and Lesion Segmentation

### Author
**Aashma Dahal**  
Youngstown State University

### Faculty Coach
**Dr. Feng “George” Yu**

### Date
**April 2026**

---

## 1. Project Overview

This project presents a deep learning-based approach for apple disease classification using the **YOLOv11s classification model**, with additional emphasis on **prediction reliability** and **model interpretability**. In addition to the main classification study, a smaller lesion segmentation experiment was also conducted using **YOLOv11l-seg** to explore the feasibility of localizing diseased regions on apple fruit images.

The central contribution of this work is not only high classification performance, but also the integration of:

- **Bootstrap sampling** for robustness and reliability analysis
- **Grad-CAM** for visual interpretability
- A minor **segmentation study** for lesion localization

This combination supports a more trustworthy computer vision pipeline for agricultural disease recognition.

---

## 2. Motivation

Apple diseases can spread quickly and significantly reduce fruit quality and crop yield. Early identification is essential for timely treatment and improved disease management. Traditional manual inspection, however, is often:

- labor-intensive
- time-consuming
- inconsistent across observers
- difficult to scale in real-world agricultural settings

Recent deep learning systems have improved disease detection performance, but many existing studies focus primarily on predictive accuracy without providing adequate evidence of model reliability or interpretability. This project addresses those gaps by combining strong classification performance with statistical stability analysis and attention-based explanation.

---

## 3. Problem Statement

Existing apple disease detection models frequently report high accuracy, but they often do not address two important concerns:

1. **Reliability** — whether the reported performance remains stable under repeated sampling
2. **Interpretability** — whether the model is making decisions based on meaningful lesion regions rather than irrelevant image artifacts

Accordingly, this work aims to build a system that is:

- accurate
- efficient
- statistically reliable
- visually interpretable

---

## 4. Dataset Description

The study used the **Apple Disease Dataset (Version 3)** published on **Mendeley Data** on **November 3, 2025**. The dataset was collected in **Poncokusumo, Malang Regency, Indonesia** under natural orchard conditions and includes images acquired using multiple devices, including DSLR and smartphone cameras.

### 4.1 Classes

The dataset was organized into five classes:

- Anthracnose
- Black Pox
- Black Rot
- Healthy
- Powdery Mildew

### 4.2 Image Characteristics

- **Format:** JPG
- **Environment:** Real orchard conditions
- **Capture devices:** DSLR, Realme 9 Pro, iPhone 13, Samsung Galaxy S series
- **License:** CC BY 4.0

The class distribution was kept balanced across training, validation, and testing, which is important for reducing bias and enabling more reliable evaluation.

---

## 5. Methodology

### 5.1 Data Split

The dataset was partitioned as follows:

- **70%** training
- **15%** validation
- **15%** testing

### 5.2 Preprocessing

All images were resized to **224 × 224 pixels** before being used as model input.

### 5.3 Classification Model

The main model used for disease classification was the pretrained **YOLOv11s classification model**. The model was trained for **30 epochs** with the following training settings:

- **Optimizer:** AdamW
- **Batch size:** 64
- **Learning rate:** 0.001
- **Dropout:** 0.15

After training, the best-performing checkpoint was selected for final evaluation.

---

## 6. Experimental Design and Evaluation

The trained classifier was evaluated on the held-out test set using standard multiclass classification metrics. The evaluation framework included:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Weighted Precision, Recall, and F1-score
- Classification report
- Confusion matrix
- Bootstrap sampling
- Grad-CAM analysis

This evaluation design was intended to measure not only predictive correctness, but also class balance, statistical stability, and interpretability.

---

## 7. Classification Results

The YOLOv11s classifier achieved strong performance on the test set.

### 7.1 Aggregate Metrics

- **Test Accuracy:** 96.77%
- **Macro Precision:** 96.77%
- **Macro Recall:** 96.77%
- **Macro F1-score:** 96.76%
- **Weighted F1-score:** 96.76%

These results indicate that the classifier performed at a high level overall and also maintained balanced behavior across the five categories, as evidenced by the close agreement between macro and weighted metrics.

### 7.2 Class-Level Trends

The strongest-performing classes were:

- Black Pox
- Healthy
- Powdery Mildew

The principal area of confusion occurred between:

- Anthracnose
- Black Rot

This misclassification likely reflects visual similarity between these lesion patterns. The confusion matrix showed that most predictions remained concentrated along the diagonal, indicating that the majority of test samples were classified correctly.

---

## 8. Reliability Analysis via Bootstrap Sampling

A distinguishing feature of the project is the use of **bootstrap sampling** to verify that classification performance was not overly dependent on a single fixed test split. The model was repeatedly evaluated on resampled test subsets to estimate the stability of the reported accuracy.

### 8.1 Bootstrap Results

- **Number of bootstrap samples:** 2000
- **Point estimate accuracy:** 0.9677
- **Mean bootstrap accuracy:** 0.9675
- **95% confidence interval:** 0.9579 – 0.9767
- **Standard deviation:** 0.0049

### 8.2 Interpretation

These results suggest that model performance is both **stable** and **robust**. The close agreement between the point estimate and mean bootstrap accuracy, together with the narrow confidence interval and low standard deviation, indicates that the observed classification quality is unlikely to be an artifact of one favorable test partition.

---

## 9. Interpretability via Grad-CAM

To address the black-box nature of deep learning models, Grad-CAM was applied to test images in order to visualize the regions that contributed most strongly to the classifier’s predictions.

The Grad-CAM heatmaps consistently highlighted lesion-relevant areas such as:

- dark spots
- infected patches
- discolored regions
- visible lesion zones on the fruit surface

This is important because it shows that the classifier was primarily focusing on disease-related visual features rather than background content or irrelevant structures in the image. As a result, Grad-CAM strengthened the interpretability and trustworthiness of the proposed system.

---

## 10. Lesion Segmentation: Minor Supporting Study

In addition to classification, the project included a smaller segmentation experiment to explore whether YOLO could also localize diseased regions on apple fruit images. This was a **minor study**, not the central focus of the report.

### 10.1 Segmentation Workflow

The segmentation dataset was prepared in **Roboflow**, using a mixture of:

- manual annotation
- SAM3-assisted mask generation
- manual correction

After preprocessing and augmentation, the data was exported in **COCO segmentation format** and then converted to **YOLO format**. A pretrained **YOLOv11l-seg** model was used for training.

### 10.2 Segmentation Results

- **Best validation:** around epoch 9
- **Early stopping:** epoch 14
- **Test Box mAP50:** 0.804
- **Test Box mAP50-95:** 0.758
- **Test Mask mAP50:** 0.804
- **Test Mask mAP50-95:** 0.758

### 10.3 Interpretation

These results show that the segmentation model was capable of identifying and localizing diseased regions with reasonable performance. However, segmentation required substantial manual annotation effort and was conducted on a relatively small dataset, with approximately 100 images per class. For that reason, segmentation remained exploratory rather than a primary contribution.

---

## 11. Discussion

The overall findings indicate that the proposed YOLOv11-based pipeline is effective for apple disease image classification and provides stronger evidence of trustworthiness than an accuracy-only study. Several points are especially notable:

- The classifier achieved **96.77% test accuracy**
- Macro and weighted metrics were nearly identical, indicating balanced performance across classes
- Bootstrap analysis demonstrated low variability and strong stability
- Grad-CAM confirmed that predictions were driven by lesion-relevant image regions
- A secondary segmentation experiment showed that lesion localization is feasible, though more data and annotation effort would be needed for a more comprehensive study

Among the five disease classes, **Black Pox** appeared to be the easiest to classify, while **Black Rot** was comparatively more difficult. The most persistent confusion occurred between Anthracnose and Black Rot, likely due to overlapping visual characteristics.

---

## 12. Limitations

Several limitations should be acknowledged:

- The segmentation study was performed on a relatively small dataset
- Segmentation required substantial annotation effort
- The work focuses on a five-class setup and may require further validation for broader agricultural deployment
- Confusion remained between visually similar disease classes, particularly Anthracnose and Black Rot

These limitations do not undermine the main classification results, but they indicate useful directions for future work.

---

## 13. Conclusion

This project developed a **YOLOv11s-based apple disease classification system** capable of distinguishing five apple lesion classes with strong overall performance. The final classifier achieved **96.77% accuracy**, with similarly high precision, recall, and F1-scores.

More importantly, the project extended beyond conventional performance reporting by incorporating:

- **Bootstrap sampling** to evaluate stability and reliability
- **Grad-CAM** to verify lesion-focused model attention
- A minor **YOLOv11 segmentation study** to explore lesion localization

Taken together, the results suggest that the proposed system is not only accurate, but also statistically robust and more interpretable than a typical black-box classifier. This makes it a promising approach for supporting early apple disease identification in real-world agricultural applications.


