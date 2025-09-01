# PneumoXplain-Explainable-Deep-Learning-for-Pneumonia-Detection-from-Chest-X-Rays
PneumoXplain is a deep learning project that detects Pneumonia vs Normal from chest X-rays using DenseNet121. The model is made explainable with Grad-CAM heatmaps, highlighting lung regions that influenced predictions. Designed for trustworthy AI in healthcare, it provides visual insights alongside high classification accuracy.

* Dataset
Source: Kaggle Chest X-ray Pneumonia Dataset (https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data)
Structure:
chest_xray/
├── train/ (Normal / Pneumonia)
├── val/   (Normal / Pneumonia)
├── test/  (Normal / Pneumonia)

* Features
Binary classification: Normal vs. Pneumonia
Pretrained DenseNet121 backbone for high accuracy
Grad-CAM heatmaps for explainability
ROC curve & AUC metrics for performance evaluation
Code to visualize and save results

* Training Strategy
Step 1: Freeze backbone, train classification head
Step 2: Fine-tune with low learning rate
Callbacks: EarlyStopping, ReduceLROnPlateau
