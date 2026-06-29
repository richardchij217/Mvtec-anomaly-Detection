# Mvtec-anomaly-Detection
# 🔍 Industrial Image Anomaly Detection using Deep Learning Backbones

> Comparative evaluation of ResNet18, ResNet50, DenseNet121, and EfficientNet-B5 for unsupervised industrial defect detection using the MVTec AD dataset.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Anomaly%20Detection-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

# Project Overview

Industrial quality inspection is a critical component of modern manufacturing, where undetected defects can result in equipment failure, product recalls, and significant financial losses.

This project develops an **unsupervised anomaly detection framework** capable of identifying manufacturing defects without requiring defective images during training.

The system is based on the **PaDiM (Patch Distribution Modeling)** framework and extends it by comparing multiple deep convolutional neural network backbones while investigating the effects of:

- Multiple CNN feature extractors
- Principal Component Analysis (PCA)
- Mahalanobis Distance anomaly scoring
- Category-specific hyperparameter tuning
- Synthetic anomaly generation
- Domain transfer experiments

The objective was to determine which backbone architecture provides the highest anomaly detection performance across industrial inspection tasks.

---

# Objectives

- Compare four state-of-the-art CNN backbones
- Build a modular PaDiM anomaly detection pipeline
- Evaluate image-level and pixel-level detection performance
- Study the effect of PCA dimensionality reduction
- Investigate synthetic anomaly generation for low-sample classes
- Benchmark models using the MVTec AD dataset

---

# Dataset

**Dataset:** MVTec AD

The benchmark contains:

- 15 industrial object and texture categories
- High-resolution RGB images
- Pixel-level ground truth masks
- Normal training images only
- Defective images reserved for testing

Categories include:

- Bottle
- Cable
- Capsule
- Carpet
- Grid
- Hazelnut
- Leather
- Metal Nut
- Pill
- Screw
- Tile
- Toothbrush
- Transistor
- Wood
- Zipper

---

# Models Evaluated

| Backbone | Purpose |
|-----------|----------|
| ResNet18 | Lightweight baseline |
| ResNet50 | Deep residual network |
| DenseNet121 | Dense feature propagation |
| EfficientNet-B5 | Compound-scaled CNN |

---

# Methodology

The complete pipeline consists of:

```
Training Images
        │
        ▼
 Image Preprocessing
        │
        ▼
 CNN Feature Extraction
        │
        ▼
 Random Channel Selection
        │
        ▼
 PCA (Optional)
        │
        ▼
 Gaussian Distribution Modeling
        │
        ▼
 Mahalanobis Distance
        │
        ▼
 Pixel Anomaly Map
        │
        ▼
 Image-Level Score
        │
        ▼
 Performance Evaluation
```

---

# Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Scikit-learn
- OpenCV
- Matplotlib
- PIL
- PCA
- Mahalanobis Distance

---

# Evaluation Metrics

Performance was evaluated using:

- AUROC
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Pixel-level Localization
- Heatmap Visualization

---

# Results

## Overall Performance

| Model | Overall Performance |
|---------|----------------|
|  EfficientNet-B5 | Best |
|  ResNet50 | Excellent |
|  DenseNet121 | Very Good |
| ResNet18 | Baseline |

### Highlights

- EfficientNet-B5 achieved the highest overall AUROC across most categories.
- ResNet50 provided consistently strong performance across both texture and object classes.
- DenseNet121 performed particularly well on fine-grained textures.
- ResNet18 was computationally efficient but less accurate on complex defects.

---

# Qualitative Results

The framework successfully generated:

- Heatmaps
- Predicted anomaly masks
- Ground-truth comparisons
- Pixel-level localization

allowing visual interpretation of defect detection performance.

---

# Industrial Applications

This research has applications in:

- Manufacturing Quality Control
- Oil & Gas Equipment Inspection
- Pipeline Monitoring
- Pharmaceutical Inspection
- Smart Manufacturing
- Predictive Maintenance
- Infrastructure Monitoring
- Automated Visual Inspection

---

# Future Improvements

Potential future work includes:

- Vision Transformers (ViT)
- Swin Transformer
- Mamba Vision Models
- PatchCore comparison
- GLASS implementation
- Real-time deployment
- Edge AI optimization
- GAN-based synthetic anomaly generation

---

# Repository Structure

```
Mvtec-anomaly-Detection
│
├── notebooks/
├── models/
├── images/
├── reports/
├── presentation/
├── results/
└── README.md
```

---

# References

- PaDiM (ICPR 2021)
- MVTec AD Dataset
- EfficientNet
- DenseNet
- ResNet
- PatchCore
- SPADE
- GLASS

---

# Author

**Richard Chijioke**

Master of Data Science  
Memorial University of Newfoundland

Mechanical Engineer | Data Scientist | Computer Vision | Machine Learning | Industrial AI

---

⭐ If you found this project interesting, consider giving this repository a star.
