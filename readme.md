
---

# 🌸 Flower Classification Model using Convolutional Neural Networks

**Author:** Nandu Mahesh
*⭐ Do leave a star if you liked it!*

---

## 🧠 Model Capability

Classifies between the following flower types:

* Daisy
* Dandelion
* Rose
* Sunflower
* Tulip

---

## 📚 Brief Overview of CNNs

Convolutional Neural Networks (CNNs) are powerful models optimized for image, audio, and speech processing. They consist of three main types of layers:

### 🔹 Convolutional Layer

* Uses a **feature detector** (kernel) to extract spatial features from the image.
* **Number of filters** affects the output depth.
* **Stride** determines how far the filter moves over the output.

### 🔹 Pooling Layer

* Performs **downsampling** and **dimensionality reduction**.
* Helps reduce model complexity, increases efficiency, and minimizes overfitting risk.

### 🔹 Fully-Connected (FC) Layer

* Every node in this layer connects to all nodes in the previous layer.
* Performs **final classification** using extracted features.
* Typically uses a **softmax** activation function for outputting class probabilities.

---

## 🚀 Version 1: Custom CNN

### 📈 Preprocessing

* Performed **data augmentation**.
* Dataset split: `60%` training / `20%` validation / `20%` testing.

### 🏗️ Architecture

| Layer | Filters          | Activation | Pooling |
| ----- | ---------------- | ---------- | ------- |
| 1     | 32               | ReLU       | 2×2     |
| 2     | 64               | ReLU       | 2×2     |
| 3     | 64               | ReLU       | 2×2     |
| 4     | 128              | ReLU       | 2×2     |
| 5     | 256              | Linear     | -       |
| 6     | *Output classes* | Softmax    | -       |

### 🏋️‍♂️ Training Details

* **Epochs:** 50
* **Early Stopping:** Enabled (patience = 3)
* **Test Accuracy:** `65% - 70%`

---

## ⚡ Version 2: Transfer Learning with ResNet18

### 🧪 Enhancements

* Data augmentation with the **Albumentations** library.
* Used **ResNet18** with frozen base layers and a custom classifier head.

### ⚙️ Training Details

* **Epochs:** 50
* **Early Stopping:** Enabled (patience = 3)
* **Test Accuracy:** `87% - 90%`

---

## 🔥 Version 3: Advanced Transfer Learning with ResNet50

### 🧪 Improvements

* Switched to **ResNet50** for deeper feature extraction and better accuracy.
* Incorporated **more aggressive Albumentations** strategies.
* Fine-tuned final ResNet blocks + classifier head for performance.

### ⚙️ Training Details

* **Epochs:** 50
* **Early Stopping:** Enabled (patience = 4)
* **Test Accuracy:** `91% - 94%`
* **Mixed Precision:** Enabled (faster training on GPU)
* **Plotly-based visualization** integrated into UI

---

## ✅ Summary

| Version | Architecture | Accuracy   | Augmentation           | Notes                       |
| ------- | ------------ | ---------- | ---------------------- | --------------------------- |
| V1      | Custom CNN   | 65–70%     | Basic transforms       | From scratch                |
| V2      | ResNet18     | 87–90%     | Albumentations         | Transfer learning           |
| V3      | ResNet50     | **91–94%** | Advanced augmentations | Fine-tuned deep ResNet + UI |

---

### 🎯 Try It Live: [flowerclassifier.streamlit.app](https://flowerclassifier.streamlit.app/)

---


