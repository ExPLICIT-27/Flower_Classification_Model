# 🌸 Flower Classification Model using Convolutional Neural Networks  
**Author:** Nandu Mahesh  
_⭐ Do leave a star if you liked it!_

---

## 🧠 Model Capability

Classifies between the following flower types:
- Daisy
- Dandelion
- Rose
- Sunflower
- Tulip

---

## 📚 Brief Overview of CNNs

Convolutional Neural Networks (CNNs) are powerful models optimized for image, audio, and speech processing. They consist of three main types of layers:

### 🔹 Convolutional Layer
- Uses a **feature detector** (kernel) to extract spatial features from the image.
- **Number of filters** affects the output depth.
- **Stride** determines how far the filter moves over the output.

### 🔹 Pooling Layer
- Performs **downsampling** and **dimensionality reduction**.
- Helps reduce model complexity, increases efficiency, and minimizes overfitting risk.

### 🔹 Fully-Connected (FC) Layer
- Every node in this layer connects to all nodes in the previous layer.
- Performs **final classification** using extracted features.
- Typically uses a **softmax** activation function for outputting class probabilities.

---

## 🚀 Version 1: Custom CNN

### 📈 Preprocessing
- Performed **data augmentation**.
- Dataset split: `60%` training / `20%` validation / `20%` testing.

### 🏗️ Architecture
| Layer | Filters | Activation | Pooling |
|-------|---------|------------|---------|
| 1     | 32      | ReLU       | 2×2     |
| 2     | 64      | ReLU       | 2×2     |
| 3     | 64      | ReLU       | 2×2     |
| 4     | 128     | ReLU       | 2×2     |
| 5     | 256     | Linear     | -       |
| 6     | _Output classes_ | Softmax | -       |

### 🏋️‍♂️ Training Details
- **Epochs:** 50  
- **Early Stopping:** Enabled (patience = 3)  
- **Test Accuracy:** `65% - 70%`

---

## ⚡ Version 2: Transfer Learning with ResNet18

### 🧪 Enhancements
- Data augmentation with the **Albumentations** library.
- Used **ResNet18** with frozen base layers (fine-tuned classifier head).

### ⚙️ Training Details
- **Epochs:** 50  
- **Early Stopping:** Enabled (patience = 3)  
- **Test Accuracy:** `87% - 90%`

---

## ✅ Summary

| Version | Architecture      | Accuracy | Augmentation       | Notes                  |
|---------|-------------------|----------|---------------------|------------------------|
| V1      | Custom CNN        | 65–70%   | Basic transforms    | From scratch           |
| V2      | Transfer Learning | 87–90%   | Albumentations      | Uses ResNet18 backbone |

---

_💡 Future improvements might include experimenting with deeper models (e.g. ResNet50), using schedulers, or optimizing data pipelines._

