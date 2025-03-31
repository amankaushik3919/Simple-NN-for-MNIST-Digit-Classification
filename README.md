# Simple-NN-for-MNIST-Digit-Classification
This project uses a Neural Network (NN) to classify handwritten digits from the MNIST dataset (42,000 images). The model is built with TensorFlow/Keras, normalizes pixel values, and predicts digits (0-9). It includes preprocessing, training, and evaluation steps with accuracy visualization. 🚀

---

# 🧠 Simple Neural Network for MNIST Digit Classification

This project uses a **Neural Network (NN)** to classify handwritten digits from the **MNIST dataset** (42,000 images). The model is built with **TensorFlow/Keras**, normalizes pixel values, and predicts digits (0–9). It includes preprocessing, training, evaluation, and accuracy visualization steps. ✨📊

---

## 📝 Project Overview

This project aims to build a **Neural Network (NN)** capable of recognizing handwritten digits from the **MNIST dataset**. Using **TensorFlow/Keras**, the model achieves high accuracy in classifying digits after being trained on 42,000 labeled images. 🖊️🔍

---

## 🌟 Key Features

- ✅ Preprocessing and normalization of image data
- 🧠 Simple **Neural Network (NN)** for classification
- 🏋️ Training and evaluation of the model
- 📈 Accuracy and loss visualization
- 🔮 Prediction on new handwritten digits

---

## 💻 Technologies Used

- **Python** 🐍
- **TensorFlow/Keras** 🧠
- **NumPy & Pandas** 📊
- **Matplotlib & Seaborn** 📉

---

## 📂 Dataset

The project uses the **MNIST dataset**, which consists of **28x28 grayscale images** of handwritten digits (0–9). The dataset contains:

- **42,000 training images**
- **28,000 test images**

🔗 [Dataset Link](https://www.kaggle.com/competitions/digit-recognizer/data)

---

## 🚀 Installation & Usage

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/mnist-handwritten-digit-recognition.git
cd mnist-handwritten-digit-recognition
```

### 2️⃣ Install Dependencies

```bash
pip install tensorflow numpy pandas matplotlib seaborn
```

### 3️⃣ Run the Model

```python
python train.py
```

---

## 📊 Dataset Analysis

The dataset comprises **785 columns**:
- The first column (`label`) represents the digit (0–9).
- The remaining **784 columns** correspond to pixel values of the 28x28 grayscale image.

Key statistical insights:
- **Count**: 42,000 entries.
- **Mean Pixel Values**: Most pixels have a mean value of 0, with some variability in edge pixels.
- **Standard Deviation**: Highlights variation in pixel intensity across images.
- **Min/Max Values**: Pixel values range from 0 to 254.

Example of the dataset structure:

| label | pixel0 | pixel1 | ... | pixel783 |
|-------|--------|--------|-----|----------|
| 1     | 0      | 0      | ... | 0        |
| 0     | 0      | 0      | ... | 0        |

---

## 🧩 Model Architecture

The Neural Network consists of the following layers:

- **Dense Layer 1**: Output Shape = `(None, 128)`, Parameters = 100,480
- **Dense Layer 2**: Output Shape = `(None, 64)`, Parameters = 8,256
- **Dense Layer 3 (Output Layer)**: Output Shape = `(None, 10)`, Parameters = 650

**Total Parameters**: 109,386  
**Trainable Parameters**: 109,386  
**Non-Trainable Parameters**: 0

---

## 📈 Training Results

The model was trained for **10 epochs** with the following performance metrics:

| Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
|-------|---------------|-------------------|-----------------|---------------------|
| 1     | 0.3007        | 0.9139            | 0.1618          | 0.9526              |
| 2     | 0.1262        | 0.9629            | 0.1539          | 0.9495              |
| ...   | ...           | ...               | ...             | ...                 |
| 10    | 0.0184        | 0.9937            | 0.1352          | 0.9681              |

**Final Validation Accuracy**: **96.81%** 🎉

---

## 🎯 Model Evaluation

- **Normalization**: Pixel values were scaled between 0 and 1 to improve convergence. 🔄
- **Optimizer**: Adam optimizer was used for efficient training. ⚡
- **Loss Function**: Categorical cross-entropy loss was applied to handle multi-class classification. 📉
- **Performance Visualization**: Accuracy and loss curves were plotted to monitor training progress. 📊

---

## 🔮 Prediction Example

The trained model successfully predicted the label of a test image as follows:

- **Predicted Label**: 0 ✅

---

## 🤝 Contributing

Contributions to this project are welcome! Feel free to **fork** this repository, make improvements, and submit a **pull request**. 🚀

---

## 📜 License

This project is **open-source** and available under a permissive license. 🌍

---

🌟 **Follow me for more projects!** 🌟

---
