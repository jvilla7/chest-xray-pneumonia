# Chest X-Ray Pneumonia Detection

## 📌 Project Overview
This project uses deep learning to classify chest X-ray images as either **Normal** or **Pneumonia**.  
We implemented a transfer learning approach using a pretrained convolutional neural network.

---

## 📊 Dataset
We used the Chest X-Ray Pneumonia dataset from Kaggle:  
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

- Total images: ~5,800  
- Classes: Normal, Pneumonia  
- Pre-split into training, validation, and testing sets  

---

## 🧠 Methodology
- Image preprocessing (resizing, normalization, augmentation)  
- Transfer learning using a pretrained CNN (e.g., ResNet / MobileNet)  
- Model training and evaluation  
- Visualization using Grad-CAM  

---

## ⚙️ How to Run
1. Download the dataset from Kaggle  
2. Place it inside a folder named `data/`  
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
