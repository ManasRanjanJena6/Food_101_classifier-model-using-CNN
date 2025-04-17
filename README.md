# Food_101_classifier-model-using-CNN

# 🍔 Food101 Classifier Project

This project is a Convolutional Neural Network (CNN)-based image classifier built using the [Food-101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/). The model is trained to classify images into 101 different food categories such as pizza, sushi, steak, and more.

## 🧠 Project Objective

To develop an image classification model using deep learning (CNNs) that can accurately recognize food categories from images using the Food-101 dataset.

## 📁 Dataset

- **Name**: Food-101

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- scikit-learn

## 🧱 Model Architecture

A simple CNN architecture:
- Conv2D → ReLU → MaxPooling
- Conv2D → ReLU → MaxPooling
- Flatten
- Dense → ReLU
- Dropout
- Dense → Softmax (output layer for 101 classes)

You can modify this architecture in `test.ipynb` (or equivalent script).

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/ManasRanjanJena6/food101_classifier_project.git
cd food101_classifier_project
