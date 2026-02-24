# Dermato-Vision-ResNet
A PyTorch-based computer vision pipeline utilizing a fine-tuned ResNet18 architecture for multi-class skin lesion classification on the HAM10000 dataset, featuring custom robustness analysis against photometric variations.
# Robust Skin Lesion Classification Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Integrated-ee4c2c)
![Computer Vision](https://img.shields.io/badge/Domain-Computer_Vision-purple)
![License](https://img.shields.io/badge/Code_License-MIT-green)

An Applied Computer Vision (ACV) case study demonstrating the fine-tuning of a deep residual network (ResNet18) for the automated diagnosis of pigmented skin lesions. 

A major focus of this project is **Robustness Analysis**, specifically evaluating how the model's predictive accuracy degrades or holds up under varying lighting and brightness conditions—a common real-world challenge in medical imaging.

## 📌 Project Overview
Automated classification of skin lesions can assist dermatologists in early skin cancer detection. This project utilizes the **HAM10000 ("Human Against Machine with 10000 training images")** dataset to classify lesions into 7 diagnostic categories, including Melanoma, Basal cell carcinoma, and Benign keratosis-like lesions.

### Key Contributions:
1. **Transfer Learning:** Fine-tuning pre-trained ResNet18 weights on the HAM10000 dataset using PyTorch and Torchvision.
2. **Medical Image Preprocessing:** Handling class imbalances and standardizing image inputs for the CNN.
3. **Photometric Robustness Testing:** A custom evaluation suite that programmatically alters the brightness (alpha/beta values) of test images to stress-test the model's reliability outside of ideal clinical lighting conditions.

## 📊 Dataset: HAM10000
The dataset consists of 10,015 dermatoscopic images. Due to size constraints, the dataset is not hosted in this repository. 
* You can download the dataset from [Kaggle's HAM10000 page](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000).
* Place the extracted `HAM10000_images` folder and metadata CSV inside the `/data` directory before running the notebook.

## 📁 Repository Structure
* `/notebooks`: Contains `Dermato-Vision-ResNet.ipynb` with the complete end-to-end pipeline (EDA, Training, Evaluation, and Robustness checks).
* `/src`: Modularized Python scripts for model definition and custom transformations.
* `/weights`: Contains the best `.pth` model checkpoints (if uploaded).

## 🚀 Getting Started

**1. Clone the repository:**
```bash
git clone [https://github.com/yourusername/Robust-Skin-Lesion-Classifier.git](https://github.com/yourusername/Robust-Skin-Lesion-Classifier.git)
cd Robust-Skin-Lesion-Classifier
