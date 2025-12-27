# Chest X-Ray Pneumonia Classification with CNN
A medical imaging project using convolutional neural networks to classify chest X-rays as pneumonia or normal, with emphasis on model evaluation and clinical metrics.

## Project Overview
This project implements an end-to-end convolutional neural network (CNN) pipeline to classify chest X-ray images as **Pneumonia** or **Normal**.  
Beyond model training, the project focuses on **correct data handling**, **medical image preprocessing**, and **interpreting model performance using appropriate evaluation metrics** such as recall, precision, AUC, and confusion matrices.

## Dataset
The dataset is sourced from the Kaggle **Chest X-Ray Images (Pneumonia)** dataset.
It contains **5,863 X-Ray images (JPEG)** organized into a directory-based structure suitable for supervised learning.

Directory structure:

- 'train/' - training images
- 'val/': - validation images
- 'test/': - test images

Each split contains two subfolders:
- 'PNEUMONIA/'
- 'NORMAL/'

## Data Preprocessing Pipeline
The following preprocessing and input pipeline steps were implemented using 'tf.data':

- Construct dataset from directory structure using 'image_dataset_from_directory'
- Resize images to a fixed resolution '(224 × 224)'
- Batch and shuffle training data
- Visual inspection of raw images and labels
- Apply **data augmentation** (random rotation, zoom, and contrast) during training
- Normalize pixel values to '[0, 1]'
- Use **cache()** and **prefetch()** to optimize data loading performance

These steps ensure both correctness and efficiency when handling medical imaging data.

## Model Architecture
A baseline CNN model was implemented using TensorFlow / Keras with the following components:

  - **Conv2D layers** to learn local spatial features such as edges and textures
  - **MaxPooling2D layers** to reduce spatial dimensions and computational cost
  - **Flatten layer** to convert feature maps into a 1D representation
  - **Dense layers** for feature integration
  - **Dropout layer** for regularization and improved generalization

The final output layer uses **sigmoid activation**, suitable for binary classification (Pneumonia vs Normal).

## Model Training & Evaluation
- Optimizer: Adam  
- Loss function: Binary Crossentropy  
- Training epochs: 3

Evaluation metrics include:
- Accuracy
- AUC (Area Under ROC Curve)
- Precision
- Recall

Model performance was further analyzed using:
- **Confusion matrix** (TP / FP / FN / TN)
- **Accuracy and loss curves**
- **Visual inspection of predictions with confidence scores**

## Results
The model demonstrated strong performance on the validation set:

- Training and validation accuracy increased consistently across epochs
- Loss decreased steadily during training
- High recall values indicate good sensitivity for pneumonia detection
- Confusion matrix showed correct classification on validation samples

These results suggest the baseline CNN can generalize reasonably well to unseen chest X-ray images.

## What I Learned
Through this project, I gained hands-on experience with:

- End-to-end CNN pipelines for medical image classification
- Handling real-world medical image datasets with directory-based labels
- Building efficient input pipelines using 'tf.data'
- Applying and validating data augmentation strategies
- Interpreting classification metrics beyond accuracy
- Analyzing model behavior using confusion matrices and visual predictions

## Future Work
Potential improvements include:

- Using transfer learning (e.g., ResNet, EfficientNet)
- Addressing class imbalance explicitly
- Performing cross-validation
- Applying explainability methods such as Grad-CAM
