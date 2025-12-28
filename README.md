# Chest X-Ray Pneumonia Classification with Convolutional Neural Networks
A medical imaging project using convolutional neural networks to classify chest X-rays as pneumonia or normal, with emphasis on model evaluation and clinical metrics.

## Project Motivation
Pneumonia is a common and potentially life-threatening condition, particularly in vulnerable populations. Chest X-ray imaging is widely used as a diagnostic tool, but manual interpretation is time-consuming and subject to inter-observer variability.

This project explores whether a convolutional neural network (CNN) can learn discriminative visual patterns from chest X-ray images to perform **binary classification (Pneumonia vs Normal)**. Rather than focusing solely on accuracy, the project emphasizes **clinically meaningful evaluation metrics**, such as recall and AUC, due to the high cost of false negatives in medical diagnosis.

## Dataset & Task Definition
The dataset is from the Kaggle **Chest X-Ray Images (Pneumonia)** collection,  organized into `train`, `validation`, and `test` folders, each containing two categories: **PNEUMONIA** and **NORMAL**..

- **Task type:** Binary image classification
- **Input:** Chest X-ray images (JPEG)
- **Output:** Probability of pneumonia presence

The folder-based structure allows labels to be inferred directly from directory names, reducing the risk of label leakage.

## Methodology: End-to-End Pipeline
This project was designed as a complete image classification pipeline rather than a standalone model experiment:

1. Data inspection and folder structure verification  
2. Dataset creation using `image_dataset_from_directory`  
3. Visual sanity checks of images and labels  
4. Data augmentation (rotation, zoom, contrast)  
5. Pixel normalization to `[0, 1]`  
6. Dataset optimization using caching and prefetching 

This structure reflects real-world workflows where data handling is as critical as model design.

## Model Architecture & Training
A baseline CNN model was implemented to establish a strong yet interpretable reference model:

- Stacked **Conv2D + MaxPooling** layers for hierarchical feature extraction  
- Fully connected layer for feature integration  
- **Dropout** to reduce overfitting  
- **Sigmoid output** layer for binary classification  

**Training setup:**
- Optimizer: Adam  
- Loss function: Binary Crossentropy  
- Epochs: 3

The goal was to validate pipeline correctness and evaluation behavior rather than aggressively optimize performance.

## Evaluation Strategy
In medical imaging tasks, accuracy alone can be misleading. Therefore, multiple complementary metrics were used:

- **Recall:** Measures the ability to detect pneumonia cases (minimizing false negatives)  
- **Precision:** Measures prediction reliability  
- **AUC:** Evaluates overall class separation capability  
- **Confusion Matrix:** Explicit analysis of TP / FP / FN / TN

Additionally, visual inspection of predictions with confidence scores was performed.

## Results & Analysis
The model achieved strong validation performance, with high recall and AUC, indicating effective discrimination between pneumonia and normal cases. While validation accuracy fluctuated across epochs, recall and AUC remained stable, highlighting the importance of multi-metric evaluation in medical contexts.

## Limitations & Future Work
- The baseline CNN is relatively shallow; deeper architectures or transfer learning may improve performance  
- Class imbalance handling could be further refined  
- Interpretability methods such as Grad-CAM could help visualize learned attention regions  

These directions represent natural extensions toward more clinically interpretable models.
