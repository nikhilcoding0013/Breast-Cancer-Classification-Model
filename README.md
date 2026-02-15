# Breast Cancer Classification Model

A deep learning project for classifying breast tumors as benign or malignant using CNN and transfer learning approaches.

## Overview

This project implements two neural network architectures to classify breast cancer histopathology images from the Multi Cancer Dataset. Through systematic experimentation with data augmentation, architecture modifications, and hyperparameter tuning, the model achieves 95% validation accuracy using a fine-tuned ResNet18 architecture.

## Dataset

**Source**: [Multi Cancer Dataset](https://doi.org/10.34740/KAGGLE/DSV/3415848) by Obuli Sai Naren (2022)

| Cancer | Classes | Images |
|-------|------------------|---------------------|
| Acute Lymphoblastic Leukemia | 4 | 20,000 |
| Brain Cancer | 3 | 15,000 |
| Breast Cancer | 2 | 10,000 |
| Cervical Cancer | 5 | 25,000 |
| Kidney Cancer | 2 | 10,000 |
| Lung and Colon Cancer | 5 | 25,000 |
| Lymphoma | 3 | 15,000 |
| Oral Cancer | 2 | 10,000 |

For the purpose of this project, we will isolate the Breast Cancer main cancer class.
There are a total of 10,000 breast cancer images across 2 subclasses:
- Benign tumors
- Malignant tumors

**Image specifications**: 512×512 pixels, RGB format

## Requirements

```bash
pip install kagglehub transformers timm torch torchvision scikit-learn matplotlib numpy
```

## Models Implemented

### 1. Custom CNN Architecture
- 2 convolutional layers (32 and 64 filters)
- Max pooling layers
- Fully connected layers with dropout (p=0.5)
- **Baseline accuracy**: 50%

### 2. Transfer Learning with ResNet18
- Pretrained on ImageNet
- Modified final layer with additional neurons and dropout
- Fine-tuned on breast cancer dataset
- **Final accuracy**: 95%

## Data Preprocessing

**Training augmentation**:
- Resize to 224×224
- Random rotation (±10°)
- Random resized crop (scale: 0.8-1.0)
- Random horizontal flip
- Normalization

**Validation preprocessing**:
- Resize to 224×224
- Center crop
- Normalization

## Training Configuration

### Optimal Hyperparameters (after random search)
- **Learning rate**: 0.0001
- **Batch size**: 64
- **Epochs**: 4
- **Optimizer**: Adam
- **Loss function**: CrossEntropyLoss
- **Hidden layer neurons**: 224
- **Dropout**: 0.5

## Usage

### 1. Download Dataset

```python
import kagglehub
path = kagglehub.dataset_download("obulisainaren/multi-cancer")
```

### 2. Prepare Data Subset

```python
# Create train/val split with desired proportions
# Default: 20 training images, 10 validation images per class
```

### 3. Train Models

```python
# Train custom CNN
cnn_history = train_and_validate(model, optimizer, criterion, 
                                  train_loader, val_loader,
                                  num_epochs=4, device=device, 
                                  model_name="CNN")

# Train ResNet18
resnet_history = train_and_validate(resnet_model, resnet_optimizer, 
                                     resnet_criterion, train_loader, 
                                     val_loader, num_epochs=4, 
                                     device=device, model_name="ResNet")
```

## Results

### Performance Comparison

| Model | Training Accuracy | Validation Accuracy |
|-------|------------------|---------------------|
| Custom CNN | 92.5% | 50% |
| ResNet18 (baseline) | 57.5% | 80% |
| ResNet18 (augmented) | 100% | 85% |
| ResNet18 (optimized) | 100% | **95%** |

<img width="960" height="480" alt="Untitled presentation (2)" src="https://github.com/user-attachments/assets/6f012470-ef3c-4ce2-afc0-ef37639d6c25" />

### Key Improvements

1. **Data Augmentation**: Addressed overfitting by introducing rotation, cropping, and flipping
2. **Architecture Enhancement**: Added dropout layers and optimized hidden layer size
3. **Hyperparameter Tuning**: Random search across 144 combinations identified optimal configuration
4. **Transfer Learning**: Leveraged pretrained ResNet18 features for better generalization

## Model Evolution

- **Baseline CNN**: 50% validation accuracy, severe overfitting
- **Initial ResNet**: 80% validation accuracy
- **Enhanced ResNet**: 85% validation accuracy (architectural improvements)
- **Final ResNet**: 95% validation accuracy (hyperparameter optimization)

## Visualization

The project includes training/validation loss and accuracy plots comparing both architectures across epochs, demonstrating the superior performance and generalization of the optimized ResNet model.

## Potential Future Work

- Expand to multi-class cancer classification
- Deploy model as web application
- Test on external validation datasets
- Explore attention mechanisms for interpretability

## Acknowledgments

Dataset provided by Obuli Sai Naren via Kaggle. This project was initially developed during a 2.5-hour hackathon and subsequently refined through systematic hyperparameter optimization.

## License

This repository is open source and is released under the MIT License.
This project uses the Multi Cancer Dataset, which is available under Kaggle's terms of use. Please refer to the original dataset for licensing information.
