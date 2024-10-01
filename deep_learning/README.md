# HAR Image Classification using a Fully Connected Neural Network

This project implements a deep learning model to classify images from the HAR (Human Activity Recognition) dataset using a fully connected neural network (FCNN) built with PyTorch. The dataset contains images of three human activities: Catch, Clap, and Hammering. The model uses image preprocessing, data augmentation, and training on GPU to achieve classification accuracy.

---

# Table of Contents

1. [Dataset](#dataset)
2. [Model Architecture](#model-architecture)
3. [Training](#training)
4. [Results](#results)
5. [Conclusion](#conclusion)

---

# Dataset

The HAR dataset used in this project contains images of three activities: `Catch`, `Clap`, and `Hammering`. The dataset is organized into the following structure:

```
HAR_Images/
├── Catch/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── Clap/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── Hammering/
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```

The dataset is downloaded from Google Drive, unzipped, and split into training and validation sets using `splitfolders`.

# Model Architecture

The model used in this project is a fully connected neural network (FCNN) with the following layers:

- Input layer: 12,288 units (3 channels x 64 x 64 image dimensions)
- Several hidden layers with `ReLU` activation functions
- Output layer: 3 units (for the 3 classes: Catch, Clap, Hammering)

## Model Summary:
```python
model = nn.Sequential(
    nn.Linear(12288, 2048),
    nn.ReLU(),
    nn.Linear(2048, 2048),
    nn.ReLU(),
    nn.Linear(2048, 1024),
    nn.ReLU(),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 3)
)
```

The model is trained using the **CrossEntropyLoss** loss function and **Stochastic Gradient Descent (SGD)** optimizer with momentum.

---

# Training

1. **Data Loading**: The images are resized to 64x64 pixels and converted to tensors using `torchvision.transforms`. The dataset is split into training and validation sets in an 80:20 ratio using `splitfolders`.
   
2. **Model Training**: The model is trained for 50 epochs. The training and validation accuracy and loss are tracked.

3. **Performance Metrics**: The model's accuracy on the training and validation datasets is calculated and plotted over time. 

---

# Results

After training for 50 epochs, the model achieves the following results:

- **Train Accuracy**: ~96%
- **Test Accuracy**: ~100%

The training loss decreases steadily over time, and the validation accuracy shows a stable improvement.

---

# Conclusion

This project demonstrates how to build a fully connected neural network for image classification using the HAR dataset. It highlights the importance of data preprocessing, model architecture, and evaluation metrics in the development of deep learning models. While the model performs well, further improvements can be made by exploring different architectures like convolutional neural networks (CNNs).