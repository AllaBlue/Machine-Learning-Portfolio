# Machine Learning and Deep Learning Portfolio

This repository showcases a collection of machine learning and deep learning projects implemented across various algorithms and datasets. The projects cover different tasks, including classification, clustering, dimensionality reduction, and deep learning-based image recognition.

## Repository Structure

The repository is organized into the following sections:

```
classifiers/
    knn_vs_naive_vs_logistic/
        knn_vs_naive_vs_logistic.ipynb
        README.md
        titanic_train.csv
    logistic_regression/
        logistic_regression.ipynb
        README.md
        Student-Pass-Fail-Data.csv
    trees/
        decision_tree_vs_random_forest.ipynb
        README.md
        titanic_train.csv
clustering/
    dbscan_vs_kmeans.ipynb
    hierarchical_vs_kmeans.ipynb
    Mall_Customers.csv
    README.md
deep_learning/
    human_activity_recognision.ipynb
    README.md
pca/
    pca_dbscan.ipynb
    README.md
    titanic_train.csv
README.md
requirements.txt
```

### Key Folders:
1. **classifiers**: Implements various classification algorithms on datasets like Titanic survival prediction and student performance.
2. **clustering**: Focuses on clustering algorithms (DBSCAN, KMeans, and Hierarchical) applied to customer segmentation and performance comparison.
3. **deep_learning**: Applies a deep learning model to classify human activities using the HAR images dataset.
4. **pca**: Implements PCA (Principal Component Analysis) combined with DBSCAN to analyze and cluster Titanic dataset features.

---

## Table of Contents
1. [Classifiers](#classifiers)
2. [Clustering](#clustering)
3. [Deep Learning](#deep-learning)
4. [Principal Component Analysis (PCA)](#pca)
5. [Installation and Setup](#installation-and-setup)

---

## Classifiers

### 1. KNN vs Naive Bayes vs Logistic Regression on Titanic Dataset
- **Description**: Compares the performance of three classification algorithms (KNN, Naive Bayes, and Logistic Regression) on the Titanic dataset.
- **Dataset**: Titanic dataset.
- **Key Features**:
  - Data preprocessing (missing value handling, feature scaling, one-hot encoding).
  - Hyperparameter tuning for KNN (`k` optimization).
  - Performance evaluation metrics (Accuracy, Precision, Recall, F1-Score).

### 2. Logistic Regression on Student Pass/Fail Data
- **Description**: Uses Logistic Regression to predict whether students pass or fail.
- **Dataset**: Student Pass/Fail dataset.
- **Key Features**:
  - Cross-Validation for model robustness.
  - Performance evaluation using metrics like Accuracy, Precision, and Recall.

### 3. Decision Tree vs Random Forest on Titanic Dataset
- **Description**: Implements Decision Tree and Random Forest models on the Titanic dataset. Explores hyperparameter tuning for optimal performance.
- **Dataset**: Titanic dataset.
- **Key Features**:
  - Feature importance analysis for Decision Tree and Random Forest.
  - Hyperparameter tuning for Random Forest (number of trees, max depth).

For more details, refer to the respective `README.md` files in the project directories.

---

## Clustering

### 1. DBSCAN vs KMeans
- **Description**: Compares the DBSCAN and KMeans clustering algorithms on the Mall Customers dataset.
- **Dataset**: Mall Customers dataset.
- **Key Features**:
  - Clustering performance evaluated using Silhouette Score and Calinski-Harabasz Index (CH-Index).
  - Hyperparameter tuning for both DBSCAN and KMeans (eps and `k` values).
  - Visual comparison of clustering results.

### 2. Hierarchical Clustering vs KMeans
- **Description**: Compares Hierarchical Clustering and KMeans on the Mall Customers dataset.
- **Dataset**: Mall Customers dataset.
- **Key Features**:
  - Dendrogram for determining the optimal number of clusters.
  - Performance comparison using Silhouette Score and CH-Index.

---

## Deep Learning

### Human Activity Recognition (HAR) Using Fully Connected Neural Network
- **Description**: A deep learning model for image classification using the HAR Images dataset. The model classifies human activities (Catch, Clap, Hammering) using a fully connected neural network (FCNN).
- **Dataset**: HAR Images dataset.
- **Key Features**:
  - Dataset preprocessing (image resizing, tensor conversion).
  - Model architecture: FCNN with multiple hidden layers and ReLU activation.
  - Training and validation with metrics tracking (accuracy, loss).
  - GPU support for faster training.

---

## Principal Component Analysis (PCA)

### Titanic Dataset Clustering with PCA and DBSCAN
- **Description**: Applies PCA for dimensionality reduction on the Titanic dataset and clusters the passengers using DBSCAN.
- **Dataset**: Titanic dataset.
- **Key Features**:
  - PCA for reducing dataset dimensions to 2, 3, and 4 components.
  - DBSCAN clustering applied on PCA-reduced data.
  - Evaluation of clustering performance using Silhouette Score and CH-Index.

---

## Installation and Setup

Install the dependencies by running:
```bash
pip install -r requirements.txt
```