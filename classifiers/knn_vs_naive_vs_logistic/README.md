# Titanic Survival Prediction

This folder contains a machine learning notebook that implements multiple classification algorithms to predict the survival of passengers on the Titanic based on the famous Titanic dataset. The notebook explores various algorithms, including **K-Nearest Neighbors (KNN)**, **Logistic Regression**, and **Naive Bayes**, and compares their performance metrics.

## Table of Contents

1. [Dataset](#dataset)
2. [Data Preprocessing](#data-preprocessing)
3. [Machine Learning Models](#machine-learning-models)
4. [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
5. [Performance Evaluation](#performance-evaluation)
6. [Summary of Results](#summary-of-results)

## Dataset

The dataset used in this project is the classic **Titanic dataset**, which contains the following key features:
- Passenger information such as `Sex`, `Age`, `Fare`, `Embarked`, etc.
- The target variable is `Survived` (1 for survived, 0 for not survived).

The dataset is cleaned, preprocessed, and split into training and test sets for model training and evaluation.

## Data Preprocessing

The notebook includes the following preprocessing steps:
1. **Dropping irrelevant columns**: Features such as `PassengerId`, `Name`, `Ticket`, and `Cabin` are dropped.
2. **Handling missing values**: 
   - Missing values in the `Age` column are filled with the mean.
   - Rows with missing values in `Embarked` are dropped.
3. **One-hot encoding**: Categorical variables like `Sex` and `Embarked` are transformed into dummy variables to prepare the data for machine learning models.
4. **Standardization**: Feature scaling is applied using `StandardScaler`.

## Machine Learning Models

The notebook trains and evaluates the following classification models:
1. **K-Nearest Neighbors (KNN)**:
   - An initial KNN model is trained and tested.
   - The optimal number of neighbors `k` is determined through error rate analysis.
2. **Logistic Regression**:
   - A Logistic Regression model is trained and evaluated both on the standardized dataset and on the PCA-reduced dataset.
3. **Naive Bayes**:
   - A Naive Bayes model is also trained and evaluated both on the standardized dataset and on the PCA-reduced dataset.

## Principal Component Analysis (PCA)

The dataset is transformed using **PCA** to reduce dimensionality and create uncorrelated features. The PCA components are then used to train Logistic Regression and Naive Bayes models, allowing for a comparison between the original and PCA-transformed feature sets.

## Performance Evaluation

The models are evaluated using the following metrics:
- **Accuracy**: The percentage of correct predictions.
- **Precision**: The proportion of positive identifications that are correct.
- **Recall**: The proportion of actual positives that were correctly identified.
- **F1-Score**: The harmonic mean of precision and recall.

Confusion matrices and classification reports are generated to give detailed insight into the performance of each model.

## Summary of Results

After testing multiple models, **K-Nearest Neighbors (KNN)** with the optimal `k` value was found to have the best accuracy on the Titanic dataset. The performance of other models, including Logistic Regression and Naive Bayes, was also evaluated, with detailed metrics provided for comparison.