# Student Pass/Fail Prediction with Logistic Regression

This folder contains a notebook that uses **Logistic Regression** to predict whether a student will pass or fail based on a dataset of student information. The model's performance is evaluated using **Cross-Validation** to ensure robustness, and various metrics such as **Accuracy**, **Precision**, **Recall**, and **F1-Score** are reported.

## Table of Contents

1. [Dataset](#dataset)
2. [Data Preprocessing](#data-preprocessing)
3. [Logistic Regression Model](#logistic-regression-model)
4. [Performance Evaluation](#performance-evaluation)
5. [Summary of Results](#summary-of-results)

## Dataset

The dataset used in this project is the **Student Pass/Fail Data**. It includes various features of students and the target variable, `Pass/Fail` (binary classification: 1 for pass, 0 for fail).

- **Features**: The dataset contains student-related attributes that can be used as predictors of academic performance.
- **Target**: `Pass/Fail` indicating whether a student passed or failed.

## Data Preprocessing

The following preprocessing steps are performed:
1. **Handling missing values**: Checked for missing values and confirmed that there are none.
2. **Feature extraction**: The features (input variables) are extracted separately from the target (output variable).
3. **Train-Test Split**: The dataset is split into training and testing sets using an 80/20 split.
4. **Feature Scaling**: Standardization is applied using `StandardScaler` to normalize the feature values.

## Logistic Regression Model

The notebook implements **Logistic Regression** to model the relationship between student features and the likelihood of passing or failing.

### Cross-Validation:
- The model is trained using **5-fold Cross-Validation (CV)** to ensure that the results are robust and not dependent on the specific train-test split.
- **Cross-Validation Score**: The average accuracy across the 5 CV folds is calculated.

### Predictions:
- The model predicts the probabilities for both the pass (1) and fail (0) classes for each student in the test set.
- The final predicted class is derived from the class probabilities.

## Performance Evaluation

The model is evaluated using various performance metrics:
1. **Confusion Matrix**: Visualized to provide a detailed breakdown of the model’s predictions vs actual values.
2. **Accuracy**: The percentage of correct predictions.
3. **Precision, Recall, and F1-Score**: These metrics provide insights into the model’s ability to correctly identify students who passed and those who failed.

A **classification report** is printed, which includes detailed metrics for both classes.


## Summary of Results

The Logistic Regression model was evaluated using Cross-Validation, and the following key results were observed:
- **Average Accuracy**: The model achieved an accuracy of ~95% on the test data.
- **Confusion Matrix**: Visualized to show the number of correct and incorrect predictions for each class.
- **Precision, Recall, and F1-Score**: These metrics provided additional insight into the model’s predictive performance, with balanced performance across both classes.

The notebook demonstrates how to apply **Logistic Regression** to a binary classification problem, use cross-validation for model evaluation, and interpret the resulting performance metrics.