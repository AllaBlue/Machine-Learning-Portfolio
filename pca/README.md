# Titanic Dataset Clustering with DBSCAN and PCA

This notebook applies **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) along with **Principal Component Analysis (PCA)** for dimensionality reduction to cluster the passengers of the Titanic dataset. The dataset is first cleaned, preprocessed, and then analyzed using unsupervised learning techniques to find patterns and structure.

## Table of Contents

1. [Dataset Overview](#dataset-overview)
2. [Data Preprocessing](#data-preprocessing)
3. [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
4. [DBSCAN Clustering](#dbscan-clustering)
5. [Performance Evaluation](#performance-evaluation)
6. [Conclusion](#conclusion)

---

## Dataset Overview

The dataset used is the famous **Titanic** dataset, which contains information about passengers, such as:
- **Pclass**: Passenger class
- **Age**: Age of the passengers
- **Sex**: Gender of the passengers
- **Fare**: Fare paid by passengers
- **Embarked**: Port of embarkation

The goal is to explore clustering using **DBSCAN** and evaluate its effectiveness on this dataset, which includes passengers who survived and those who did not.

---

## Data Preprocessing

Several steps are taken to clean and prepare the data:
1. **Dropping Irrelevant Features**: The features `Name`, `Ticket`, `Cabin`, and `PassengerId` are dropped.
2. **Handling Missing Values**: Missing values in the `Embarked` column are removed, while missing `Age` values are imputed based on the most correlated feature (`Pclass`).
3. **One-Hot Encoding**: The categorical features `Sex` and `Embarked` are converted into dummy variables.
4. **Feature Scaling**: The features are standardized using `StandardScaler` to ensure that they have comparable ranges.

---

## Principal Component Analysis (PCA)

To reduce the dimensionality of the dataset and focus on the most important components, **PCA** is applied. The notebook explores:
- **4 Components**: Explained variance of ~72%.
- **3 Components**: Explained variance of ~65%.
- **2 Components**: Explained variance of ~57%.

The two principal components are visualized to provide insight into how the data is distributed across these dimensions.

---

## DBSCAN Clustering

**DBSCAN** is used for clustering the data points based on density:
- **Initial DBSCAN Model**: Uses ε (epsilon) = 0.5 to find clusters, and the **Silhouette Score** is computed for evaluation.
- **Optimizing Parameters**: Various values of ε and `min_samples` are explored to find the best-performing model based on both the **Silhouette Score** and **Calinski-Harabasz Index (CH-Index)**.

The best values for ε and `min_samples` are identified by maximizing these two metrics.

---

## Performance Evaluation

The models are evaluated using two key metrics:
1. **Silhouette Score**: Measures how well the clusters are separated.
2. **Calinski-Harabasz Index (CH-Index)**: Measures the ratio of the sum of between-cluster dispersion and within-cluster dispersion.

The notebook compares models with different values of ε and `min_samples`, finding that:
- A model with ε = 0.4 and `min_samples` = 8 has a lower Silhouette Score but a much higher CH-Index compared to a model with ε = 0.7 and `min_samples` = 7.

---

## Conclusion

The comparison of DBSCAN models on the Titanic dataset shows that:
- A **CH-Index**-optimized DBSCAN model with ε = 0.4 and `min_samples` = 8 performs better than a model optimized for **Silhouette Score**.
- However, **DBSCAN** does not effectively separate passengers who survived from those who did not, indicating that it may not be suitable for this particular dataset.
