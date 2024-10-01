# Customer Segmentation Using Clustering Algorithms

This folder contains two notebooks that perform customer segmentation using various clustering algorithms on the **Mall Customers** dataset. The clustering algorithms applied include **KMeans**, **DBSCAN**, and **Hierarchical Clustering**, and the models are evaluated based on performance metrics such as **Silhouette Score** and **Calinski-Harabasz (CH) Index**.

## Table of Contents

1. [Dataset Overview](#dataset-overview)
2. [Clustering Algorithms](#clustering-algorithms)
    - DBSCAN
    - KMeans
    - Hierarchical Clustering
3. [Evaluation Metrics](#evaluation-metrics)
    - Silhouette Score
    - Calinski-Harabasz Index (CH-Index)
4. [Comparison of Algorithms](#comparison-of-algorithms)
5. [Conclusion](#conclusion)

---

## Dataset Overview

The dataset used for this analysis is the **Mall Customers** dataset, which contains data on customers, including:
- **Age**
- **Annual Income (k$)**
- **Spending Score (1-100)**

The goal is to segment customers into different clusters based on their purchasing behavior and demographics using different clustering algorithms.

---

## Clustering Algorithms

### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
- DBSCAN groups together points that are closely packed and marks points that are far away as outliers. 
- The performance is evaluated using a range of values for the parameters `epsilon` (neighborhood size) and `min_samples` (minimum number of points).
- The **Silhouette Score** and **CH-Index** are used to find the optimal parameters for DBSCAN.

### KMeans Clustering
- KMeans is a centroid-based clustering technique that partitions the data into `k` clusters by minimizing the variance within clusters.
- The number of clusters is determined using both **Silhouette Score** and **CH-Index** by testing values of `k` from 2 to 20.

### Hierarchical Clustering
- Hierarchical clustering is a tree-based clustering technique where data points are grouped based on their distance in a hierarchical structure.
- A **Dendrogram** is used to determine the optimal number of clusters, and the clusters are evaluated using the **Silhouette Score** and **CH-Index**.

---

## Evaluation Metrics

### Silhouette Score
- The **Silhouette Score** measures how similar a data point is to its own cluster (cohesion) compared to other clusters (separation).
- A higher Silhouette Score indicates that the points are well-matched to their own clusters and poorly matched to neighboring clusters.

### Calinski-Harabasz Index (CH-Index)
- The **CH-Index** evaluates the ratio of the sum of the between-clusters dispersion and the within-cluster dispersion. A higher score indicates well-defined clusters.

---

## Comparison of Algorithms

### DBSCAN vs KMeans (1st Notebook)
- **Silhouette Score**: KMeans achieves a higher score than DBSCAN, indicating better-defined clusters.
- **CH-Index**: KMeans also outperforms DBSCAN in terms of the CH-Index, suggesting it forms more cohesive and separate clusters.
  
The results indicate that **KMeans** is the better clustering algorithm for this dataset compared to DBSCAN.

### KMeans vs Hierarchical Clustering (2nd Notebook)
- **Silhouette Score**: KMeans performs better than Hierarchical Clustering in terms of the Silhouette Score, with the optimal number of clusters for KMeans being 10.
- **CH-Index**: KMeans also achieves a higher CH-Index than Hierarchical Clustering, with 9 clusters being the best number of clusters for KMeans.

Overall, **KMeans** outperforms Hierarchical Clustering on this dataset.

---

## Conclusion

Based on the evaluation metrics, **KMeans** is the best clustering algorithm for segmenting the Mall Customers dataset. Both **Silhouette Score** and **CH-Index** indicate that KMeans provides more distinct and cohesive clusters compared to both **DBSCAN** and **Hierarchical Clustering**. 

The optimal number of clusters for KMeans is found to be **9 or 10**, depending on the metric used.
