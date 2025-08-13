![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# K-Means Clustering

---

## Introduction

**K-Means** is one of the most popular **unsupervised learning** algorithms for clustering.  
It groups similar data points into **K clusters** based on feature similarity, **without any labeled data**.

Clustering is useful for:
- Understanding data structure
- Grouping similar items
- Data compression
- Customer segmentation

---

## How K-Means Works

K-Means tries to find **K centroids** (points representing each cluster) such that:
- Each data point belongs to the cluster with the nearest centroid.
- The sum of squared distances between points and their assigned centroid is minimized.

---

## Steps of the Algorithm

1. **Choose K** – the number of clusters.
2. **Initialize centroids** – randomly pick K points as starting centroids.
3. **Assign clusters** – assign each point to the nearest centroid (using a distance metric, usually Euclidean).
4. **Update centroids** – compute new centroids as the mean of all points assigned to each cluster.
5. **Repeat steps 3 & 4** until centroids no longer change significantly (convergence).

---

## Mathematical Formulation

Given:
- Dataset $X = \{x_1, x_2, \dots, x_n\}$
- Number of clusters $K$

K-Means minimizes:

$$
J = \sum_{i=1}^{K} \sum_{x_j \in C_i} \|x_j - \mu_i\|^2
$$

Where:
- $C_i$ = set of points in cluster $i$
- $\mu_i$ = centroid of cluster $i$
- $\|x_j - \mu_i\|^2$ = squared Euclidean distance

---

## Code Example: K-Means in Python

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate synthetic data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

# Apply K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Plot results
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, label='Centroids')
plt.title("K-Means Clustering")
plt.legend()
plt.show()
````

---

## Choosing K: The Elbow Method

One common method for selecting $K$ is the **Elbow Method**:

1. Run K-Means for different values of K (e.g., 1 to 10).
2. Compute the **inertia** (sum of squared distances from points to their closest centroid).
3. Plot inertia vs. K.
4. The "elbow point" (where the curve bends) suggests a good K.

**Code Example:**

```python
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for K")
plt.show()
```

---

## Visual: K-Means Clustering Process

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/kmeans-clustering-process.png" alt="K-Means Clustering Process" width="500"/>
</p>

---

## Advantages and Limitations

**Advantages:**

* Simple and easy to implement.
* Efficient on large datasets.
* Works well when clusters are roughly spherical.

**Limitations:**

* Must choose K beforehand.
* Sensitive to outliers (can distort centroids).
* Assumes clusters are of similar size and density.
* May converge to a local minimum (use multiple initializations with `n_init` parameter).

---

## Best Practices

* Scale features before running K-Means (especially if units differ).
* Use `n_init > 10` to improve stability.
* Try different K values using the elbow or silhouette method.
* For large datasets, use **MiniBatchKMeans** for faster training.

---

## Real-World Applications

* Customer segmentation in marketing.
* Image compression (color quantization).
* Document clustering for topic discovery.
* Grouping sensor data in IoT applications.

---

## References

* [Scikit-learn: K-Means Clustering](https://scikit-learn.org/stable/modules/clustering.html#k-means)
* [Kaggle: K-Means Tutorial](https://www.kaggle.com/code/prashant111/k-means-clustering-with-python)
* [Wikipedia: K-Means Clustering](https://en.wikipedia.org/wiki/K-means_clustering)
* [Stanford CS229: Clustering Notes](https://cs229.stanford.edu/notes2021fall/cs229-notes8.pdf)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>