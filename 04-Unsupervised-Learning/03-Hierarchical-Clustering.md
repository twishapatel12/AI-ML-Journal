![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# Hierarchical Clustering

---

## Introduction

**Hierarchical Clustering** is an unsupervised learning method that builds a hierarchy (tree structure) of clusters.  
Unlike **K-Means** or **DBSCAN**, it doesn’t require you to specify the number of clusters upfront (although you can decide later by “cutting” the tree).

It is especially useful for:
- Understanding the **nested grouping** in data
- Creating **dendrograms** for visualization
- Detecting relationships between clusters

---

## Two Main Approaches

### 1. Agglomerative (Bottom-Up)
- Start with each point as its **own cluster**.
- Iteratively merge the two closest clusters until only one remains (or desired number of clusters is reached).
- Most common type in practice.

### 2. Divisive (Top-Down)
- Start with all points in **one cluster**.
- Recursively split clusters into smaller ones.
- Less commonly used due to higher computational cost.

---

## Key Concept: Linkage Criteria

The **linkage method** determines how to measure the distance between clusters.

1. **Single Linkage**: Distance between the closest points in two clusters.
2. **Complete Linkage**: Distance between the farthest points in two clusters.
3. **Average Linkage**: Average distance between all pairs of points in two clusters.
4. **Ward’s Method**: Merges clusters that lead to the minimum increase in total within-cluster variance.

---

## Distance Metrics

- **Euclidean distance** (most common)
- Manhattan, Cosine, or other metrics depending on data type.

---

## Visual: Dendrogram

A **dendrogram** is a tree-like diagram showing the merge process.

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/hierarchical-dendrogram.png" alt="Hierarchical Clustering Dendrogram" width="500"/>
</p>

- The vertical axis = distance or dissimilarity between clusters.
- Cutting the dendrogram at a certain height gives the desired number of clusters.

---

## Code Example: Agglomerative Clustering with Dendrogram

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering

# Create synthetic data
X, _ = make_blobs(n_samples=10, centers=3, random_state=42)

# Generate linkage matrix for dendrogram
linked = linkage(X, method='ward')

# Plot dendrogram
plt.figure(figsize=(8, 4))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.show()

# Apply Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = agg.fit_predict(X)
print("Cluster labels:", labels)
````

---

## Advantages and Limitations

**Advantages:**

* No need to specify number of clusters upfront.
* Produces a hierarchy (useful for exploratory analysis).
* Can work with different distance metrics.

**Limitations:**

* Computationally expensive for large datasets (`O(n^3)` complexity).
* Sensitive to noise and outliers.
* Once a merge/split is done, it cannot be undone.

---

## Best Practices

* Scale/normalize data before clustering.
* Use **Ward’s method** for numeric, continuous data (minimizes variance).
* For large datasets, consider **fast approximations** or use hierarchical clustering only on cluster centroids from another method.

---

## Real-World Applications

* **Biology**: Building phylogenetic trees of species.
* **Marketing**: Customer segmentation with hierarchical relationships.
* **Document Clustering**: Organizing news articles by topics and subtopics.
* **Gene Expression Data**: Grouping genes with similar activity patterns.

---

## References

* [Scikit-learn: Agglomerative Clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)
* [SciPy: Hierarchical Clustering](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html)
* [Wikipedia: Hierarchical Clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering)
* [Stanford CS229 Notes on Clustering](https://cs229.stanford.edu/notes2021fall/cs229-notes8.pdf)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>