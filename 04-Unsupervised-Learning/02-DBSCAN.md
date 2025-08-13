![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

---

## Introduction

**DBSCAN** is a clustering algorithm that groups together data points that are close to each other **based on density**, and marks points in low-density regions as **noise** (outliers).

Unlike **K-Means**, DBSCAN:
- **Does not require specifying the number of clusters in advance**.
- Can find **arbitrarily shaped clusters**.
- Is robust to outliers.

---

## When to Use DBSCAN

- Data has irregular-shaped clusters (not spherical).
- You want automatic noise/outlier detection.
- You don’t know the number of clusters beforehand.

---

## Key Parameters

1. **eps (ε)**:  
   - Maximum distance between two samples for them to be considered **neighbors**.
   - A smaller ε means stricter clustering; larger ε means broader clusters.

2. **min_samples**:  
   - Minimum number of points required to form a **dense region** (core point).
   - Includes the point itself.

---

## Types of Points in DBSCAN

- **Core Point**:  
  A point with at least `min_samples` points (including itself) within ε distance.

- **Border Point**:  
  A point within ε distance of a core point but not meeting the `min_samples` requirement.

- **Noise Point (Outlier)**:  
  A point that is neither a core nor a border point.

---

## DBSCAN Algorithm Steps

1. Pick an **unvisited** point.
2. If it’s a core point, create a new cluster.
3. Expand the cluster by recursively including density-reachable points.
4. If it’s not a core point and not reachable from any core point, mark it as noise.
5. Repeat until all points are visited.

---

## Visual: DBSCAN Concepts

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/dbscan-core-border-noise.png" alt="DBSCAN Core, Border, and Noise Points" width="500"/>
</p>

---

## Code Example: DBSCAN in Python

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

# Create dataset with non-spherical clusters
X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=5)
labels = dbscan.fit_predict(X)

# Plot results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='plasma', s=30)
plt.title("DBSCAN Clustering")
plt.show()

# -1 label means noise points
print("Unique cluster labels:", set(labels))
````

---

## Choosing Parameters: eps and min\_samples

* **eps**:
  Use a **k-distance graph** to find an appropriate value.

  * Compute the distance to the k-th nearest neighbor for each point (k = `min_samples`).
  * Sort distances and plot them.
  * Look for the "elbow" point — a sharp change in slope.

* **min\_samples**:

  * Common rule: `min_samples = D + 1` (D = number of dimensions).
  * Can also adjust based on expected cluster density.

---

## Advantages and Limitations

**Advantages:**

* No need to specify number of clusters beforehand.
* Can detect arbitrarily shaped clusters.
* Automatically identifies noise/outliers.
* Works well for spatial/geographic data.

**Limitations:**

* Sensitive to parameter settings (ε and `min_samples`).
* Struggles with varying density clusters.
* Performance can degrade with high-dimensional data (curse of dimensionality).

---

## Best Practices

* Standardize or normalize features before applying DBSCAN (especially for distance-based clustering).
* Use **domain knowledge** to choose parameters.
* Try different ε values and compare results visually.
* For large datasets, use approximate nearest neighbor methods to speed up DBSCAN.

---

## Real-World Applications

* **Geospatial Analysis**: Grouping GPS coordinates (e.g., identifying hotspots in traffic or crime data).
* **Anomaly Detection**: Detecting fraudulent transactions or sensor malfunctions.
* **Image Segmentation**: Grouping pixels based on color/intensity similarity.
* **Social Network Analysis**: Finding communities of tightly connected users.

---

## References

* [Scikit-learn: DBSCAN Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
* [Wikipedia: DBSCAN](https://en.wikipedia.org/wiki/DBSCAN)
* [Stanford CS229: Clustering Notes](https://cs229.stanford.edu/notes2021fall/cs229-notes8.pdf)
* [Kaggle: DBSCAN Examples](https://www.kaggle.com/code/fabienj/dbscan-clustering-explanation-and-examples)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>