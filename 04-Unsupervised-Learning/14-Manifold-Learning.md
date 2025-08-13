![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# Manifold Learning

---

## Introduction

**Manifold Learning** is a family of **non-linear dimensionality reduction** techniques designed to uncover low-dimensional structures (manifolds) embedded in high-dimensional data.

The core idea:
- Real-world high-dimensional data often lies on or near a **manifold** of much lower dimension.
- By mapping data to this lower-dimensional space, we can visualize it, remove noise, and simplify learning tasks.

**Example:**  
Images of a rotating object in 3D may lie on a smooth 2D surface (rotation angle × lighting angle) inside a very high-dimensional pixel space.

---

## Why Manifold Learning?

- Many datasets have **non-linear relationships** that PCA (linear method) cannot capture.
- Manifold methods preserve **local geometry** and **neighborhood relationships**.
- Useful for **visualizing complex datasets** in 2D/3D.

---

## Popular Manifold Learning Algorithms

### 1. Isomap
- Extends classical multidimensional scaling (MDS) by replacing Euclidean distance with **geodesic distance** along the manifold.
- Steps:
  1. Build nearest neighbor graph.
  2. Compute shortest paths (geodesic distances).
  3. Apply MDS to these distances.

---

### 2. Locally Linear Embedding (LLE)
- Preserves local linear relationships between neighbors.
- Steps:
  1. Each point is expressed as a weighted sum of its neighbors.
  2. Find low-dimensional embedding that preserves these weights.

---

### 3. Laplacian Eigenmaps
- Constructs a graph from data points and computes the **graph Laplacian**.
- Uses the smallest non-zero eigenvectors to embed points in low dimensions.

---

### 4. t-SNE and UMAP (Modern Non-Linear Methods)
- **t-SNE**: Preserves local structure, useful for visualization.
- **UMAP**: Preserves more global structure, faster for large datasets.

---

## Visual: Manifold Learning Concept

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/manifold-learning-concept.png" alt="Manifold Learning Concept Diagram" width="500"/>
</p>

*Shows high-dimensional curved surface (manifold) being “unfolded” into a flat, low-dimensional representation.*

---

## Code Example: Isomap and LLE in Python

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import Isomap, LocallyLinearEmbedding

# Generate 3D Swiss roll dataset
X, color = make_swiss_roll(n_samples=1500, noise=0.05, random_state=42)

# Apply Isomap
isomap = Isomap(n_neighbors=10, n_components=2)
X_iso = isomap.fit_transform(X)

# Apply LLE
lle = LocallyLinearEmbedding(n_neighbors=10, n_components=2, method='standard')
X_lle = lle.fit_transform(X)

# Plot results
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
axs[0].scatter(X[:, 0], X[:, 2], c=color, cmap=plt.cm.Spectral)
axs[0].set_title("Original Swiss Roll (3D)")

axs[1].scatter(X_iso[:, 0], X_iso[:, 1], c=color, cmap=plt.cm.Spectral)
axs[1].set_title("Isomap Embedding (2D)")

axs[2].scatter(X_lle[:, 0], X_lle[:, 1], c=color, cmap=plt.cm.Spectral)
axs[2].set_title("LLE Embedding (2D)")

plt.show()
````

---

## Advantages and Limitations

**Advantages:**

* Captures complex, non-linear structures in data.
* Preserves local relationships between points.
* Useful for visualization and noise reduction.

**Limitations:**

* Computationally expensive for large datasets.
* Sensitive to choice of parameters (e.g., number of neighbors).
* Not always suitable for out-of-sample data (embedding new points may require retraining).

---

## Best Practices

* Normalize data before applying manifold learning.
* Experiment with different algorithms (Isomap, LLE, UMAP) for your dataset.
* Use cross-validation to tune parameters like `n_neighbors`.
* For very large datasets, prefer faster methods like UMAP.

---

## Real-World Applications

* **Image processing**: Uncovering pose/lighting manifolds.
* **Genomics**: Visualizing gene expression profiles.
* **Natural language processing**: Embedding semantic spaces.
* **Robotics**: Learning low-dimensional state spaces.

---

## References

* [Scikit-learn: Manifold Learning](https://scikit-learn.org/stable/modules/manifold.html)
* [Wikipedia: Manifold Learning](https://en.wikipedia.org/wiki/Nonlinear_dimensionality_reduction)
* [Tenenbaum et al., Science 2000 - Isomap](https://science.sciencemag.org/content/290/5500/2319)
* [Roweis & Saul, Science 2000 - LLE](https://science.sciencemag.org/content/290/5500/2323)
* [UMAP: Uniform Manifold Approximation and Projection](https://arxiv.org/abs/1802.03426)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>