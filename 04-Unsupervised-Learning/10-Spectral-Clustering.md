![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# Spectral Clustering

---

## Introduction

**Spectral Clustering** is an unsupervised learning algorithm that uses **graph theory** and **linear algebra** to group data points.  

Instead of clustering directly in the input space (like K-Means), spectral clustering:
1. Builds a **graph** from the data.
2. Computes the **graph Laplacian**.
3. Uses the **eigenvectors** of this Laplacian to embed the data into a low-dimensional space.
4. Applies a standard clustering algorithm (often K-Means) in this new space.

This makes spectral clustering **powerful for non-convex and irregularly shaped clusters**.

---

## When to Use Spectral Clustering

- Data has **non-spherical** or **non-linearly separable** clusters.
- Clustering based on **similarity** rather than geometric distance alone.
- Problems where data naturally forms a graph (e.g., social networks, image segmentation).

---

## How It Works (Step-by-Step)

1. **Construct a similarity graph**
   - Represent data points as nodes.
   - Connect nodes with edges weighted by similarity (e.g., Gaussian kernel, k-nearest neighbors).

2. **Build the adjacency matrix** $A$
   - $A_{ij}$ represents the similarity between points $i$ and $j$.

3. **Compute the degree matrix** $D$
   - Diagonal matrix where $D_{ii} = \sum_j A_{ij}$.

4. **Compute the graph Laplacian**:

$$
L = D - A
$$

   or the normalized Laplacian:

$$
L_{\text{sym}} = D^{-1/2} L D^{-1/2}
$$

5. **Eigen-decomposition**
   - Compute the first $k$ eigenvectors of $L$ (corresponding to the smallest eigenvalues).
   - These eigenvectors form a new feature space.

6. **Cluster in eigenvector space**
   - Apply K-Means (or other algorithm) to the rows of the eigenvector matrix.

---

## Code Example: Spectral Clustering in Python

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import SpectralClustering

# Create non-spherical data
X, y = make_moons(n_samples=300, noise=0.05, random_state=42)

# Apply Spectral Clustering
sc = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=42)
labels = sc.fit_predict(X)

# Plot results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30)
plt.title("Spectral Clustering on Two Moons Dataset")
plt.show()
````

---

## Visual: Spectral Clustering Concept

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/spectral-clustering-concept.png" alt="Spectral Clustering Concept Diagram" width="500"/>
</p>

*Shows transforming a similarity graph into an embedding space and applying clustering.*

---

## Advantages and Limitations

**Advantages:**

* Can detect complex, non-convex cluster shapes.
* Works well for graph-based data.
* Flexible choice of similarity measures.

**Limitations:**

* Computationally expensive for large datasets (requires eigen-decomposition).
* Performance depends heavily on choice of similarity measure and parameters.
* Requires knowing the number of clusters in advance.

---

## Best Practices

* Choose **affinity='nearest\_neighbors'** for sparse graphs, or **'rbf'** for dense data.
* Scale features before computing similarity.
* For large datasets, use approximate nearest neighbor methods to speed up similarity computation.
* Use domain knowledge to select the number of clusters and similarity parameters.

---

## Real-World Applications

* **Image segmentation** (grouping similar pixels).
* **Social network analysis** (community detection).
* **Speech separation** (grouping frequency components).
* **Document clustering** (based on word similarity).

---

## References

* [Scikit-learn: Spectral Clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html)
* [Wikipedia: Spectral Clustering](https://en.wikipedia.org/wiki/Spectral_clustering)
* [Tutorial: A Tutorial on Spectral Clustering (Ng, Jordan, Weiss, 2002)](https://ai.stanford.edu/~ang/papers/nips01-spectral.pdf)
* [Graph Laplacian Notes - Stanford CS224W](http://web.stanford.edu/class/cs224w/)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>