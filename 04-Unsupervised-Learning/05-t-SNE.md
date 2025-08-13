![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# t-SNE (t-Distributed Stochastic Neighbor Embedding)

---

## Introduction

**t-SNE** is a popular **non-linear dimensionality reduction** technique, mainly used for **visualizing high-dimensional data** in 2D or 3D.

It is especially good at:
- Preserving **local structure** (nearby points remain close together in lower dimensions)
- Revealing **clusters and patterns** in complex datasets
- Working with non-linear relationships between features

However, t-SNE is **not a clustering algorithm**—it’s only for visualization.

---

## Key Idea

t-SNE works by:
1. Measuring similarity between points in **high-dimensional space** (using probabilities based on distances).
2. Mapping points to a **low-dimensional space** (2D or 3D) where the similarities are preserved as much as possible.
3. Using a **Kullback-Leibler (KL) divergence** cost function to minimize the difference between high-dimensional and low-dimensional similarities.

---

## How It Works (Step-by-Step)

1. **Compute pairwise similarities**  
   - In high-dimensional space, similarity between points $x_i$ and $x_j$ is based on Gaussian probability:

$$
p_{j|i} = \frac{\exp(-||x_i - x_j||^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-||x_i - x_k||^2 / 2\sigma_i^2)}
$$

   - The bandwidth $\sigma_i$ is chosen to match a given *perplexity*.

2. **Create joint probability distribution** $P$ in high dimensions.

3. **Initialize points** randomly in low-dimensional space.

4. **Compute similarities** $Q$ in low-dimensional space using a Student’s t-distribution (heavier tails than Gaussian to avoid “crowding” problem).

5. **Minimize KL divergence** between $P$ and $Q$ using gradient descent.

---

## Key Parameters

- **n_components**: Target dimension (usually 2 or 3).
- **perplexity**: Controls balance between local and global structure. Typical values: 5–50.
- **learning_rate**: Step size in optimization; too low = slow convergence, too high = unstable.
- **n_iter**: Number of iterations (>= 1000 recommended).

---

## Code Example: t-SNE in Python

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Load dataset
digits = load_digits()
X, y = digits.data, digits.target

# Standardize data
X_scaled = StandardScaler().fit_transform(X)

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Plot results
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=15)
plt.legend(*scatter.legend_elements(), title="Digits", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("t-SNE Visualization of Digits Dataset")
plt.show()
````

---

## Visual: t-SNE Concept

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/tsne-concept.png" alt="t-SNE Concept Diagram" width="500"/>
</p>

*Shows mapping from high-dimensional clusters to low-dimensional space while preserving local neighborhood relationships.*

---

## Advantages and Limitations

**Advantages:**

* Great for visualization of complex, high-dimensional datasets.
* Captures non-linear relationships.
* Preserves local neighborhood structure.

**Limitations:**

* Computationally expensive for large datasets (use **Multicore t-SNE** or **openTSNE** for speed).
* Non-deterministic unless you fix the random seed.
* Perplexity and learning rate require tuning.
* Not suitable for embedding new points without re-running.

---

## Best Practices

* Standardize or normalize features before running t-SNE.
* Try multiple perplexity values (5–50) to find best structure.
* Use PCA to reduce dimensions first (e.g., from 1000 → 50) before t-SNE for faster performance.
* Avoid interpreting distances between clusters literally—t-SNE focuses on local relationships.

---

## Real-World Applications

* Visualizing word embeddings (NLP)
* Exploring gene expression data (bioinformatics)
* Visualizing intermediate neural network activations
* Customer segmentation exploration

---

## References

* [Scikit-learn: t-SNE Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
* [Original t-SNE Paper (van der Maaten & Hinton, 2008)](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)
* [Distill.pub: How t-SNE Works](https://distill.pub/2016/misread-tsne/)
* [openTSNE: Fast t-SNE Implementation](https://opentsne.readthedocs.io/en/latest/)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>