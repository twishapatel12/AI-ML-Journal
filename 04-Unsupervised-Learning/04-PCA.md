![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# Principal Component Analysis (PCA)

---

## Introduction

**Principal Component Analysis (PCA)** is one of the most widely used **dimensionality reduction** techniques in machine learning and data analysis.  

It transforms high-dimensional data into a smaller set of **new variables** (called **principal components**) that retain most of the important information in the dataset.

PCA is useful for:
- Reducing computation time for algorithms
- Visualizing high-dimensional data
- Removing noise and redundant features
- Mitigating the curse of dimensionality

---

## Key Idea

PCA finds **directions in the data** (principal components) along which the variance is maximized:
- **First principal component (PC1):** direction of maximum variance.
- **Second principal component (PC2):** orthogonal to PC1, with the next highest variance.
- And so on.

---

## Steps of PCA

1. **Standardize the data**  
   - Ensure features have mean = 0 and variance = 1.
   - This prevents features with large scales from dominating.

2. **Compute the covariance matrix** 

$$
\Sigma = \frac{1}{n-1} X^T X
$$

3. **Calculate eigenvalues and eigenvectors** of the covariance matrix.
   - Eigenvectors → directions (principal components)
   - Eigenvalues → variance explained by each component

4. **Sort and select top k components** that explain most of the variance.

5. **Transform data** onto the selected components.

---

## Mathematical Formulation

Given dataset $X$ with shape $n \times p$:

- Covariance matrix:

$$
C = \frac{1}{n-1} X^\top X
$$

- Solve:

$$
C v = \lambda v
$$

Where:
- $v$ = eigenvector (principal component direction)
- $\lambda$ = eigenvalue (variance explained)

---

## Code Example: PCA in Python

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Standardize features
X_scaled = StandardScaler().fit_transform(X)

# Apply PCA (2 components for visualization)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Explained variance ratio:", pca.explained_variance_ratio_)

# Plot results
plt.figure(figsize=(6, 4))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA on Iris Dataset")
plt.show()
````

---

## Choosing the Number of Components

One way is to look at the **explained variance ratio**:

```python
pca_full = PCA().fit(X_scaled)
plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title("Explained Variance by PCA Components")
plt.grid(True)
plt.show()
```

The “elbow” in the curve shows where adding more components gives diminishing returns.

---

## Visual: PCA Concept

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/pca-concept.png" alt="PCA Concept Diagram" width="500"/>
</p>

*Illustrates projecting data onto a new axis (principal component) to reduce dimensions.*

---

## Advantages and Limitations

**Advantages:**

* Reduces dimensionality while preserving most information.
* Removes multicollinearity between features.
* Improves speed for some algorithms.
* Useful for visualization.

**Limitations:**

* Linear method (cannot capture complex, non-linear relationships).
* Components are hard to interpret.
* Sensitive to scaling of data.
* Not suitable for categorical variables without preprocessing.

---

## Best Practices

* Always **standardize/normalize** features before applying PCA.
* Use PCA mainly for **exploration** and **preprocessing** rather than as a final model (unless dimensionality is truly a problem).
* Combine PCA with clustering or classification for better results in high dimensions.

---

## Real-World Applications

* **Image compression** (reducing pixels while keeping key patterns).
* **Genomics** (reducing gene expression features).
* **Finance** (reducing correlated stock indicators).
* **Face recognition** (Eigenfaces).

---

## References

* [Scikit-learn: PCA Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
* [Wikipedia: Principal Component Analysis](https://en.wikipedia.org/wiki/Principal_component_analysis)
* [Stanford CS229: Dimensionality Reduction Notes](https://cs229.stanford.edu/notes2021fall/cs229-notes10.pdf)
* [Khan Academy: Eigenvalues and Eigenvectors](https://www.khanacademy.org/math/linear-algebra/alternate-bases/pca/v/principal-component-analysis)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>