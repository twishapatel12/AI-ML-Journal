![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# Eigenvalues and Eigenvectors for Machine Learning

---

## Introduction

**Eigenvalues and eigenvectors** are fundamental concepts in linear algebra, used heavily in machine learning (ML) for tasks such as dimensionality reduction (PCA), matrix factorization, and understanding data transformations.

They help reveal the “main directions” in which data varies and simplify complex problems—making them more manageable for ML algorithms.

---

## What are Eigenvalues and Eigenvectors?

Given a square matrix \( A \), an **eigenvector** \( v \) and its corresponding **eigenvalue** \( \lambda \) satisfy:

\[
A v = \lambda v
\]

- **Eigenvector:** A non-zero vector that only changes in scale (not direction) when multiplied by \( A \).
- **Eigenvalue:** The scale factor by which the eigenvector is stretched or compressed.

---

### Simple Example

If \( A = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix} \):

- Eigenvectors are \([1, 0]\) and \([0, 1]\)
- Eigenvalues are 2 and 3

**Explanation:**  
Multiplying \([1, 0]\) by \( A \) gives \([2, 0]\):  
Same direction, just stretched by 2.

---

### Visual: Eigenvector Transformation

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/eigenvector-transformation.png" alt="Eigenvector Transformation" width="440"/>
</p>

---

## Why Do They Matter in ML?

- **Principal Component Analysis (PCA):**  
  Finds directions (principal components) where data varies most, using eigenvectors of the covariance matrix.
- **Data Compression:**  
  Reduces dimensionality with minimal information loss.
- **Spectral Clustering & Graph Algorithms:**  
  Uses eigenvalues/eigenvectors for partitioning data and graphs.
- **Stability Analysis:**  
  Analyzes system stability and convergence.

---

## How to Compute Eigenvalues and Eigenvectors

**In Python (NumPy):**

```python
import numpy as np

A = np.array([[2, 0], [0, 3]])
values, vectors = np.linalg.eig(A)
print("Eigenvalues:", values)
print("Eigenvectors:\n", vectors)
````

**Output:**

```
Eigenvalues: [2. 3.]
Eigenvectors:
[[1. 0.]
 [0. 1.]]
```

---

## Geometric Meaning

* Eigenvectors point in directions that remain unchanged (except for scale) under the transformation $A$.
* Eigenvalues tell you **how much** those directions are stretched or squashed.

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/eigen-geometric-meaning.png" alt="Eigenvectors Geometric Meaning" width="400"/>
</p>

---

## Real World Example: PCA with Eigenvalues/Eigenvectors

Suppose you have high-dimensional data (e.g., many features per user).
**PCA** (Principal Component Analysis) finds the directions (principal components) where the data spreads out the most.

* These directions are the eigenvectors of the data’s covariance matrix.
* The size/importance of each direction is given by the corresponding eigenvalue.

**Code Example:**

```python
from sklearn.decomposition import PCA
import numpy as np

X = np.array([[2.5, 2.4],
              [0.5, 0.7],
              [2.2, 2.9],
              [1.9, 2.2],
              [3.1, 3.0]])

pca = PCA(n_components=2)
pca.fit(X)
print("Principal components (eigenvectors):\n", pca.components_)
print("Explained variance (eigenvalues):", pca.explained_variance_)
```

---

## Applications in Machine Learning

* **Dimensionality Reduction (PCA, LDA)**
* **Noise Filtering:**
  Remove less important components (low eigenvalues).
* **Feature Engineering:**
  Find new, better features by projecting data onto eigenvectors.
* **Network Analysis:**
  Community detection, ranking, stability.

---

## Further Reading & References

* [Khan Academy: Eigenvalues and Eigenvectors](https://www.khanacademy.org/math/linear-algebra/alternate-bases/eigen-everything/v/linear-algebra-eigen-everything)
* [Essence of Linear Algebra (3Blue1Brown, Video)](https://www.3blue1brown.com/lessons/linear-algebra)
* [Wikipedia: Eigenvalues and Eigenvectors](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors)
* [DeepLearning.ai: Mathematics for ML](https://www.deeplearning.ai/resources/mathematics-for-machine-learning/)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>