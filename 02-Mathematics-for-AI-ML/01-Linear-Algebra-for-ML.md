![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# Linear Algebra for Machine Learning

---

## Introduction

Linear algebra is the mathematics of vectors and matrices.  
It’s the foundation for many machine learning algorithms—especially in deep learning, data transformations, and optimization.

This guide explains the core concepts, how they are used in ML, and gives examples with code and diagrams.

---

## Why Is Linear Algebra Important in ML?

- Data in ML is often represented as vectors (lists) and matrices (tables of numbers).
- Operations like dot product, matrix multiplication, and transposition are essential for working with datasets, images, word embeddings, and more.
- Neural networks, principal component analysis (PCA), and linear regression are all built on linear algebra.

---

## Key Concepts

### 1. Scalars, Vectors, and Matrices

- **Scalar:** A single number (e.g., 5).
- **Vector:** A 1D array of numbers (e.g., `[2, 3, 5]`).
- **Matrix:** A 2D array (table) of numbers, with rows and columns.

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/scalar-vector-matrix.png" alt="Scalar, Vector, Matrix Diagram" width="420"/>
</p>

---

### 2. Tensors

- A **tensor** is a generalization:  
  - 0D: Scalar  
  - 1D: Vector  
  - 2D: Matrix  
  - 3D or higher: Tensor (e.g., a color image is a 3D tensor—height × width × color channels)

---

### 3. Vectors in ML

- Used for feature representation (e.g., each data point as a vector of features).
- **Example:**  
  - `[height, weight, age] = [170, 65, 23]`

**Code Example:**

```python
import numpy as np

vec = np.array([170, 65, 23])  # Feature vector for a person
print("Vector shape:", vec.shape)
````

---

### 4. Matrix Operations

#### a) Matrix Addition

* Add two matrices elementwise (must have same shape).

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print("A + B =\n", A + B)
```

#### b) Matrix Multiplication

* For ML, matrix multiplication combines data and model weights, or transforms data.
* Multiply `A (m x n)` by `B (n x p)` to get `C (m x p)`.

```python
C = np.dot(A, B)
print("A * B =\n", C)
```

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/matrix-multiplication.png" alt="Matrix Multiplication Example" width="420"/>
</p>

#### c) Transpose

* Flips a matrix over its diagonal.

```python
print("A Transpose =\n", A.T)
```

---

### 5. Dot Product

* **Vectors:** The dot product combines two vectors into a single number.

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print("Dot product:", np.dot(a, b))
# Output: 1*4 + 2*5 + 3*6 = 32
```

* Used in ML for:

  * Calculating similarity between vectors (cosine similarity)
  * Neural network forward passes

---

### 6. Identity and Inverse Matrices

* **Identity matrix:** Square matrix with 1s on diagonal, 0s elsewhere.

  * Acts like "1" for matrices: `A * I = A`

```python
I = np.eye(3)
print("Identity matrix:\n", I)
```

* **Inverse matrix:** Matrix that "undoes" the effect of another: `A * A^-1 = I`

  * Used in solving linear equations.

```python
A = np.array([[1, 2], [3, 4]])
A_inv = np.linalg.inv(A)
print("Inverse of A:\n", A_inv)
```

---

### 7. Eigenvalues and Eigenvectors

* Key for understanding dimensionality reduction (e.g., PCA).
* Eigenvectors indicate “main directions” of data; eigenvalues indicate their importance.

```python
vals, vecs = np.linalg.eig(A)
print("Eigenvalues:", vals)
print("Eigenvectors:\n", vecs)
```

---

## Where Linear Algebra Appears in ML

* **Linear Regression:**
  Model is solved using matrix multiplication and inversion.
* **Neural Networks:**
  Inputs, weights, activations are all vectors and matrices.
* **PCA:**
  Uses eigenvalues/eigenvectors for dimensionality reduction.
* **Embeddings:**
  Words/images represented as high-dimensional vectors.

---

## Visual: Linear Algebra in Machine Learning

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/linear-algebra-in-ml.png" alt="Linear Algebra in ML Diagram" width="500"/>
</p>

---

## Further Reading & References

* [Stanford CS231n: Linear Algebra Review](https://cs231n.github.io/linear-algebra/)
* [DeepLearning.ai: Linear Algebra for Machine Learning](https://www.deeplearning.ai/resources/linear-algebra-for-machine-learning/)
* [Khan Academy: Linear Algebra](https://www.khanacademy.org/math/linear-algebra)
* [3Blue1Brown: Essence of Linear Algebra (Video Series)](https://www.3blue1brown.com/lessons/linear-algebra)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>
