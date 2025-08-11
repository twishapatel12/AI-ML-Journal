![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# Matrix Operations for Machine Learning

---

## Introduction

Matrices are at the core of almost all machine learning (ML) algorithms.  
Matrix operations allow us to represent and manipulate data efficiently—especially for tasks like data transformation, neural network computation, image processing, and optimization.

This guide covers the most important matrix operations for ML, their meanings, and real code examples.

---

## What is a Matrix?

A **matrix** is a rectangular array of numbers, arranged in rows and columns.

Example:  
A 3x2 matrix (3 rows, 2 columns):

$$
A = \begin{bmatrix}
1 & 2 \\
3 & 4 \\
5 & 6 \\
\end{bmatrix}
$$

**In Python (NumPy):**

```python
import numpy as np
A = np.array([[1, 2],
              [3, 4],
              [5, 6]])
print("Matrix A:\n", A)
````

---

## 1. Matrix Addition and Subtraction

You can add or subtract matrices if they have the same shape—just add/subtract each element.

**Example:**

$$
B = \begin{bmatrix}
7 & 8 \\
9 & 10 \\
11 & 12 \\
\end{bmatrix}
$$

$$
A + B = \begin{bmatrix}
1+7 & 2+8 \\
3+9 & 4+10 \\
5+11 & 6+12 \\
\end{bmatrix}
$$

**Code:**

```python
B = np.array([[7, 8],
              [9, 10],
              [11, 12]])
print("A + B =\n", A + B)
```

---

## 2. Scalar Multiplication

Multiply every element by the same number (scalar).

```python
print("2 * A =\n", 2 * A)
```

---

## 3. Matrix Transpose

Flips a matrix over its diagonal.
Rows become columns and vice versa.

$$
A^T = \begin{bmatrix}
1 & 3 & 5 \\
2 & 4 & 6 \\
\end{bmatrix}
$$

**Code:**

```python
print("A Transpose =\n", A.T)
```

---

## 4. Matrix Multiplication (Dot Product)

**Not** the same as elementwise multiplication!
If $A$ is of shape (m x n) and $B$ is of shape (n x p),
then their product $C = AB$ will have shape (m x p).

**Example:**

$$
C = A_{(3x2)} \times B_{(2x2)} = \begin{bmatrix}
1 & 2 \\
3 & 4 \\
5 & 6 \\
\end{bmatrix}
\times
\begin{bmatrix}
7 & 8 \\
9 & 10 \\
\end{bmatrix}
$$

**Code:**

```python
A = np.array([[1, 2],
              [3, 4],
              [5, 6]])
B = np.array([[7, 8],
              [9, 10]])
C = np.dot(A, B)
print("A * B =\n", C)
```

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/matrix-multiplication.png" alt="Matrix Multiplication Example" width="420"/>
</p>

---

## 5. Identity Matrix

A **square matrix** with 1s on the diagonal and 0s elsewhere.
It acts like “1” in matrix multiplication: $A \times I = A$.

```python
I = np.eye(3)
print("Identity Matrix:\n", I)
```

---

## 6. Inverse Matrix

The inverse of a square matrix $A$ (if it exists) is $A^{-1}$ such that
$A \times A^{-1} = I$.

**Used for:** Solving systems of equations, some ML algorithms (not always needed in large-scale ML).

```python
A = np.array([[1, 2], [3, 4]])
A_inv = np.linalg.inv(A)
print("Inverse of A:\n", A_inv)
```

---

## 7. Elementwise (Hadamard) Product

Multiply corresponding elements of two matrices of the same shape.

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[10, 20], [30, 40]])
print("Elementwise product:\n", A * B)
```

---

## 8. Slicing and Indexing

Extract rows, columns, or blocks from a matrix (useful for data selection).

```python
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("First row:", A[0])
print("First column:", A[:, 0])
print("2x2 top-left block:\n", A[:2, :2])
```

---

## Real World Example: Images as Matrices

* A grayscale image is a matrix (pixels = numbers 0–255).
* A color image is a 3D tensor: height x width x color channels (R, G, B).

---

## Where Matrix Operations are Used in ML

* **Data transformation:** Feature scaling, PCA, word embeddings
* **Linear regression:** Closed-form solution uses matrix inversion and multiplication
* **Neural networks:** Forward and backward passes are matrix multiplications
* **Clustering and similarity:** Distance calculations between feature vectors

---

## Visual: Matrix Operations Flow

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/matrix-operations-flow.png" alt="Matrix Operations Flow Diagram" width="500"/>
</p>

---

## Further Reading & References

* [NumPy User Guide: Array Manipulation](https://numpy.org/doc/stable/user/absolute_beginners.html)
* [Khan Academy: Matrix Operations](https://www.khanacademy.org/math/linear-algebra/matrix-transformations)
* [Essence of Linear Algebra (3Blue1Brown)](https://www.3blue1brown.com/lessons/linear-algebra)
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