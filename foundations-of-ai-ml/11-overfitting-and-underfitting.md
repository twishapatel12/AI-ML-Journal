![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# Overfitting and Underfitting in Machine Learning

---

## Introduction

Two of the most important challenges in machine learning are **overfitting** and **underfitting**.  
Both affect how well your model can make accurate predictions on new, unseen data.  
Understanding, detecting, and preventing overfitting and underfitting are essential steps in any ML workflow.

---

## What is Underfitting?

**Underfitting** happens when a model is too simple to capture the patterns in the training data.  
It performs poorly on both the training set and the test set.

**Symptoms:**
- High training error and high test error
- Model cannot learn the underlying relationship

**Technical Cause:**  
- Model has **high bias** (see previous lesson)
- Example: Fitting a straight line to data that is clearly curved

**Example Code (Underfitting):**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate quadratic data
X = np.arange(-3, 3, 0.5).reshape(-1, 1)
y = X.flatten()**2 + np.random.randn(*X.flatten().shape) * 2

model = LinearRegression()  # Linear, too simple for quadratic data
model.fit(X, y)
plt.scatter(X, y, label='Data')
plt.plot(X, model.predict(X), color='red', label='Underfitting Model')
plt.title('Underfitting Example')
plt.legend()
plt.show()
````

---

## What is Overfitting?

**Overfitting** happens when a model is too complex and fits the noise in the training data, not just the underlying pattern.
It performs very well on the training set but poorly on the test set.

**Symptoms:**

* Low training error, but high test error
* Model learns “randomness” or noise

**Technical Cause:**

* Model has **high variance**
* Example: Fitting a 10th-degree polynomial to just a few data points

**Example Code (Overfitting):**

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly = PolynomialFeatures(degree=10)
X_poly = poly.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)
plt.scatter(X, y, label='Data')
plt.plot(X, model.predict(X_poly), color='green', label='Overfitting Model')
plt.title('Overfitting Example')
plt.legend()
plt.show()
```

---

## Visual: Underfitting vs Overfitting

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/underfitting-vs-overfitting-diagram.png" alt="Underfitting vs Overfitting Diagram" width="520"/>
</p>

**Figure:**

* **Underfitting:** Model is too simple and misses the pattern
* **Overfitting:** Model is too complex and chases the noise
* **Good Fit:** Model captures the trend without memorizing the data

---

## Error Curves: Model Performance vs Complexity

When plotting **error on the training set** and **error on the test set** against model complexity:

* Underfitting: Both errors are high (left side, too simple)
* Good fit: Training and test errors are both low (middle, just right)
* Overfitting: Training error keeps dropping, but test error goes up again (right side, too complex)

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/error-vs-complexity.png" alt="Error vs Complexity Diagram" width="520"/>
</p>

---

## How to Detect Overfitting and Underfitting

* **Underfitting:**

  * Low accuracy on both train and test sets
  * High bias (model is too simple)
* **Overfitting:**

  * High accuracy on train, low on test
  * High variance (model is too complex)

**Use validation data and cross-validation to check for these issues.**

---

## How to Fix Underfitting

* Use a more complex model (e.g., add more features or use non-linear models)
* Reduce regularization strength
* Increase training time (for neural networks)
* Improve feature engineering

---

## How to Fix Overfitting

* Use a simpler model
* Get more training data
* Use regularization (L1, L2, dropout)
* Use cross-validation for model selection
* Early stopping (for neural networks)
* Prune decision trees

**Example: Add regularization to control overfitting:**

```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)  # Regularization strength
model.fit(X, y)
```

---

## Real-World Analogy

Imagine learning for an exam:

* **Underfitting:** You barely study, so you don’t understand the material or the test
* **Overfitting:** You memorize the exact questions from last year but can’t answer new ones
* **Good fit:** You understand the concepts and can apply them to new problems

---

## Further Reading and References

* [Scikit-learn User Guide: Underfitting vs Overfitting](https://scikit-learn.org/stable/underfitting_overfitting.html)
* [Google ML Crash Course: Overfitting](https://developers.google.com/machine-learning/crash-course/regularization-for-simplicity/video-lecture)
* [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (Aurélien Géron)](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>
