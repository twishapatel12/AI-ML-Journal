![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# Bias vs Variance in Machine Learning

---

## Introduction

Bias and variance are two fundamental sources of error in any machine learning model.  
Understanding the trade-off between them is essential to build models that are accurate, reliable, and not over- or under-fitted.  
In this guide, you’ll learn what bias and variance mean, how they affect model performance, and how to balance them.

---

## What is Bias?

**Bias** refers to the error introduced by approximating a real-world problem with a simplified model.  
- High bias means the model makes strong assumptions and fails to capture the underlying patterns in the data.
- Models with high bias are often **too simple** (e.g., using a straight line to fit a curved relationship).

**Symptoms:**  
- Underfitting (model performs poorly on both training and test data)
- High training error

**Example:**

```python
# High bias: fitting a linear model to quadratic data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Quadratic data
X = np.arange(-3, 3, 0.5).reshape(-1, 1)
y = X**2 + np.random.randn(*X.shape)

model = LinearRegression()
model.fit(X, y)
plt.scatter(X, y, label='Data')
plt.plot(X, model.predict(X), color='red', label='High Bias Model')
plt.legend()
plt.title("High Bias Example: Underfitting")
plt.show()
````

---

## What is Variance?

**Variance** refers to how much the model’s predictions change when using different training data.

* High variance means the model is very sensitive to fluctuations in the training set and may capture noise instead of the actual pattern.
* Models with high variance are often **too complex** (e.g., a very wiggly curve passing through every data point).

**Symptoms:**

* Overfitting (model does very well on training data but poorly on test data)
* Low training error, high test error

**Example:**

```python
# High variance: fitting a polynomial model with too many degrees
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=10)
X_poly = poly.fit_transform(X)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_poly, y)
plt.scatter(X, y, label='Data')
plt.plot(X, model.predict(X_poly), color='green', label='High Variance Model')
plt.legend()
plt.title("High Variance Example: Overfitting")
plt.show()
```

---

## Visual: Bias vs Variance

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/bias-vs-variance-diagram.png" alt="Bias vs Variance Diagram" width="520"/>
</p>

**Figure:**

* **High bias:** Model is too simple (underfitting)
* **High variance:** Model is too complex (overfitting)
* **Just right:** Good balance between bias and variance (generalizes well)

---

## The Bias-Variance Trade-Off

* **Reducing bias** usually means making the model more complex (risking higher variance)
* **Reducing variance** usually means making the model simpler or using more data (risking higher bias)

The goal is to find a “sweet spot” where the model captures real patterns but ignores noise.

|               | Training Error | Test Error | Real-world Performance |
| ------------- | :------------: | :--------: | :--------------------: |
| High Bias     |      High      |    High    |      Underfitting      |
| High Variance |       Low      |    High    |       Overfitting      |
| Balanced      |       Low      |     Low    |    Generalizes well    |

---

## How to Control Bias and Variance

**To reduce bias (underfitting):**

* Use a more complex model
* Add more or better features
* Reduce regularization

**To reduce variance (overfitting):**

* Use a simpler model
* Use more data for training
* Use regularization (e.g., L1/L2 penalty)
* Use cross-validation

**Example: Using Regularization**

```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)  # alpha controls regularization strength
model.fit(X, y)
```

---

## Visual: Error vs Model Complexity

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/error-vs-complexity.png" alt="Error vs Model Complexity" width="520"/>
</p>

**Figure:**
As model complexity increases:

* **Bias decreases** (better fit to training data)
* **Variance increases** (risk of overfitting)
* The best generalization happens at the minimum point of the test error curve.

---

## Real-World Analogy

Think of bias as a student who always gives the same wrong answer because they never learned the full lesson.
Variance is a student who memorizes every detail of their notes—even the mistakes—so they do well in practice but poorly on the actual test.

---

## Further Reading and References

* [Google Developers: Reducing Loss - Bias and Variance](https://developers.google.com/machine-learning/crash-course/reducing-loss/bias-variance)
* [DeepAI: Bias vs Variance](https://deepai.org/machine-learning-glossary-and-terms/bias-variance-tradeoff)
* [Scikit-learn User Guide: Underfitting vs Overfitting](https://scikit-learn.org/stable/underfitting_overfitting.html)
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
