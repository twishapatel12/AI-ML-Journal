![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# Regularization in Machine Learning: L1 & L2 Techniques

---

## Introduction

**Regularization** is a set of techniques used in machine learning to prevent overfitting.  
Overfitting happens when a model learns the noise in the training data instead of just the useful patterns—making it perform poorly on new, unseen data.

Regularization helps by making the model “simpler” and discouraging it from relying too much on any single feature or parameter.

---

## Why Do We Need Regularization?

- **Complex models can memorize training data.**
- This makes them very accurate on training data but bad on test data (poor generalization).
- Regularization **adds a penalty** for large or complex model weights, so the model favors “simpler” solutions.

---

## The Two Most Common Types

### 1. L1 Regularization (Lasso)

- Adds a penalty equal to the **absolute value** of the weights.
- The penalty term is \( \lambda \sum_{j} |w_j| \), where \( w_j \) are the weights and \( \lambda \) controls the strength.
- Encourages some weights to become exactly zero (feature selection).

**Mathematical Form (for linear regression):**

\[
\text{Loss} = \text{MSE} + \lambda \sum_{j} |w_j|
\]

**Code Example:**

```python
from sklearn.linear_model import Lasso

# X = features, y = target values
model = Lasso(alpha=0.1)  # alpha is the regularization strength (λ)
model.fit(X, y)
````

---

### 2. L2 Regularization (Ridge)

* Adds a penalty equal to the **square** of the weights.
* The penalty term is $\lambda \sum_{j} w_j^2$.
* Encourages small, but non-zero, weights (shrinks them toward zero).

**Mathematical Form (for linear regression):**

$$
\text{Loss} = \text{MSE} + \lambda \sum_{j} w_j^2
$$

**Code Example:**

```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=0.1)  # alpha is the regularization strength (λ)
model.fit(X, y)
```

---

## Visual: L1 vs L2 Regularization Effects

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/l1-vs-l2-regularization.png" alt="L1 vs L2 Regularization Comparison" width="500"/>
</p>

*L1 (Lasso) produces sparse models (many weights exactly zero); L2 (Ridge) produces small weights (shrinks but rarely zeroes).*

---

## Choosing Between L1 and L2

* **L1 (Lasso):**

  * Good for feature selection.
  * Useful if you think many features are irrelevant.
* **L2 (Ridge):**

  * Good for most situations.
  * Keeps all features, but shrinks their impact.
* **Elastic Net:**

  * Combines both L1 and L2 penalties. Use when you want both feature selection and smooth weights.

---

## When to Use Regularization

* Always try regularization when your model overfits (training error much lower than test error).
* Especially important for:

  * High-dimensional data (many features, few examples)
  * Neural networks (deep learning)
  * Polynomial or complex models

---

## How to Set the Regularization Parameter ($\lambda$ or `alpha`)

* **Small λ (weak regularization):** Model is flexible but can overfit.
* **Large λ (strong regularization):** Model is very simple but may underfit.
* Use cross-validation to find the best value.

---

## Code Example: Comparing Lasso and Ridge

```python
import numpy as np
from sklearn.linear_model import Lasso, Ridge

X = np.random.randn(100, 5)
y = X @ np.array([1, 0, 0, 2, 0]) + np.random.randn(100) * 0.1

lasso = Lasso(alpha=0.2)
ridge = Ridge(alpha=0.2)
lasso.fit(X, y)
ridge.fit(X, y)
print("Lasso coefficients:", lasso.coef_)
print("Ridge coefficients:", ridge.coef_)
```

*Notice how Lasso sets some coefficients exactly to zero!*

---

## Real-World Example

Suppose you’re predicting house prices with hundreds of features (e.g., size, age, location, and many one-hot-encoded categories).

* **L1 regularization** might automatically drop irrelevant features, making the model simpler and easier to interpret.
* **L2 regularization** helps ensure no single feature dominates, reducing the risk of overfitting.

---

## Further Reading & References

* [Scikit-learn: Regularization in Linear Models](https://scikit-learn.org/stable/modules/linear_model.html#regularization)
* [DeepLearning.ai: Regularization](https://www.deeplearning.ai/resources/regularization-in-machine-learning/)
* [Khan Academy: L1 and L2 Regularization](https://www.khanacademy.org/math/statistics-probability)
* [Analytics Vidhya: Regularization Techniques](https://www.analyticsvidhya.com/blog/2021/06/regularization-techniques-in-machine-learning/)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>