![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# Ridge and Lasso Regression

---

## Introduction

**Ridge** and **Lasso** regression are two powerful techniques that extend linear regression by adding **regularization**.  
Regularization helps prevent overfitting—when a model fits the training data too closely and performs poorly on new data.

Both Ridge and Lasso are especially useful when you have many features (high-dimensional data) or when features are highly correlated.

---

## Why Use Regularized Regression?

- **Ordinary Linear Regression** can overfit, especially with many features, noise, or multicollinearity (features that are highly correlated).
- **Regularization** adds a penalty to large model coefficients, making the model “simpler” and more generalizable.

---

## Ridge Regression (L2 Regularization)

- **Ridge regression** adds a penalty equal to the sum of the squared coefficients (L2 norm).
- It shrinks all coefficients toward zero, but rarely makes them exactly zero.
- Useful for reducing model complexity and handling multicollinearity.

**Mathematical formula:**

$$
\text{Loss} = \text{MSE} + \lambda \sum_{j} w_j^2
$$

- $\lambda$ (alpha in code) controls the strength of the penalty.  
  Higher $\lambda$: stronger regularization.

**Code Example:**

```python
from sklearn.linear_model import Ridge
import numpy as np

X = np.random.randn(100, 5)
y = X @ np.array([2, 0, 0, 3, 0]) + np.random.randn(100) * 0.5

ridge = Ridge(alpha=1.0)
ridge.fit(X, y)
print("Ridge coefficients:", ridge.coef_)
````

---

## Lasso Regression (L1 Regularization)

* **Lasso regression** adds a penalty equal to the sum of the absolute values of coefficients (L1 norm).
* It can shrink some coefficients exactly to zero, performing **feature selection** (identifies important features).

**Mathematical formula:**

$$
\text{Loss} = \text{MSE} + \lambda \sum_{j} |w_j|
$$

* $\lambda$ (alpha in code) controls regularization strength.

**Code Example:**

```python
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.5)
lasso.fit(X, y)
print("Lasso coefficients:", lasso.coef_)
```

---

## Visual: Ridge vs. Lasso Effect

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/ridge-lasso-effect.png" alt="Ridge vs Lasso Coefficient Effect" width="480"/>
</p>

*Lasso (L1) sets some coefficients to exactly zero; Ridge (L2) shrinks all coefficients but rarely zeroes them.*

---

## When to Use Ridge vs. Lasso

| Scenario                         | Ridge | Lasso |
| -------------------------------- | ----- | ----- |
| Most features are useful         | ✔     |       |
| Only a few features are relevant |       | ✔     |
| Features are highly correlated   | ✔     |       |
| Want automatic feature selection |       | ✔     |

You can also combine both penalties using **Elastic Net**.

---

## Hyperparameter Tuning

* The **regularization parameter** ($\lambda$ or `alpha`) controls how much regularization is applied.
* Use **cross-validation** to choose the best value (try several and pick the one with best validation performance).

---

## Real-World Use Cases

* Predicting house prices with lots of features (location, size, amenities).
* Genetics: finding which genes impact disease.
* Finance: modeling with highly correlated financial indicators.

---

## Best Practices

* Standardize or normalize features before applying regularized regression.
* Use Lasso if you expect many irrelevant features (automatic selection).
* Use Ridge for stability when features are correlated.
* Try Elastic Net if you’re unsure—combines benefits of both.

---

## Further Reading & References

* [Scikit-learn: Ridge and Lasso Regression](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression)
* [Wikipedia: Ridge Regression](https://en.wikipedia.org/wiki/Tikhonov_regularization)
* [Wikipedia: Lasso (Statistics)](https://en.wikipedia.org/wiki/Lasso_%28statistics%29)
* [Analytics Vidhya: Lasso and Ridge Regression Guide](https://www.analyticsvidhya.com/blog/2021/08/ridge-and-lasso-regression/)
* [Khan Academy: Regularization](https://www.khanacademy.org/math/statistics-probability)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>