![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# Polynomial Regression

---

## Introduction

**Polynomial regression** is an extension of linear regression that allows us to fit curved (non-linear) relationships between features and the target variable.  
While linear regression draws a straight line, polynomial regression can draw a curve—by including powers of the input feature(s).

---

## When to Use Polynomial Regression

- When your data shows a clear curve or non-linear trend that a straight line cannot fit well.
- When residuals (differences between actual and predicted values) show a pattern rather than random scatter (indicating linear regression isn’t enough).

---

## How Does Polynomial Regression Work?

Polynomial regression transforms the input features by adding powers (squared, cubed, etc.) of those features.

**For one variable:**

- **Linear regression:** $y = w_0 + w_1x$
- **Quadratic regression:** $y = w_0 + w_1x + w_2x^2$
- **Cubic regression:** $y = w_0 + w_1x + w_2x^2 + w_3x^3$
- **And so on...**

You still use ordinary linear regression algorithms—the trick is to expand your feature set.

---

## Visual: Linear vs. Polynomial Regression

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/polynomial-regression.png" alt="Polynomial Regression Example" width="480"/>
</p>

*Shows both a straight line (linear) and a curved fit (polynomial) on the same data.*

---

## Code Example: Polynomial Regression in Python

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Generate some data (with a curve)
X = np.linspace(0, 10, 20).reshape(-1, 1)
y = 2 + 0.5 * X.flatten()**2 - 3 * X.flatten() + np.random.randn(20) * 3

# Fit linear regression
lin_model = LinearRegression()
lin_model.fit(X, y)
y_lin_pred = lin_model.predict(X)

# Fit polynomial regression (degree=2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_poly_pred = poly_model.predict(X_poly)

plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, y_lin_pred, color='red', label='Linear Fit')
plt.plot(X, y_poly_pred, color='green', label='Polynomial Fit')
plt.legend()
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear vs Polynomial Regression")
plt.show()
````

---

## Choosing the Degree of the Polynomial

* **Degree 1:** Linear (straight line)
* **Degree 2:** Quadratic (single bend)
* **Degree 3+:** More complex curves

**Warning:**

* Higher degrees can fit the training data very closely (low bias), but risk **overfitting** (high variance).
* Use cross-validation to choose the right degree for your data.

---

## Advantages and Limitations

**Advantages:**

* Can fit non-linear relationships while still using linear models.
* Easy to implement with feature engineering.

**Limitations:**

* Prone to overfitting with high-degree polynomials.
* Can become unstable outside the range of training data (extrapolation).
* Not suitable for very high-dimensional or noisy data.

---

## Real-World Applications

* Modeling growth curves (biology, economics)
* Physics and engineering (e.g., projectile motion)
* Fitting trends in business or scientific data

---

## Best Practices

* Plot your data and fit before/after using polynomial regression.
* Use regularization (Ridge/Lasso) to reduce overfitting if using high degrees.
* Always evaluate on test/validation data, not just training data.

---

## Further Reading & References

* [Scikit-learn: Polynomial Regression Example](https://scikit-learn.org/stable/auto_examples/linear_model/plot_polynomial_interpolation.html)
* [Wikipedia: Polynomial Regression](https://en.wikipedia.org/wiki/Polynomial_regression)
* [Analytics Vidhya: Polynomial Regression](https://www.analyticsvidhya.com/blog/2020/03/polynomial-regression-python/)
* [Khan Academy: Polynomial Functions](https://www.khanacademy.org/math/algebra2/polynomial-functions)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>