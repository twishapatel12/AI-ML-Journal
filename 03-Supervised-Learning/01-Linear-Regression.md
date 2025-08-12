![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# Linear Regression

---

## Introduction

**Linear regression** is one of the simplest and most widely used algorithms in machine learning.  
It is used for predicting a continuous value (like house price, temperature, salary) from one or more input features.

Linear regression tries to fit a straight line (or plane/hyperplane in higher dimensions) through the data points that best predicts the output.

---

## When Do You Use Linear Regression?

- When you want to **predict a number** based on known features (e.g., predict a student’s exam score from study hours).
- When you believe the relationship between the inputs and output is roughly linear (can be drawn as a straight line).

---

## How Does Linear Regression Work?

**Simple linear regression** fits a line:

$$
y = mx + c
$$
- $y$: predicted value (target)
- $x$: input feature
- $m$: slope (how much y changes as x changes)
- $c$: intercept (value of y when x = 0)

**Multiple linear regression** (more than one input):

$$
y = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + c
$$

---

## Visual: Linear Regression Fit

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/linear-regression-fit.png" alt="Linear Regression Fit" width="480"/>
</p>

*The blue dots are data points. The red line is the best-fit line found by linear regression.*

---

## How Does the Algorithm Find the Best Line?

- It finds the line (or plane) that **minimizes the difference** between the actual and predicted values.
- The difference for each point is called the **residual** (error).
- It usually uses **mean squared error (MSE)** as the measure of error:

  $$
  MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
  $$ 
- Linear regression uses calculus (optimization) to find the slope and intercept that minimize MSE.

---

## Code Example: Linear Regression in Python

```python
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Example: Study hours vs. Exam score
X = np.array([[1], [2], [3], [4], [5]])  # Hours studied
y = np.array([40, 50, 60, 65, 80])       # Exam score

model = LinearRegression()
model.fit(X, y)
pred = model.predict(X)

plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, pred, color='red', label='Regression Line')
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.legend()
plt.show()

print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)
````

---

## Multiple Linear Regression

If you have several features (e.g., square footage, number of bedrooms, and age of house for house price prediction):

$$
y = w_1 x_1 + w_2 x_2 + w_3 x_3 + c
$$

The algorithm finds the best values for all the weights $w_i$ and the intercept $c$.

---

## Assumptions of Linear Regression

* The relationship between features and target is linear.
* Errors are normally distributed and have constant variance (homoscedasticity).
* Features are independent of each other.
* No extreme outliers.

---

## Advantages and Limitations

**Advantages:**

* Simple, fast, and easy to interpret.
* Good baseline model for regression tasks.

**Limitations:**

* Can’t model non-linear relationships.
* Sensitive to outliers and irrelevant features.
* May underfit if the true relationship is complex.

---

## Real-World Use Cases

* Predicting house prices from features.
* Forecasting sales or demand.
* Estimating risk or insurance premiums.
* Predicting salary based on education and experience.

---

## Further Reading & References

* [Scikit-learn: Linear Regression Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
* [Khan Academy: Linear Regression](https://www.khanacademy.org/math/statistics-probability/describing-relationships-quantitative-data)
* [Wikipedia: Linear Regression](https://en.wikipedia.org/wiki/Linear_regression)
* [Analytics Vidhya: Linear Regression Guide](https://www.analyticsvidhya.com/blog/2016/01/guide-on-linear-regression/)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>