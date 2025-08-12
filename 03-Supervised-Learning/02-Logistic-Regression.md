![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# Logistic Regression

---

## Introduction

**Logistic regression** is a popular algorithm for classification problems in machine learning.  
Unlike linear regression, which predicts continuous values, logistic regression predicts the **probability that an input belongs to a certain class** (for example: “spam” or “not spam”).

It’s simple, fast, and a standard baseline for binary (yes/no) classification tasks.

---

## When to Use Logistic Regression

- When you want to predict whether an input belongs to one of two classes (binary classification), e.g.:
  - Will a customer buy a product? (yes/no)
  - Is an email spam? (spam/not spam)
  - Will a student pass or fail?
- Also works for multi-class problems (with some extensions).

---

## How Does Logistic Regression Work?

- It computes a weighted sum of the input features, just like linear regression:

$$
z = w_1 x_1 + w_2 x_2 + ... + w_n x_n + c
$$

- It then **applies the sigmoid function** to squash the output between 0 and 1:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

- The output, $\sigma(z)$, is interpreted as the **probability of belonging to the positive class**.

---

## Visual: The Sigmoid (Logistic) Function

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/sigmoid-function.png" alt="Sigmoid Function" width="420"/>
</p>

*The sigmoid curve always outputs values between 0 and 1.*

---

## Making Predictions

- If $\sigma(z) > 0.5$, predict class 1 (positive).
- If $\sigma(z) < 0.5$, predict class 0 (negative).
- The **threshold** can be changed (for imbalanced data, risk tolerance, etc).

---

## Model Training: Loss and Optimization

- **Loss function:** Logistic regression uses **log loss** (cross-entropy loss) to measure error:

$$
\text{Log Loss} = -[y \log(p) + (1-y) \log(1-p)]
$$

- **Optimization:** Uses gradient descent or similar algorithms to find weights $w$ and intercept $c$ that minimize loss.

---

## Code Example: Logistic Regression in Python

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Simple binary example: Exam score vs. Pass (1) or Fail (0)
X = np.array([[50], [60], [65], [70], [75], [80], [85]])
y = np.array([0, 0, 0, 1, 1, 1, 1])

model = LogisticRegression()
model.fit(X, y)

# Predict probabilities
x_test = np.linspace(40, 90, 100).reshape(-1, 1)
probs = model.predict_proba(x_test)[:, 1]

plt.scatter(X, y, color='blue', label='Data')
plt.plot(x_test, probs, color='red', label='Sigmoid Curve')
plt.xlabel("Exam Score")
plt.ylabel("Probability of Passing")
plt.legend()
plt.title("Logistic Regression")
plt.show()
````

---

## Multi-Class Logistic Regression (Softmax Regression)

* Logistic regression can be extended to more than two classes using the **softmax function** (see Mathematics-for-AI-ML/11-Softmax-and-Log-Loss.md).
* Each class gets its own set of weights; the output is a probability distribution over all classes.

---

## Advantages and Limitations

**Advantages:**

* Fast and easy to implement.
* Produces interpretable probability scores.
* Works well with linearly separable data.

**Limitations:**

* Can’t capture non-linear relationships unless you add more features.
* Sensitive to outliers and highly correlated features.
* Less powerful than more complex models (e.g., decision trees, neural nets) for some problems.

---

## Real-World Applications

* Spam email detection.
* Predicting customer churn.
* Medical diagnosis (disease: yes/no).
* Credit scoring (loan default: yes/no).
* Click-through prediction in online ads.

---

## Further Reading & References

* [Scikit-learn: Logistic Regression Documentation](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
* [Khan Academy: Logistic Regression](https://www.khanacademy.org/math/statistics-probability/describing-relationships-quantitative-data)
* [Wikipedia: Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression)
* [Analytics Vidhya: Logistic Regression Guide](https://www.analyticsvidhya.com/blog/2021/08/logistic-regression-detailed-overview/)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>