![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# Multivariate Distributions in Machine Learning

---

## Introduction

In machine learning, we often work with **multiple variables at the same time** (e.g., height and weight, or pixel values in an image).  
**Multivariate distributions** describe the probability relationships between two or more random variables—capturing not just their individual properties, but also how they interact.

Understanding these distributions is key for tasks like anomaly detection, dimensionality reduction, and generative modeling.

---

## What is a Multivariate Distribution?

A **multivariate distribution** is a probability distribution for a vector of random variables.

- **Univariate distribution:** Probability for a single variable (e.g., height).
- **Multivariate distribution:** Probability for multiple variables **together** (e.g., height and weight, or all pixel intensities in a 2D image).

---

### Real-World Example

Suppose you’re analyzing heights and weights of people.  
A univariate distribution tells you the probability of someone being a certain height.  
A **bivariate** (2-variable) distribution tells you the probability of someone being both a certain height **and** weight.

---

## Key Concepts

### 1. Joint Probability

The probability of two or more events occurring **together**.

$$
P(X = x, Y = y)
$$

- Example: Probability that a person is 170 cm tall **and** 65 kg in weight.

---

### 2. Marginal Probability

The probability of one variable, **regardless** of the other(s):

$$
P(X = x) = \sum_{y} P(X = x, Y = y)
$$

---

### 3. Conditional Probability

The probability of one variable, **given** another:

$$
P(X = x \mid Y = y)
$$

- Example: Probability someone is 170 cm tall, **given** they weigh 65 kg.

---

## The Multivariate Normal (Gaussian) Distribution

The most common multivariate distribution in ML is the **multivariate normal distribution** (generalizes the bell curve to higher dimensions).

**Probability density function:**

$$
f(x) = \frac{1}{\sqrt{(2\pi)^k |\Sigma|}} \exp\left( -\frac{1}{2}(x-\mu)^T \Sigma^{-1} (x-\mu) \right)
$$

Where:
- $x$: Vector of variables (e.g., height and weight)
- $\mu$: Mean vector (center of distribution)
- $\Sigma$: Covariance matrix (how variables move together)

---

### Visual: Multivariate Normal Distribution

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/multivariate-normal.png" alt="Multivariate Normal Distribution" width="500"/>
</p>

*Shows a 3D bell-shaped surface for two variables (bivariate normal).*

---

## Covariance Matrix

- Measures how much variables “move together.”
- **Diagonal elements:** Variance of each variable.
- **Off-diagonal elements:** Covariance between pairs.

**If covariance is positive:** As one variable increases, so does the other.  
**If negative:** As one increases, the other decreases.

**Code Example:**

```python
import numpy as np

data = np.array([[170, 65], [180, 70], [160, 60]])  # [height, weight]
print("Covariance matrix:\n", np.cov(data.T))
````

---

## Why Do Multivariate Distributions Matter in ML?

* **Anomaly Detection:** Identify unusual data points by seeing if their combination is unlikely.
* **Gaussian Mixture Models:** Used in clustering and density estimation.
* **Dimensionality Reduction (PCA):** Analyzes covariance structure of multivariate data.
* **Generative Models:** Used in sampling new data, data imputation, and generative neural nets (e.g., VAEs).

---

## Code Example: Drawing Samples from a Multivariate Normal

```python
import numpy as np
import matplotlib.pyplot as plt

mean = [0, 0]
cov = [[1, 0.8], [0.8, 1]]  # Strong positive correlation
samples = np.random.multivariate_normal(mean, cov, 500)

plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Samples from a Bivariate Normal Distribution")
plt.show()
```

---

## Visual: Covariance Shapes

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/covariance-shapes.png" alt="Covariance Shapes" width="500"/>
</p>

*Ellipses showing different correlation structures: round (no correlation), tilted (positive or negative correlation).*

---

## Summary

* Multivariate distributions capture relationships among multiple variables.
* Covariance tells us about linear relationships between variables.
* The multivariate normal is central to many ML algorithms, but other multivariate distributions (e.g., multinomial, exponential family) are also used.

---

## Further Reading & References

* [Khan Academy: Multivariate Distributions](https://www.khanacademy.org/math/statistics-probability/probability-library)
* [Wikipedia: Multivariate Normal Distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)
* [DeepLearning.ai: Mathematics for ML](https://www.deeplearning.ai/resources/mathematics-for-machine-learning/)
* [Stanford CS229: Probability Review](https://see.stanford.edu/materials/aimlcs229/cs229-prob.pdf)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>