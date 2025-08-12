![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# Loss Functions in Machine Learning

---

## Introduction

A **loss function** is a mathematical formula that measures how well (or badly) a machine learning model is performing.  
During training, ML algorithms try to **minimize the loss**—that is, they tweak the model’s parameters to make predictions closer to the true answers.

Loss functions are at the heart of model optimization, from linear regression to deep neural networks.

---

## Why Are Loss Functions Important?

- They define what it means for a model to be “good” or “bad.”
- Guide the optimization process (e.g., gradient descent).
- Affect the speed and stability of learning.
- Different problems (regression, classification, etc.) use different loss functions.

---

## Main Types of Loss Functions

### 1. Regression Loss Functions

Used when predicting **numbers** (continuous values).

#### a. Mean Squared Error (MSE)

\[
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

- Penalizes large errors more than small ones (squares the difference).
- Very common in linear regression.

**Code Example:**

```python
import numpy as np
y_true = [3, 5, 2.5]
y_pred = [2.5, 5, 4]
mse = np.mean((np.array(y_true) - np.array(y_pred))**2)
print("MSE:", mse)
````

#### b. Mean Absolute Error (MAE)

$$
MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

* Takes the average of absolute errors.
* Less sensitive to outliers than MSE.

---

### 2. Classification Loss Functions

Used when predicting **categories** (e.g., spam vs not spam).

#### a. Cross-Entropy Loss (Log Loss)

$$
\text{Cross-Entropy} = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i)]
$$

* Measures the difference between true class labels and predicted probabilities.
* Used in logistic regression, neural networks, and deep learning classifiers.

**Code Example:**

```python
import numpy as np

y_true = [1, 0, 1]
y_pred = [0.9, 0.1, 0.8]
cross_entropy = -np.mean([
    y_t * np.log(y_p) + (1 - y_t) * np.log(1 - y_p)
    for y_t, y_p in zip(y_true, y_pred)
])
print("Cross-Entropy Loss:", cross_entropy)
```

#### b. Hinge Loss

* Used in Support Vector Machines (SVMs) for binary classification.
* Penalizes predictions on the wrong side of the decision boundary.

---

### 3. Other Common Loss Functions

#### a. Huber Loss

* Mixes MSE and MAE: quadratic for small errors, linear for large errors.
* Useful for regression with outliers.

#### b. Kullback-Leibler (KL) Divergence

* Measures how one probability distribution diverges from another.
* Used in advanced models like variational autoencoders and reinforcement learning.

---

## Visual: Loss Functions Comparison

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/loss-functions-comparison.png" alt="Loss Functions Comparison" width="500"/>
</p>

*Shows how MSE, MAE, and Huber Loss respond to prediction errors.*

---

## How Loss Functions Are Used

1. **Calculate loss:** For current predictions.
2. **Compute gradients:** How should each parameter change to reduce loss?
3. **Update parameters:** Use gradient descent (or a similar optimizer).
4. **Repeat:** Until the loss is as small as possible or stops improving.

---

## Real-World Example: Choosing a Loss Function

* **House Price Prediction:**
  Use MSE or MAE (regression problem).
* **Spam Email Detection:**
  Use cross-entropy loss (binary classification).
* **Image Classification (many classes):**
  Use categorical cross-entropy.

---

## Best Practices

* Match the loss function to your **problem type**.
* Watch for outliers: MSE is sensitive, MAE is robust.
* For deep learning, cross-entropy is usually best for classification.
* For imbalanced datasets, consider using custom loss or class weights.

---

## Further Reading & References

* [DeepLearning.ai: Loss Functions](https://www.deeplearning.ai/resources/loss-functions-in-machine-learning/)
* [Scikit-learn: Loss Functions Documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)
* [Wikipedia: Loss Function](https://en.wikipedia.org/wiki/Loss_function)
* [Analytics Vidhya: Most Common Loss Functions](https://www.analyticsvidhya.com/blog/2021/06/common-loss-functions-in-machine-learning/)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>