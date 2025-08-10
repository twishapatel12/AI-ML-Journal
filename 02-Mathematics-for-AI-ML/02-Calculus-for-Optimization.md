![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# Calculus for Optimization in Machine Learning

---

## Introduction

Calculus is the branch of mathematics that studies how things change.  
In machine learning and AI, calculus is essential for understanding **optimization**—the process of adjusting model parameters to make predictions as accurate as possible.

Most modern ML algorithms, especially deep learning, rely on calculus for training, using tools like **gradients** and **derivatives** to improve the model step by step.

---

## Why is Calculus Important in AI and ML?

- **Optimization:**  
  Training a model means minimizing (or maximizing) some objective function, such as the error or “loss” between predictions and actual values.
- **Gradients:**  
  Calculus helps us find the direction and rate of fastest change of a function. In ML, this tells us how to update model parameters to reduce error.
- **Backpropagation:**  
  Neural networks use calculus (the chain rule) to efficiently compute gradients for all weights, allowing deep models to learn from data.

---

## Key Concepts in Calculus for ML

### 1. Functions

A **function** maps inputs (like features in ML) to outputs (like predictions).

Example:  
- In linear regression: `y = mx + c`

---

### 2. Derivatives

The **derivative** of a function tells us how the output changes as the input changes.  
In ML, we use derivatives to figure out how a small change in model parameters (weights) changes the loss.

**Notation:**  
- For a function `f(x)`, the derivative is `f'(x)` or `df/dx`.

---

### 3. Gradient

The **gradient** is a vector of derivatives—one for each parameter.  
- It points in the direction of steepest increase of the function.
- To minimize error, we move in the **opposite** direction of the gradient.

---

### 4. Gradient Descent

**Gradient Descent** is the core optimization algorithm in ML.

- It starts with random values for model parameters.
- Computes the gradient of the loss function with respect to parameters.
- Updates parameters by moving a small step against the gradient.
- Repeats this process until the loss is minimized.

**Mathematical update:**  
`w_new = w_old - learning_rate * gradient`

---

## Example: Gradient Descent for a Simple Function

Suppose we want to minimize the function `f(w) = (w - 3)^2`.

- The minimum occurs at `w = 3`.

**Step-by-step (in Python):**

```python
import numpy as np
import matplotlib.pyplot as plt

# Function and its derivative
def f(w): return (w - 3)**2
def df(w): return 2 * (w - 3)

w = 0.0
learning_rate = 0.1
history = [w]

for i in range(20):
    grad = df(w)
    w = w - learning_rate * grad
    history.append(w)

plt.plot([f(x) for x in history])
plt.title("Loss decreasing during Gradient Descent")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

print("Optimal w found:", w)
````

*This code shows how gradient descent quickly finds the minimum by using the derivative to guide each step.*

---

## How Calculus is Used in AI/ML

* **Linear Regression:**
  Uses calculus to find the best-fitting line by minimizing mean squared error.
* **Logistic Regression:**
  Uses gradient descent to optimize the likelihood of classification.
* **Neural Networks:**
  Use the chain rule for backpropagation to compute gradients of complex, multi-layer models.
* **Support Vector Machines (SVM):**
  Use optimization with constraints, powered by calculus.

---

## Visual: Gradient Descent Illustration

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/gradient-descent-curve.png" alt="Gradient Descent Curve" width="400"/>
</p>

*Shows the loss function curve, the starting point, and arrows illustrating the steps of gradient descent down the slope to the minimum.*

---

## Key Terms

| Term            | Meaning                                                 |
| --------------- | ------------------------------------------------------- |
| Derivative      | How a function output changes with input                |
| Gradient        | Vector of partial derivatives (one for each parameter)  |
| Loss            | The value we want to minimize (e.g., error)             |
| Learning Rate   | Step size for each update                               |
| Backpropagation | Efficient algorithm to compute gradients in neural nets |

---

## Further Reading & References

* [Stanford CS231n: Calculus Review](https://cs231n.github.io/optimization-1/)
* [3Blue1Brown: Gradient Descent Animation](https://www.3blue1brown.com/lessons/gradient-descent)
* [DeepLearning.ai: Gradient Descent](https://www.deeplearning.ai/resources/gradient-descent-for-machine-learning/)
* [Khan Academy: Calculus for Machine Learning](https://www.khanacademy.org/math/calculus-1)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>
