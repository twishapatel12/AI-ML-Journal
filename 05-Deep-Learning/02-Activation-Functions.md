![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# Activation Functions in Neural Networks

---

## Introduction

**Activation functions** are mathematical functions applied to a neuron's output to introduce **non-linearity** into the network.  

Without activation functions, a neural network would just be a **linear model** (a stack of linear transformations).  
Non-linearity allows neural networks to learn **complex patterns** and relationships.

---

## Why Do We Need Activation Functions?

- Enable the network to approximate **non-linear functions**.
- Control the output range of neurons.
- Help with training convergence.
- Allow learning of complex tasks like image recognition and NLP.

---

## Common Activation Functions

### 1. Sigmoid

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

- Outputs values between **0 and 1**.
- Used for probabilities in binary classification.

**Pros:** Smooth, bounded.  
**Cons:** Vanishing gradient for large positive/negative inputs.

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-6, 6, 100)
y = 1 / (1 + np.exp(-x))
plt.plot(x, y)
plt.title("Sigmoid Activation")
plt.show()
````

---

### 2. Hyperbolic Tangent (tanh)

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

* Outputs between **-1 and 1**.
* Centered at 0 (better than sigmoid).

**Pros:** Symmetric around zero.
**Cons:** Still suffers from vanishing gradients.

---

### 3. ReLU (Rectified Linear Unit)

$$
f(x) = \max(0, x)
$$

* Outputs **0 for negative values** and the input itself for positive values.
* Most commonly used in hidden layers.

**Pros:** Computationally efficient, mitigates vanishing gradient.
**Cons:** Can cause "dying ReLU" (neurons stuck at 0).

---

### 4. Leaky ReLU

$$
f(x) = \begin{cases} 
x & \text{if } x > 0 \\
\alpha x & \text{if } x \leq 0
\end{cases}
$$

* Allows a small slope ($\alpha$, e.g., 0.01) for negative values.
* Helps avoid dead neurons.

---

### 5. Softmax

$$
\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}
$$

* Converts logits into **probability distribution**.
* Used in the **output layer** for multi-class classification.

---

## Visual: Activation Functions Comparison

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/activation-functions-comparison.png" alt="Activation Functions Comparison" width="550"/>
</p>

---

## How to Choose an Activation Function?

* **Hidden layers**: ReLU (or Leaky ReLU/ELU for advanced cases).
* **Binary classification output**: Sigmoid.
* **Multi-class classification output**: Softmax.
* **Regression output**:

  * Linear (no activation) for unbounded outputs.
  * Sigmoid/tanh if output range is known.

---

## Code Example: Using Activation Functions in Keras

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(16, activation='relu', input_shape=(10,)),   # hidden layer
    Dense(8, activation='tanh'),
    Dense(1, activation='sigmoid')  # output layer for binary classification
])

model.summary()
```

---

## Advantages and Limitations

**Advantages:**

* Allow deep networks to learn complex representations.
* Different functions suit different tasks.

**Limitations:**

* Wrong choice can slow or prevent training.
* Some functions (sigmoid, tanh) suffer from **vanishing gradient problem**.
* ReLU variants help but aren’t perfect.

---

## References

* [DeepLearning.ai: Activation Functions](https://www.deeplearning.ai/resources/activation-functions/)
* [Goodfellow et al., Deep Learning Book](https://www.deeplearningbook.org/)
* [Wikipedia: Activation Function](https://en.wikipedia.org/wiki/Activation_function)
* [Scikit-learn: Neural Networks](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>