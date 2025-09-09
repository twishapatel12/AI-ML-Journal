![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# Backpropagation in Neural Networks

---

## Introduction

**Backpropagation (Backward Propagation of Errors)** is the core algorithm that allows neural networks to learn.  

It is a method of **training neural networks** by propagating the error from the output layer back through the network and updating the weights using **gradient descent**.  

Without backpropagation, deep learning would not be possible.

---

## Key Idea

1. Perform a **forward pass**: compute outputs and loss.
2. Compute the **loss function** (difference between predicted and true labels).
3. Use the **chain rule of calculus** to compute gradients of the loss with respect to each weight.
4. Perform a **backward pass**: propagate these gradients from output → hidden layers → input.
5. Update weights using an optimization algorithm (e.g., Gradient Descent, Adam).

---

## Mathematical Formulation

### 1. Forward Pass

For neuron \( j \) in layer \( l \):

\[
z_j^{(l)} = \sum_i w_{ij}^{(l)} a_i^{(l-1)} + b_j^{(l)}
\]

\[
a_j^{(l)} = f(z_j^{(l)})
\]

Where:
- \( w_{ij}^{(l)} \) = weight  
- \( b_j^{(l)} \) = bias  
- \( f \) = activation function  

---

### 2. Loss Function

For example, Mean Squared Error (MSE):

\[
L = \frac{1}{2} \sum (y - \hat{y})^2
\]

---

### 3. Backward Pass (Gradients)

Using the chain rule:

\[
\frac{\partial L}{\partial w_{ij}^{(l)}} = \delta_j^{(l)} a_i^{(l-1)}
\]

Where the error term is:

\[
\delta_j^{(l)} = \frac{\partial L}{\partial z_j^{(l)}}
\]

- For the output layer:
\[
\delta^{(L)} = (a^{(L)} - y) \cdot f'(z^{(L)})
\]

- For hidden layers:
\[
\delta^{(l)} = \left( \sum_k w_{jk}^{(l+1)} \delta_k^{(l+1)} \right) \cdot f'(z^{(l)})
\]

---

### 4. Weight Update

\[
w_{ij}^{(l)} \leftarrow w_{ij}^{(l)} - \eta \cdot \frac{\partial L}{\partial w_{ij}^{(l)}}
\]

\[
b_j^{(l)} \leftarrow b_j^{(l)} - \eta \cdot \delta_j^{(l)}
\]

Where \( \eta \) is the **learning rate**.

---

## Visual: Backpropagation Process

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/backpropagation-concept.png" alt="Backpropagation Concept Diagram" width="520"/>
</p>

---

## Code Example: Backpropagation in PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Example dataset
X = torch.randn(5, 3)   # 5 samples, 3 features
y = torch.randn(5, 1)   # 5 target values

# Define a simple neural network
model = nn.Sequential(
    nn.Linear(3, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)

criterion = nn.MSELoss()         # Loss function
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Optimizer

# Training loop (1 step demonstration)
optimizer.zero_grad()       # Reset gradients
outputs = model(X)          # Forward pass
loss = criterion(outputs, y) # Compute loss
loss.backward()             # Backward pass (compute gradients)
optimizer.step()            # Update weights

print("Loss:", loss.item())
````

---

## Advantages and Limitations

**Advantages:**

* Efficient way to train deep networks.
* Works with any differentiable activation function and loss function.
* Scales to very large networks.

**Limitations:**

* Suffers from **vanishing/exploding gradients** in very deep networks.
* Sensitive to learning rate and initialization.
* Computationally expensive for huge datasets.

---

## Best Practices

* Use ReLU/variants to reduce vanishing gradients.
* Normalize/standardize inputs before training.
* Use optimizers like **Adam** for faster convergence.
* Apply **batch normalization** and **dropout** to stabilize training.

---

## Real-World Applications

* Computer vision (image classification, object detection).
* NLP (machine translation, text summarization).
* Speech recognition.
* Reinforcement learning (deep Q-networks).

---

## References

* [DeepLearning.ai: Backpropagation Explained](https://www.deeplearning.ai/resources/backpropagation/)
* [Goodfellow et al., Deep Learning Book - Chapter 6](https://www.deeplearningbook.org/)
* [Wikipedia: Backpropagation](https://en.wikipedia.org/wiki/Backpropagation)
* [PyTorch Autograd Documentation](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>

---

## Image Generation Prompt

**backpropagation-concept.png**
*Prompt:*

> Draw a diagram of a simple 3-layer neural network. Show forward propagation with arrows (input → hidden → output) in one color, and backpropagation of errors with arrows going backward in another color. Label weight updates and gradients.

```

---

Would you like me to continue with **"04-Optimizers.md"** next in the Deep Learning section? It connects directly with backpropagation since optimizers update weights.
```
