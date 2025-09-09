![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# Neural Networks Basics

---

## Introduction

**Neural networks** are the foundation of deep learning.  
Inspired by the human brain, they consist of **layers of interconnected nodes (neurons)** that process data and learn complex patterns.

Neural networks are widely used in:
- Image recognition
- Natural language processing (NLP)
- Speech recognition
- Recommendation systems

---

## Structure of a Neural Network

A typical **feedforward neural network** has:

1. **Input Layer**  
   - Accepts raw data features (e.g., pixels in an image, words in a sentence).

2. **Hidden Layers**  
   - Perform transformations using weights, biases, and activation functions.
   - More hidden layers = deeper networks.

3. **Output Layer**  
   - Produces predictions (e.g., probability of class, regression value).

---

## Neuron Computation

Each neuron performs:

\[
z = \sum_{i=1}^{n} w_i x_i + b
\]

\[
a = f(z)
\]

Where:
- \( x_i \): input features  
- \( w_i \): weights  
- \( b \): bias  
- \( f \): activation function (e.g., sigmoid, ReLU)  
- \( a \): output of the neuron  

---

## Activation Functions

- **Sigmoid**: Squashes values into (0, 1) → good for probabilities.  
- **ReLU (Rectified Linear Unit)**: Outputs 0 if negative, else value → helps avoid vanishing gradients.  
- **Tanh**: Outputs between (-1, 1).  
- **Softmax**: Converts logits into probabilities across multiple classes.

---

## Forward and Backpropagation

1. **Forward Propagation**  
   - Data flows from input → hidden layers → output.  
   - Predictions are generated.  

2. **Loss Function**  
   - Measures error between predictions and true values.  
   - Examples: Mean Squared Error (regression), Cross-Entropy Loss (classification).  

3. **Backpropagation**  
   - Gradients of the loss are calculated using the **chain rule**.  
   - Parameters (weights & biases) are updated using **Gradient Descent** or its variants (SGD, Adam).  

---

## Code Example: Simple Neural Network in Keras

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Build a simple neural network
model = Sequential([
    Dense(16, activation='relu', input_shape=(10,)),  # hidden layer
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')  # output layer (binary classification)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display model architecture
model.summary()
````

---

## Visual: Neural Network Basics

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/neural-network-basics.png" alt="Neural Network Architecture" width="520"/>
</p>

---

## Advantages and Limitations

**Advantages:**

* Can learn complex non-linear patterns.
* Flexible architecture (layers, activation functions).
* State-of-the-art results in vision, language, and audio tasks.

**Limitations:**

* Requires large datasets.
* Computationally expensive (training deep nets).
* Harder to interpret compared to simpler models.

---

## Best Practices

* Normalize/standardize input features.
* Use ReLU for hidden layers and Softmax/Sigmoid for outputs.
* Apply regularization (dropout, L2) to reduce overfitting.
* Start with small networks before scaling deeper.

---

## Real-World Applications

* **Computer Vision**: Image classification, object detection.
* **NLP**: Sentiment analysis, machine translation, chatbots.
* **Healthcare**: Disease prediction from medical images.
* **Finance**: Fraud detection, stock price forecasting.

---

## References

* [DeepLearning.ai: Neural Networks](https://www.deeplearning.ai/resources/deep-learning-specialization/)
* [Goodfellow et al., Deep Learning Book](https://www.deeplearningbook.org/)
* [Scikit-learn: Neural Networks](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)
* [Wikipedia: Artificial Neural Network](https://en.wikipedia.org/wiki/Artificial_neural_network)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>