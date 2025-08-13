![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# Autoencoders

---

## Introduction

**Autoencoders** are a type of **artificial neural network** used for **unsupervised learning**, mainly for **dimensionality reduction**, **feature learning**, and **data reconstruction**.

They work by:
1. **Encoding** input data into a smaller (compressed) representation.
2. **Decoding** this representation back to the original data format.

If trained well, the compressed representation captures the most important information in the data.

---

## Architecture of an Autoencoder

An autoencoder has three main parts:

1. **Encoder**  
   - Maps input $x$ to a lower-dimensional representation $z$ (latent space).
   - Function: $z = f_{\text{encoder}}(x)$

2. **Latent Space (Code)**  
   - The compressed feature vector.
   - Size is smaller than the input (for compression tasks).

3. **Decoder**  
   - Reconstructs the input from $z$.
   - Function: $\hat{x} = f_{\text{decoder}}(z)$

**Objective:** Minimize reconstruction error between $x$ and $\hat{x}$.

---

## Mathematical Formulation

The loss function is typically **Mean Squared Error (MSE)**:

$$
L(x, \hat{x}) = \frac{1}{n} \sum_{i=1}^{n} (x_i - \hat{x}_i)^2
$$

Where:
- $x$ = original input
- $\hat{x}$ = reconstructed input

Other losses (e.g., binary cross-entropy) can be used depending on data type.

---

## Visual: Autoencoder Architecture

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/autoencoder-architecture.png" alt="Autoencoder Architecture" width="500"/>
</p>

---

## Code Example: Autoencoder in Keras

```python
import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist

# Load data
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), -1))
x_test = x_test.reshape((len(x_test), -1))

# Encoder
input_dim = x_train.shape[1]
encoding_dim = 32
input_img = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_img)

# Decoder
decoded = Dense(input_dim, activation='sigmoid')(encoded)

# Autoencoder model
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train
autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# Compressed representation
encoder = Model(input_img, encoded)
encoded_imgs = encoder.predict(x_test)
````

---

## Types of Autoencoders

1. **Vanilla Autoencoder**

   * Basic encoder-decoder structure with fully connected layers.

2. **Convolutional Autoencoder (CAE)**

   * Uses convolutional layers; ideal for images.

3. **Denoising Autoencoder (DAE)**

   * Learns to reconstruct original input from noisy input.

4. **Sparse Autoencoder**

   * Uses sparsity constraints in latent space for feature selection.

5. **Variational Autoencoder (VAE)**

   * Learns a **probabilistic latent space** for generative modeling.

---

## Applications

* **Dimensionality Reduction**: Alternative to PCA for non-linear data.
* **Denoising**: Remove noise from images or signals.
* **Anomaly Detection**: Large reconstruction error indicates anomalies.
* **Data Generation**: Using VAEs to generate synthetic samples.
* **Feature Extraction**: Learned latent representation used as input to other models.

---

## Advantages and Limitations

**Advantages:**

* Learns non-linear transformations.
* Flexible architecture (can be tailored to different data types).
* Can be stacked or combined with other neural network architectures.

**Limitations:**

* Requires large amounts of data for training.
* Computationally expensive compared to PCA.
* If not regularized, may simply memorize inputs without learning useful features.

---

## Best Practices

* Normalize or standardize input data before training.
* Use appropriate activation functions for encoder/decoder.
* Choose latent space size carefully—too small loses info, too large doesn’t compress well.
* Regularize with dropout, sparsity, or noise to improve generalization.

---

## References

* [Keras: Autoencoder Example](https://blog.keras.io/building-autoencoders-in-keras.html)
* [Goodfellow et al., Deep Learning Book - Chapter 14](https://www.deeplearningbook.org/)
* [Wikipedia: Autoencoder](https://en.wikipedia.org/wiki/Autoencoder)
* [Variational Autoencoders (Kingma & Welling, 2014)](https://arxiv.org/abs/1312.6114)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>
