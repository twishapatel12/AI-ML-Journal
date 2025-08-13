![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# Self-Organizing Maps (SOM)

---

## Introduction

**Self-Organizing Maps (SOM)** are a type of **artificial neural network** introduced by Teuvo Kohonen in the 1980s.  
They are used for **unsupervised learning**, especially for **visualizing and clustering high-dimensional data** onto a **low-dimensional (usually 2D) grid**.

Key characteristics:
- Preserves the **topology** of the input space (similar inputs are mapped close to each other on the grid).
- Helps in data exploration and pattern recognition.

---

## How SOM Works

SOM consists of:
- **Input layer**: Accepts feature vectors.
- **Map layer (grid)**: Nodes (neurons) arranged in 1D or 2D grid. Each node has a weight vector of the same dimension as the input.

---

### Training Process

1. **Initialize Weights**  
   - Assign small random values to each node’s weight vector.

2. **Select an Input Vector**  
   - Randomly choose a sample from the dataset.

3. **Find Best Matching Unit (BMU)**  
   - The BMU is the node whose weight vector is **closest** to the input vector (using Euclidean distance).

4. **Update Weights**  
   - Adjust the BMU's weight vector and its neighbors towards the input vector:

$$
w(t+1) = w(t) + \alpha(t) \cdot h_{\text{BMU}}(t) \cdot (x - w(t))
$$

     Where:
     - $\alpha(t)$ = learning rate (decreases over time)
     - $h_{\text{BMU}}(t)$ = neighborhood function (larger at start, shrinks over time)

5. **Repeat**  
   - Continue until convergence (weights no longer change significantly).

---

## Visual: SOM Concept

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/som-concept.png" alt="Self-Organizing Map Concept Diagram" width="500"/>
</p>

*Shows high-dimensional points being mapped to a 2D grid while preserving neighborhood relationships.*

---

## Code Example: SOM in Python

Scikit-learn does not have SOM built-in, but we can use `MiniSom`:

```python
!pip install minisom

import numpy as np
from minisom import MiniSom
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

# Load and scale data
data = load_iris()
X = data.data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Initialize SOM
som = MiniSom(x=10, y=10, input_len=X_scaled.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(X_scaled)
som.train_random(X_scaled, num_iteration=100)

# Plot SOM distance map
plt.figure(figsize=(7, 7))
plt.pcolor(som.distance_map().T, cmap='coolwarm')  # U-Matrix
plt.colorbar()
plt.title("SOM Distance Map (U-Matrix)")
plt.show()
````

---

## Applications of SOM

* **Data visualization**: Map high-dimensional data into 2D.
* **Clustering**: Group similar samples together.
* **Market segmentation**: Identify customer groups.
* **Image compression**: Reduce color palettes.
* **Bioinformatics**: Gene expression pattern analysis.

---

## Advantages and Limitations

**Advantages:**

* Preserves topology of input data.
* Great for visualization and pattern discovery.
* Works without labeled data.

**Limitations:**

* Requires tuning of parameters (map size, learning rate, neighborhood size).
* Training can be slow for large datasets.
* Does not always guarantee globally optimal mapping.

---

## Best Practices

* Normalize or standardize input data.
* Choose map size proportional to dataset size.
* Gradually decrease learning rate and neighborhood size during training.
* Use U-Matrix visualization to interpret results.

---

## References

* [MiniSom Documentation](https://github.com/JustGlowing/minisom)
* [Wikipedia: Self-Organizing Map](https://en.wikipedia.org/wiki/Self-organizing_map)
* [Teuvo Kohonen’s Original Paper](https://ieeexplore.ieee.org/document/58325)
* [Practical SOM Applications in Data Mining](https://link.springer.com/chapter/10.1007/3-540-36530-0_8)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>