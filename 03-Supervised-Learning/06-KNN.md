![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# K-Nearest Neighbors (KNN)

---

## Introduction

**K-Nearest Neighbors (KNN)** is one of the simplest and most intuitive machine learning algorithms.  
It’s used for both classification (predicting categories) and regression (predicting numbers).

KNN makes predictions based on the “closeness” of data points—it looks at the labels or values of the nearest neighbors to a given input and decides accordingly.

---

## How Does KNN Work?

1. **Choose K:** Pick how many neighbors to consider (e.g., K=3).
2. **Measure distance:** For a new input, calculate its distance to every point in the training data (using Euclidean, Manhattan, or other distance metrics).
3. **Find the K nearest neighbors:** Select the K closest points.
4. **Predict:**
   - **Classification:** Use majority vote (the most common class among neighbors).
   - **Regression:** Average the values of neighbors.

---

## Visual: KNN Classification

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/knn-classification.png" alt="KNN Classification Example" width="420"/>
</p>

*Shows a test point and lines to its K nearest neighbors, with voting for the predicted class.*

---

## Code Example: KNN in Python

```python
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Simple example: features and labels
X = np.array([[1, 2], [2, 3], [3, 3], [6, 7], [7, 8], [8, 8]])
y = np.array([0, 0, 0, 1, 1, 1])  # 2 classes: 0 and 1

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)
prediction = model.predict([[5, 5]])
print("Predicted class for [5,5]:", prediction)
````

---

## K in KNN: How to Choose?

* **Small K (e.g., 1):** Very sensitive to noise/outliers (low bias, high variance).
* **Large K:** More robust, but can blur the boundaries between classes (higher bias, lower variance).
* **Odd K:** For classification, avoids ties.

Use **cross-validation** to find the best K for your problem.

---

## Distance Metrics

* **Euclidean distance** (standard for KNN):

$$
d(x, y) = \sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2 + \ldots}
$$

* **Other options:** Manhattan, Minkowski, cosine, etc.

---

## Advantages and Limitations

**Advantages:**

* Very easy to understand and implement.
* No training phase—just store the data.
* Naturally handles multi-class problems.

**Limitations:**

* **Slow prediction:** Needs to compute distance to every point in the dataset.
* Sensitive to irrelevant features and scale of features (normalize your data).
* Performance drops with high-dimensional data (curse of dimensionality).

---

## Use Cases

* Handwritten digit recognition (e.g., MNIST dataset)
* Recommender systems (finding users/items similar to a target)
* Anomaly detection (find points far from any cluster)

---

## Real-World Example: KNN for Iris Flower Classification

```python
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X, y = iris.data, iris.target

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
sample = [[5.7, 3.0, 4.2, 1.2]]  # Unknown iris
prediction = knn.predict(sample)
print("Predicted iris class:", iris.target_names[prediction][0])
```

---

## Further Reading & References

* [Scikit-learn: KNN Documentation](https://scikit-learn.org/stable/modules/neighbors.html#classification)
* [Wikipedia: K-nearest neighbors algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
* [Khan Academy: Introduction to KNN](https://www.khanacademy.org/math/statistics-probability)
* [Analytics Vidhya: KNN Algorithm Guide](https://www.analyticsvidhya.com/blog/2018/03/introduction-k-neighbours-algorithm-clustering/)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>