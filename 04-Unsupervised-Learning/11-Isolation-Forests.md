![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# Isolation Forests

---

## Introduction

**Isolation Forest** is an **unsupervised anomaly detection** algorithm that identifies outliers by isolating observations.  

The core idea:
- Anomalies are **few** and **different**.
- They are easier to isolate from the rest of the data compared to normal points.
- The algorithm randomly selects a feature and splits the data.
- Points that require fewer splits to be isolated are more likely to be anomalies.

Isolation Forest is widely used for:
- Fraud detection
- Intrusion detection
- Fault detection in manufacturing

---

## How It Works (Step-by-Step)

1. **Random Partitioning**  
   - Randomly select a feature.
   - Randomly select a split value between the min and max of that feature.

2. **Recursive Splitting**  
   - Continue splitting subsets recursively to create a binary tree.
   - Isolation happens when a point is separated from others.

3. **Path Length**  
   - The number of splits required to isolate a point is the **path length**.
   - Outliers have **shorter average path lengths**.

4. **Anomaly Score**  
   - Normalized path length → anomaly score in [0, 1].
   - Higher score = more likely to be an anomaly.

---

## Mathematical Insight

Expected path length for $n$ samples is:

$$
c(n) = 2 H(n-1) - \frac{2(n-1)}{n}
$$

Where:
- $H(i)$ = harmonic number.

Anomaly score for a point $x$:

$
s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}
$

Where:
- $E(h(x))$ = average path length for $x$ over all trees.

---

## Code Example: Isolation Forest in Python

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Create synthetic data
rng = np.random.RandomState(42)
X_normal = 0.3 * rng.randn(100, 2)
X_normal = np.r_[X_normal + 2, X_normal - 2]
X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))

X = np.r_[X_normal, X_outliers]

# Fit Isolation Forest
clf = IsolationForest(contamination=0.1, random_state=42)
clf.fit(X)
y_pred = clf.predict(X)  # 1 = normal, -1 = anomaly
scores = clf.decision_function(X)

# Plot results
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='coolwarm', s=30)
plt.title("Isolation Forest Anomaly Detection")
plt.show()
````

---

## Parameters

* **n\_estimators**: Number of trees in the forest (default=100).
* **max\_samples**: Number of samples to draw to train each tree.
* **contamination**: Proportion of anomalies in the dataset.
* **max\_features**: Number of features to use when splitting.

---

## Advantages and Limitations

**Advantages:**

* Efficient for large datasets (linear time complexity).
* Works well in high-dimensional spaces.
* No need for data labeling.

**Limitations:**

* Performance depends on contamination parameter.
* May not perform well when anomalies are not well-separated from normal data.

---

## Best Practices

* Standardize or normalize data before applying.
* Tune **contamination** based on domain knowledge.
* Use decision function scores for threshold-based detection.
* Compare with other methods (e.g., One-Class SVM, DBSCAN) for robustness.

---

## Real-World Applications

* **Finance**: Detecting fraudulent transactions.
* **Cybersecurity**: Intrusion and malware detection.
* **IoT**: Identifying faulty sensor readings.
* **Manufacturing**: Predicting machine breakdowns.

---

## Visual: Isolation Forest Concept

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/isolation-forest-concept.png" alt="Isolation Forest Concept Diagram" width="500"/>
</p>

*Shows random splits isolating anomalies with fewer partitions.*

---

## References

* [Scikit-learn: Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
* [Isolation Forest Original Paper (Liu et al., 2008)](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)
* [Kaggle: Isolation Forest Tutorial](https://www.kaggle.com/code/solomonk/unsupervised-learning-isolation-forest)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>