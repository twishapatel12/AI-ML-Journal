![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# Anomaly Detection

---

## Introduction

**Anomaly detection** (also called **outlier detection**) is the process of identifying data points, events, or observations that deviate significantly from the norm.

These anomalies can indicate:
- Fraudulent transactions
- Machine failures
- Cybersecurity breaches
- Rare diseases
- Data entry errors

In **unsupervised learning**, anomaly detection is performed **without labeled examples**—the system learns the “normal” patterns from the data and flags deviations.

---

## Types of Anomalies

1. **Point Anomalies**  
   - Single observations far from the rest.  
   - Example: Credit card purchase 100× higher than usual.

2. **Contextual Anomalies**  
   - Anomalous only in a specific context.  
   - Example: High temperature in winter is abnormal, but normal in summer.

3. **Collective Anomalies**  
   - A sequence of observations that together are anomalous.  
   - Example: Multiple failed login attempts in a short time.

---

## Common Unsupervised Anomaly Detection Methods

### 1. Statistical Methods
- Assume data follows a known distribution (e.g., Gaussian).
- Flag points with low probability density.
```python
import numpy as np

data = np.random.normal(0, 1, 1000)
mean, std = np.mean(data), np.std(data)
threshold = 3
anomalies = [x for x in data if abs(x - mean) > threshold * std]
````

---

### 2. Distance-Based Methods (KNN)

* Points far from neighbors are anomalies.
* Uses Euclidean or other distance metrics.

---

### 3. Clustering-Based Methods (e.g., DBSCAN)

* Points not belonging to any cluster are anomalies.
* Naturally flags noise points.

---

### 4. Isolation Forest

* Randomly partitions data.
* Anomalies are easier to isolate and have shorter average path lengths in the trees.

**Code Example:**

```python
from sklearn.ensemble import IsolationForest
import numpy as np

# Synthetic data
rng = np.random.RandomState(42)
X = 0.3 * rng.randn(100, 2)
X_train = np.r_[X + 2, X - 2]
X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))
X_all = np.r_[X_train, X_outliers]

# Fit Isolation Forest
clf = IsolationForest(contamination=0.1, random_state=42)
clf.fit(X_train)
y_pred = clf.predict(X_all)  # -1 = anomaly, 1 = normal
```

---

### 5. One-Class SVM

* Learns a boundary around normal data.
* Flags points outside as anomalies.

```python
from sklearn.svm import OneClassSVM

clf = OneClassSVM(kernel='rbf', nu=0.05, gamma='auto')
clf.fit(X_train)
predictions = clf.predict(X_all)  # -1 = anomaly, 1 = normal
```

---

### 6. Autoencoder-Based Anomaly Detection

* Train an autoencoder on normal data.
* High reconstruction error indicates an anomaly.

---

## Evaluation Metrics

For labeled data:

* **Precision**: Fraction of detected anomalies that are true anomalies.
* **Recall**: Fraction of true anomalies detected.
* **F1-Score**: Harmonic mean of precision and recall.
* **AUC-ROC**: Measures ability to rank anomalies above normal points.

---

## Visual: Anomaly Detection Concept

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/anomaly-detection-concept.png" alt="Anomaly Detection Concept Diagram" width="500"/>
</p>

---

## Advantages and Limitations

**Advantages:**

* Works in domains where anomalies are rare and labels are unavailable.
* Flexible choice of methods (statistical, distance-based, model-based).

**Limitations:**

* Choice of method and parameters greatly affects results.
* High-dimensional data can make distance metrics less meaningful (curse of dimensionality).
* In unsupervised settings, hard to validate without labels.

---

## Best Practices

* Scale/normalize features before applying distance-based methods.
* Use domain knowledge to select features.
* Combine multiple detection methods for robustness.
* Monitor models over time—normal patterns can drift.

---

## Real-World Applications

* **Fraud detection** in banking and e-commerce.
* **Intrusion detection** in cybersecurity.
* **Predictive maintenance** in manufacturing.
* **Healthcare anomaly detection** (e.g., rare medical conditions).
* **Data cleaning** (removing invalid entries).

---

## References

* [Scikit-learn: Outlier Detection](https://scikit-learn.org/stable/modules/outlier_detection.html)
* [Isolation Forest Paper (Liu et al., 2008)](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)
* [One-Class SVM Paper (Schölkopf et al., 2001)](https://dl.acm.org/doi/10.5555/1119748.1119749)
* [Kaggle: Anomaly Detection Guides](https://www.kaggle.com/learn/anomaly-detection)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>