![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# Supervised vs Unsupervised Learning

---

## Introduction

Supervised and unsupervised learning are the two main paradigms in machine learning. Understanding their differences is essential for designing and choosing the right algorithms for any data-driven project.

---

## What is Supervised Learning?

In supervised learning, the algorithm is trained on a labeled dataset. This means for every input, the correct output (label) is provided. The goal is to learn a mapping from inputs to outputs so that the model can accurately predict labels for new, unseen data.

**Key characteristics:**
- Uses labeled data (input–output pairs)
- Predicts outcomes for new inputs
- Common for classification and regression tasks

**Examples:**
- Email spam detection (spam or not spam)
- Handwritten digit recognition (0–9)
- House price prediction (numeric value)

---

## What is Unsupervised Learning?

In unsupervised learning, the algorithm works with data that has **no labels**. The goal is to find patterns, groupings, or structures within the data itself.

**Key characteristics:**
- Uses unlabeled data (only inputs, no outputs)
- Finds hidden patterns or clusters
- Common for clustering and dimensionality reduction

**Examples:**
- Customer segmentation for marketing
- Grouping news articles by topic
- Reducing data dimensions for visualization (PCA)

---

## Visual: Supervised vs Unsupervised Learning

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/supervised-vs-unsupervised.png" alt="Supervised vs Unsupervised Learning Diagram" width="500"/>
</p>

**Figure:**  
Supervised learning learns from labeled examples to predict outputs.  
Unsupervised learning organizes or describes data without any known outputs.

---

## Table: Side-by-Side Comparison

| Feature                    | Supervised Learning                           | Unsupervised Learning                       |
|----------------------------|-----------------------------------------------|---------------------------------------------|
| Training Data              | Labeled                                      | Unlabeled                                   |
| Goal                       | Predict output from input                     | Discover patterns or structure              |
| Common Tasks               | Classification, Regression                    | Clustering, Association, Dimensionality Reduction |
| Example Algorithms         | Linear Regression, Decision Trees, SVM        | K-Means, Hierarchical Clustering, PCA       |
| Example Application        | Email spam filter, image classification       | Market segmentation, anomaly detection      |

---

## Minimal Code Example

**Supervised Learning: Predicting House Prices**

```python
from sklearn.linear_model import LinearRegression

X = [[1000], [1500], [2000], [2500]]  # Feature: square footage
y = [200000, 250000, 300000, 350000]  # Label: price

model = LinearRegression()
model.fit(X, y)
print(model.predict([[1800]]))  # Predict price for 1800 sq.ft.
````

**Unsupervised Learning: Customer Segmentation (Clustering)**

```python
from sklearn.cluster import KMeans

# Example: annual income and spending score
X = [[15, 39], [16, 81], [17, 6], [18, 77], [19, 40]]
model = KMeans(n_clusters=2)
model.fit(X)
print(model.labels_)  # Cluster assignments for each customer
```

---

## When to Use Which?

* **Use supervised learning** when you have historical data with known outcomes and want to predict or classify new data.
* **Use unsupervised learning** when you only have raw data and want to explore, group, or understand its structure.

---

## Further Reading and References

* [Stanford: CS229 Lecture Notes – Supervised vs Unsupervised Learning](https://cs229.stanford.edu/notes2022fall/cs229-notes1.pdf)
* [DeepAI: Supervised vs Unsupervised Learning](https://deepai.org/machine-learning-glossary-and-terms/supervised-learning)
* [Scikit-learn User Guide: Supervised vs Unsupervised Learning](https://scikit-learn.org/stable/supervised_learning.html)
* [Pattern Recognition and Machine Learning, Bishop (Springer, 2006)](https://www.springer.com/gp/book/9780387310732)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>