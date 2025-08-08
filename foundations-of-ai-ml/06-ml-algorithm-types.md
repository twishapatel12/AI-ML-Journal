![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# ML Algorithm Types

---

## Introduction

Machine learning algorithms are the building blocks of all ML applications. Each type of algorithm solves different kinds of problems and works best with certain kinds of data. In this guide, you'll learn about the main types of ML algorithms in simple terms, with visuals and code examples.

---

## Categories of Machine Learning Algorithms

### 1. Supervised Learning Algorithms

These algorithms learn from **labeled data**. The goal is to predict an output (label) for new inputs.

**Main tasks:**  
- Classification (predicting categories)
- Regression (predicting numbers)

**Popular Supervised Algorithms:**
- **Linear Regression** (for predicting continuous values)
- **Logistic Regression** (for binary classification)
- **Decision Trees**
- **Random Forests**
- **Support Vector Machines (SVM)**
- **K-Nearest Neighbors (KNN)**

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/supervised-algorithms-diagram.png" alt="Supervised Algorithms Diagram" width="400"/>
</p>

**Example: Predicting if an email is spam or not (classification) or predicting house prices (regression).**

---

### 2. Unsupervised Learning Algorithms

These algorithms find patterns in **unlabeled data**. They help organize and explore data without predefined categories.

**Main tasks:**  
- Clustering (grouping similar data)
- Dimensionality Reduction (simplifying data)

**Popular Unsupervised Algorithms:**
- **K-Means Clustering**
- **Hierarchical Clustering**
- **Principal Component Analysis (PCA)**
- **t-SNE**

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/unsupervised-algorithms-diagram.png" alt="Unsupervised Algorithms Diagram" width="400"/>
</p>

**Example: Grouping customers by shopping habits or reducing features in a dataset for visualization.**

---

### 3. Semi-Supervised and Self-Supervised Learning

These approaches use a mix of labeled and unlabeled data, or use the data itself as its own label.

- **Semi-supervised:** Few labeled examples + many unlabeled examples.
- **Self-supervised:** Creates its own labels from the structure of the data (popular in deep learning and language models).

---

### 4. Reinforcement Learning Algorithms

Algorithms that **learn by trial and error**, receiving feedback as rewards or penalties.

**Used in:**  
- Robotics  
- Game-playing (like AlphaGo)
- Self-driving cars

**Popular Reinforcement Algorithms:**
- **Q-Learning**
- **Deep Q-Networks (DQN)**
- **Policy Gradient Methods**

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/reinforcement-algorithms-diagram.png" alt="Reinforcement Learning Diagram" width="400"/>
</p>

**Example: Training an agent to win at chess or navigate a maze.**

---

## Quick Table: ML Algorithm Types and When to Use

| Algorithm Type      | Typical Use Case        | Example Algorithms             |
|---------------------|------------------------|-------------------------------|
| Supervised          | Labeled data           | Linear Regression, SVM, KNN   |
| Unsupervised        | No labels, find groups | K-Means, PCA, t-SNE           |
| Reinforcement       | Learn by feedback      | Q-Learning, DQN               |
| Semi/Self-Supervised| Mix or self-label      | Ladder Networks, BERT pretraining |

---

## Visual: ML Algorithm Types Overview

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/ml-algorithm-types-overview.png" alt="ML Algorithm Types Overview Diagram" width="500"/>
</p>

---

## Minimal Code Examples

**Supervised (Classification with KNN):**

```python
from sklearn.neighbors import KNeighborsClassifier

X = [[1,2], [2,3], [3,4], [6,7]]  # Example features
y = [0, 0, 1, 1]  # Labels

model = KNeighborsClassifier(n_neighbors=1)
model.fit(X, y)
print(model.predict([[2,2]]))  # Predicts class for new input
````

**Unsupervised (Clustering with K-Means):**

```python
from sklearn.cluster import KMeans

X = [[1,2], [1,4], [10,8], [10,10]]
model = KMeans(n_clusters=2)
model.fit(X)
print(model.labels_)  # Shows group for each point
```

**Reinforcement Learning (Conceptual Q-Learning update):**

```python
# Pseudocode only, not runnable as-is
Q[state, action] = Q[state, action] + alpha * (reward + gamma * max(Q[next_state, :]) - Q[state, action])
```

---

## Further Reading and References

* [Scikit-learn User Guide: Choosing the Right Estimator](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)
* [Stanford CS229: Machine Learning Algorithms](https://cs229.stanford.edu/)
* [DeepAI: Types of Machine Learning Algorithms](https://deepai.org/machine-learning-glossary-and-terms/machine-learning-algorithm)
* [Google AI Blog: Advances in Self-Supervised Learning](https://ai.googleblog.com/2020/07/exploring-self-supervised-learning.html)
* [OpenAI: Introduction to Reinforcement Learning](https://openai.com/research/publications/introduction-to-reinforcement-learning)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>