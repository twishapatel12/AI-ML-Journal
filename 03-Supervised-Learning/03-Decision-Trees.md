![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# Decision Trees

---

## Introduction

**Decision trees** are a popular and powerful machine learning algorithm for both classification and regression tasks.  
They work by breaking down a dataset into smaller and smaller subsets, while at the same time an associated decision tree is incrementally developed.

The result is a tree-like model of decisions—just like a flowchart—that makes predictions based on asking a sequence of “yes/no” questions about the input features.

---

## When to Use Decision Trees

- When you want a model that’s easy to interpret and visualize.
- When your features are a mix of numerical and categorical data.
- For both **classification** (e.g., is this customer likely to churn?) and **regression** (e.g., what will the house price be?).

---

## How Does a Decision Tree Work?

- At each **node**, the tree asks a question about a feature (e.g., “Is age > 30?”).
- Based on the answer (yes/no), it splits the data into two groups.
- Each split aims to make the resulting groups as “pure” as possible (mostly one class).
- This process repeats recursively until a **stopping condition** is met (e.g., max depth, minimum samples per leaf, or all examples are of the same class).

---

## Visual: Decision Tree Structure

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/decision-tree-example.png" alt="Decision Tree Example" width="480"/>
</p>

*Shows a simple tree with nodes splitting on questions (e.g., “Age > 30?”), with final predictions at the leaves.*

---

## Key Concepts

- **Root Node:** The top node (first decision).
- **Internal Nodes:** Nodes that split on features.
- **Leaf/Terminal Node:** End node, represents a prediction (class or value).
- **Branch:** A path from a question to an answer (yes/no).
- **Depth:** Number of levels from root to leaf.

---

## How Does the Tree Decide Where to Split?

The tree chooses splits that **maximize the separation** of classes or minimize error for regression.

- **Classification:** Uses measures like **Gini impurity** or **entropy** to pick the best split.
- **Regression:** Uses **variance reduction** or **mean squared error (MSE)**.

**Example: Gini impurity for binary split**

$$
Gini = 1 - \sum_{i=1}^{C} p_i^2
$$

where $p_i$ is the proportion of examples in class $i$.

---

## Code Example: Decision Tree in Python

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

X = [[25, 1], [40, 0], [35, 1], [20, 0]]  # [Age, Owns House (1=yes, 0=no)]
y = [0, 1, 1, 0]  # 1 = Will buy product, 0 = Will not

model = DecisionTreeClassifier(max_depth=2)
model.fit(X, y)

plt.figure(figsize=(8,4))
tree.plot_tree(model, feature_names=['Age', 'Owns House'], class_names=['No', 'Yes'], filled=True)
plt.show()

print("Prediction for [30, 1]:", model.predict([[30, 1]]))
````

---

## Advantages and Limitations

**Advantages:**

* Easy to interpret and visualize.
* Handles both numerical and categorical data.
* Requires little data preprocessing (no scaling needed).
* Can capture non-linear relationships.

**Limitations:**

* Prone to **overfitting** (especially deep trees).
* Can be unstable: small changes in data can change the tree.
* Not as accurate as ensemble methods (like random forest) on complex tasks.

---

## Regularization Techniques

To prevent overfitting:

* Limit the **maximum depth** of the tree.
* Set a **minimum number of samples** required to split a node or to be a leaf.
* Use **pruning** to remove unnecessary branches.

---

## Real-World Applications

* Credit scoring: decide whether to approve a loan.
* Medical diagnosis: predict disease presence.
* Customer segmentation and marketing.
* Predicting risk or outcomes in business, finance, and engineering.

---

## Further Reading & References

* [Scikit-learn: Decision Tree Documentation](https://scikit-learn.org/stable/modules/tree.html)
* [Wikipedia: Decision Tree Learning](https://en.wikipedia.org/wiki/Decision_tree_learning)
* [Analytics Vidhya: Decision Tree Algorithm Guide](https://www.analyticsvidhya.com/blog/2021/08/decision-tree-classifier/)
* [Khan Academy: Decision Trees](https://www.khanacademy.org/math/statistics-probability/probability-library)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>