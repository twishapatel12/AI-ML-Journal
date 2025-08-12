![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# Random Forests

---

## Introduction

**Random Forests** are powerful and widely used machine learning algorithms for both classification and regression.  
A random forest combines the predictions of many decision trees, each trained on slightly different data, to produce more accurate and stable results than a single tree.

Random forests are a classic example of an **ensemble method**—using the “wisdom of the crowd” to improve prediction.

---

## What Is a Random Forest?

A **random forest** is a collection (ensemble) of decision trees.  
- Each tree is built from a random subset of the data (using **bagging**).
- At each split, the tree considers a random subset of features.
- Final prediction is made by combining the output of all trees (majority vote for classification, average for regression).

---

## Why Use Random Forests?

- **Reduce overfitting:** Individual decision trees can be unstable and overfit to training data. Random forests average their predictions, making them less likely to overfit.
- **Handle many features and data types:** Good for high-dimensional and mixed data.
- **Robust:** Work well even if some features are noisy or irrelevant.
- **Minimal tuning needed:** Usually work well “out of the box”.

---

## How Does a Random Forest Work?

1. **Bagging (Bootstrap Aggregation):**
   - Randomly sample the data (with replacement) to create many different training sets.
   - Build a decision tree on each sample.

2. **Random Feature Selection:**
   - At each split in the tree, pick a random subset of features to consider for the best split.

3. **Prediction:**
   - **Classification:** Each tree votes for a class, and the majority wins.
   - **Regression:** Take the average of all tree outputs.

---

## Visual: Random Forest Structure

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/random-forest-diagram.png" alt="Random Forest Structure" width="480"/>
</p>

*Shows multiple trees, each built from different random samples, with their outputs combined.*

---

## Code Example: Random Forest in Python

```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Example: Predict buying (1) or not (0) from Age and Income
X = np.array([[25, 40000], [40, 50000], [35, 80000], [20, 20000], [45, 120000]])
y = np.array([0, 1, 1, 0, 1])

model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
model.fit(X, y)
print("Prediction for [30, 70000]:", model.predict([[30, 70000]]))
print("Feature importances:", model.feature_importances_)
````

---

## Feature Importance

Random forests can measure how important each feature is for making decisions, by looking at how much each feature improves the splits across all trees.

* Useful for understanding which features drive predictions.

---

## Advantages and Limitations

**Advantages:**

* High accuracy and works for both classification and regression.
* Handles missing data and mixed data types well.
* Less likely to overfit than a single tree.

**Limitations:**

* Slower to predict than a single tree (many trees to combine).
* Less interpretable than a single tree.
* Can be memory-intensive with many trees.

---

## Hyperparameters to Tune

* **n\_estimators:** Number of trees in the forest.
* **max\_depth:** Maximum depth of each tree.
* **max\_features:** Number of features to consider at each split.
* **min\_samples\_split/leaf:** Minimum number of samples to split a node or be a leaf.

Try different settings using cross-validation to optimize performance.

---

## Real-World Applications

* Credit scoring and risk assessment.
* Medical diagnosis and disease prediction.
* Fraud detection.
* Customer churn prediction.
* Image and speech recognition (as part of larger systems).

---

## Further Reading & References

* [Scikit-learn: Random Forest Documentation](https://scikit-learn.org/stable/modules/ensemble.html#random-forests)
* [Wikipedia: Random Forest](https://en.wikipedia.org/wiki/Random_forest)
* [Analytics Vidhya: Random Forest Guide](https://www.analyticsvidhya.com/blog/2021/06/random-forest-classifier/)
* [Khan Academy: Ensemble Methods](https://www.khanacademy.org/math/statistics-probability)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>