![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# Cross-Validation in Machine Learning

---

## Introduction

**Cross-validation (CV)** is a statistical method used to **evaluate the performance of a machine learning model**.  
It helps estimate how well the model will perform on **unseen data** by splitting the dataset into different training and testing subsets.

Instead of using a single train-test split, cross-validation runs multiple splits to get a more reliable measure of model performance.

---

## Why Do We Need Cross-Validation?

- **Single train-test split can be misleading**  
  If we’re unlucky, our single test set might not represent the overall data well.
- **Better estimate of generalization**  
  CV reduces variance in performance estimation.
- **Helps with model selection and tuning**  
  Use CV scores to pick the best model or hyperparameters.

---

## Key Concepts

### 1. Training Set vs. Test Set
- **Training set:** Used to fit the model.
- **Test set:** Used to measure performance on unseen data.

### 2. Validation Set
- Sometimes we create a **third set** (validation set) to fine-tune hyperparameters, separate from the final test set.

### 3. Overfitting Check
- CV helps detect overfitting—when the model performs well on training data but poorly on validation/test data.

---

## Types of Cross-Validation

### 1. k-Fold Cross-Validation

- Split data into **k equal-sized folds**.
- Use \(k-1\) folds for training and 1 fold for testing.
- Repeat k times, each time using a different fold as the test set.
- Average the scores over all k runs.

**Example:**  
If \( k = 5 \), the data is split into 5 folds, and the model is trained/tested 5 times.

**Code Example:**

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=200)

scores = cross_val_score(model, X, y, cv=5)
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())
````

---

### 2. Stratified k-Fold Cross-Validation

* Like k-Fold, but preserves the **class proportions** in each fold.
* Useful for classification tasks with imbalanced classes.

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

skf = StratifiedKFold(n_splits=5)
scores = cross_val_score(model, X, y, cv=skf)
print("Stratified CV Mean accuracy:", scores.mean())
```

---

### 3. Leave-One-Out Cross-Validation (LOOCV)

* Special case where $k = n$ (number of samples).
* Each observation is used once as a test set, and the rest as training data.
* Gives the most unbiased estimate, but is computationally expensive.

---

### 4. Shuffle Split

* Randomly splits the dataset into training and test sets multiple times.
* Useful when you want more control over the train/test sizes.

---

## Visual: k-Fold Cross-Validation Process

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/kfold-cross-validation.png" alt="k-Fold Cross-Validation Diagram" width="500"/>
</p>

*Each fold acts as the test set once, and training set k-1 times.*

---

## Cross-Validation for Hyperparameter Tuning

Often used with **GridSearchCV** or **RandomizedSearchCV** in Scikit-learn:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(max_iter=200), param_grid, cv=5)
grid.fit(X, y)
print("Best parameters:", grid.best_params_)
print("Best CV score:", grid.best_score_)
```

---

## Advantages and Limitations

**Advantages:**

* Gives a better estimate of how the model will generalize.
* Makes efficient use of the dataset (all data points get to be in the test set once).
* Helps with model comparison and selection.

**Limitations:**

* More computationally expensive than a single train/test split.
* Can be slow for large datasets or complex models.

---

## Best Practices

* Use **Stratified k-Fold** for classification with imbalanced data.
* Don’t tune hyperparameters on the final test set—use a validation approach like CV.
* Choose k based on dataset size (5 or 10 folds are common defaults).
* For very small datasets, LOOCV can be helpful.

---

## Summary

* Cross-validation is essential for **reliable model evaluation**.
* It reduces variance and bias in performance estimation.
* It’s widely used for **model selection** and **hyperparameter tuning**.

---

## Further Reading & References

* [Scikit-learn: Cross-validation Documentation](https://scikit-learn.org/stable/modules/cross_validation.html)
* [Khan Academy: Cross-validation Concepts](https://www.khanacademy.org/math/statistics-probability)
* [Wikipedia: Cross-validation](https://en.wikipedia.org/wiki/Cross-validation_%28statistics%29)
* [Analytics Vidhya: Cross-validation Guide](https://www.analyticsvidhya.com/blog/2021/11/cross-validation-in-machine-learning/)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>