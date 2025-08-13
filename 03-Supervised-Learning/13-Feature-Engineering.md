![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# Feature Engineering in Machine Learning

---

## Introduction

**Feature engineering** is the process of transforming raw data into meaningful input variables (**features**) that improve the performance of a machine learning model.  
It is often considered **the most important step** in the ML pipeline—good features can be more impactful than the choice of algorithm.

The goal is to:
- Improve **model accuracy**
- Reduce **model complexity**
- Enhance **interpretability**

---

## Why Feature Engineering Matters

- Machine learning algorithms learn patterns from **features**, not raw data.
- Poorly engineered features can cause **underfitting** or **overfitting**.
- High-quality features can help even simple models outperform complex ones.

> "Better data beats fancier algorithms." — Peter Norvig (Google Research Director)

---

## Steps in Feature Engineering

### 1. **Feature Creation**
Creating new features from existing ones to better represent the underlying patterns.

Examples:
- **Polynomial features:** $x^2, x^3$ to capture non-linear relationships.
- **Ratios:** `total_sales / num_customers`.
- **Date/time parts:** Extracting `year`, `month`, `day`, `day_of_week` from a timestamp.

```python
import pandas as pd
df['purchase_ratio'] = df['total_sales'] / df['num_customers']
df['purchase_month'] = pd.to_datetime(df['purchase_date']).dt.month
````

---

### 2. **Feature Transformation**

Changing the scale or distribution of features to help algorithms perform better.

* **Normalization (Min-Max Scaling):**

$$
x' = \frac{x - \min(x)}{\max(x) - \min(x)}
$$

  Useful for algorithms like KNN, SVM.

* **Standardization (Z-score Scaling):**

$$
x' = \frac{x - \mu}{\sigma}
$$

  Useful for linear models and gradient descent.

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = StandardScaler()
df[['feature1', 'feature2']] = scaler.fit_transform(df[['feature1', 'feature2']])
```

---

### 3. **Encoding Categorical Variables**

Most algorithms require numeric inputs.

* **One-Hot Encoding:** For nominal categories.
* **Label Encoding:** For ordinal categories.
* **Target Encoding:** Replace category with mean target value (use with caution to avoid leakage).

```python
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
encoded = encoder.fit_transform(df[['category']])
```

---

### 4. **Handling Missing Values**

* **Deletion:** Remove rows or columns with too many missing values.
* **Imputation:** Replace missing values with mean, median, mode, or predictions from other features.

```python
df['age'].fillna(df['age'].median(), inplace=True)
```

---

### 5. **Feature Selection**

Choosing the most relevant features to:

* Improve model performance
* Reduce overfitting
* Decrease training time

Methods:

* **Filter methods:** Correlation, Chi-square test
* **Wrapper methods:** Recursive Feature Elimination (RFE)
* **Embedded methods:** Lasso (L1 regularization), tree-based feature importance

```python
from sklearn.feature_selection import SelectKBest, f_classif
X_new = SelectKBest(f_classif, k=10).fit_transform(X, y)
```

---

### 6. **Binning**

Grouping continuous values into discrete bins.

Example:

* Age → "child", "teen", "adult", "senior"

```python
bins = [0, 12, 19, 59, 100]
labels = ['child', 'teen', 'adult', 'senior']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)
```

---

### 7. **Log Transformation**

Useful for skewed data to reduce the impact of outliers.

```python
import numpy as np
df['log_income'] = np.log1p(df['income'])
```

---

### 8. **Dimensionality Reduction**

Reducing the number of features while retaining most of the information.

* **PCA (Principal Component Analysis)**
* **t-SNE** (for visualization)

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)
```

---

## Best Practices

* Understand the **domain** and business logic before creating features.
* Avoid **data leakage** (using future or target-related info in features).
* Keep a pipeline of preprocessing steps using `sklearn.Pipeline` for reproducibility.
* Test feature importance using model explainability tools like **SHAP** or **LIME**.

---

## Example: Feature Engineering Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

numeric_features = ['age', 'income']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['gender', 'city']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', LogisticRegression())])

model.fit(X_train, y_train)
```

---

## References

* [Scikit-learn: Preprocessing Data](https://scikit-learn.org/stable/modules/preprocessing.html)
* [Kaggle: Feature Engineering Courses](https://www.kaggle.com/learn/feature-engineering)
* [Analytics Vidhya: Guide to Feature Engineering](https://www.analyticsvidhya.com/blog/2020/10/feature-engineering-in-machine-learning/)
* [Google Developers: Feature Engineering](https://developers.google.com/machine-learning/data-prep/construct/transform)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>