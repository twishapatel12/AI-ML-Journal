![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# Handling Missing Data in Machine Learning

---

## Introduction

In real-world datasets, **missing data** is very common.  
It can occur due to:
- Human error (e.g., incomplete forms)
- System failures (e.g., sensors not recording)
- Data corruption or merging issues
- Intentional omission (e.g., privacy concerns)

If not handled properly, missing values can:
- Reduce the amount of usable data
- Bias model results
- Lower model performance

---

## 1. Identifying Missing Data

In Python (Pandas):
```python
import pandas as pd

df.isnull().sum()   # Count missing values per column
df.info()           # Check non-null counts
````

In datasets:

* Missing values might appear as **NaN**, `None`, `"?"`, `"NA"`, or even empty strings.

---

## 2. Types of Missing Data

### 2.1 MCAR — Missing Completely at Random

* Missingness is unrelated to any data (observed or unobserved).
* Example: Random server error causing data loss.

### 2.2 MAR — Missing at Random

* Missingness is related to observed variables but not the missing ones themselves.
* Example: Income missing more often for younger people.

### 2.3 MNAR — Missing Not at Random

* Missingness is related to the missing values themselves.
* Example: People with very high income not reporting it.

> The type of missingness affects which imputation methods are appropriate.

---

## 3. Strategies for Handling Missing Data

### 3.1 Deletion Methods

#### a. Listwise Deletion (Remove Rows)

* Remove all rows with missing values.

```python
df.dropna(inplace=True)
```

**Pros:**

* Simple and fast.

**Cons:**

* Can lose a lot of data.
* Risk of bias if data is not MCAR.

#### b. Column Deletion

* Remove columns with too many missing values (e.g., >70% missing).

```python
df.drop(columns=['column_name'], inplace=True)
```

---

### 3.2 Imputation Methods

#### a. Simple Imputation

* Fill missing values with **mean**, **median**, or **mode**.

```python
df['age'].fillna(df['age'].mean(), inplace=True)  # Mean imputation
```

**When to use:**

* Numerical data: mean/median
* Categorical data: mode

#### b. Constant Value Imputation

* Fill with a fixed value (e.g., 0, `"Unknown"`).

```python
df['city'].fillna('Unknown', inplace=True)
```

#### c. Forward/Backward Fill (Time Series)

* Use previous or next value in the column.

```python
df['value'].fillna(method='ffill', inplace=True)
```

---

### 3.3 Advanced Imputation

#### a. k-Nearest Neighbors (KNN) Imputation

* Uses similar rows to estimate missing values.

```python
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df_imputed = imputer.fit_transform(df)
```

#### b. Multivariate Imputation (MICE)

* Uses regression models to predict missing values from other variables.

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer()
df_imputed = imputer.fit_transform(df)
```

#### c. Model-Based Imputation

* Train a model to predict the missing values.

---

### 3.4 Indicator for Missingness

* Add a binary column indicating whether the value was missing.

```python
df['age_missing'] = df['age'].isnull().astype(int)
```

* Helps the model learn if missingness itself carries information.

---

## 4. Choosing the Right Method

| Situation                       | Recommended Method                             |
| ------------------------------- | ---------------------------------------------- |
| Few missing values              | Simple imputation (mean/median/mode)           |
| Many missing values in a column | Drop the column                                |
| MCAR                            | Any standard method                            |
| MAR                             | Imputation using related variables (KNN, MICE) |
| MNAR                            | Add missing indicator + advanced imputation    |
| Time series data                | Forward/Backward fill                          |

---

## 5. Best Practices

* Always analyze the **pattern** of missing data before choosing a method.
* Avoid imputing test data using test statistics—**fit imputer on training data only**.
* Compare model performance with and without imputation.
* For categorical variables, consider `"Unknown"` category instead of dropping.

---

## Visual: Handling Missing Data Workflow

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/missing-data-workflow.png" alt="Missing Data Handling Workflow" width="500"/>
</p>

---

## References

* [Scikit-learn: Imputation](https://scikit-learn.org/stable/modules/impute.html)
* [Kaggle: Handling Missing Values](https://www.kaggle.com/code/rtatman/data-cleaning-challenge-handling-missing-values)
* [Analytics Vidhya: Missing Value Imputation](https://www.analyticsvidhya.com/blog/2021/10/a-definitive-guide-to-data-imputation-in-machine-learning/)
* [Little, R. J., & Rubin, D. B. (2019). Statistical Analysis with Missing Data](https://onlinelibrary.wiley.com/doi/book/10.1002/9781119482260)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>