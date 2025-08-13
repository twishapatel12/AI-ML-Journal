![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# Encoding Categorical Variables

---

## Introduction

Many machine learning algorithms **require numerical inputs**—they cannot directly process text labels or symbolic values.  
**Encoding categorical variables** means converting categories into numbers in a way that preserves (or appropriately represents) their meaning.

The choice of encoding method depends on:
- Whether the categorical variable is **nominal** (no order) or **ordinal** (order matters)
- The **number of unique categories**
- The algorithm being used
- The amount of data available

---

## 1. Types of Categorical Variables

### 1.1 Nominal
- Categories have **no inherent order**
- Examples:  
  - Color: Red, Blue, Green  
  - City: London, Paris, New York

### 1.2 Ordinal
- Categories have a **meaningful order**
- Examples:  
  - Education level: High School < Bachelor’s < Master’s < PhD  
  - Size: Small < Medium < Large

---

## 2. Encoding Methods

### 2.1 Label Encoding

Assigns an integer value to each category.

**Example:**
| Education Level | Encoded |
|-----------------|---------|
| High School     | 0       |
| Bachelor’s      | 1       |
| Master’s        | 2       |
| PhD             | 3       |

**Pros:**
- Simple and memory efficient.
- Keeps ordinal relationships intact.

**Cons:**
- For **nominal variables**, may mislead algorithms into thinking there’s an order.

**Code Example:**
```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df['education_encoded'] = encoder.fit_transform(df['education'])
````

---

### 2.2 One-Hot Encoding

Creates a binary column for each category, indicating presence (1) or absence (0).

**Example:**

| Color | Color\_Red | Color\_Blue | Color\_Green |
| ----- | ---------- | ----------- | ------------ |
| Red   | 1          | 0           | 0            |
| Blue  | 0          | 1           | 0            |
| Green | 0          | 0           | 1            |

**Pros:**

* No assumption of order.
* Works well with nominal variables.

**Cons:**

* Increases dimensionality (many columns for many categories).
* Can lead to **sparse data** for high-cardinality features.

**Code Example:**

```python
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

encoder = OneHotEncoder(sparse=False, drop='first')  # drop first to avoid multicollinearity
encoded = encoder.fit_transform(df[['color']])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['color']))
df = pd.concat([df, encoded_df], axis=1)
```

---

### 2.3 Ordinal Encoding

Assigns integers to categories in the order specified.

**Example:**

| Size   | Encoded |
| ------ | ------- |
| Small  | 1       |
| Medium | 2       |
| Large  | 3       |

**Pros:**

* Preserves the order for ordinal data.
* Simple and efficient.

**Cons:**

* Should **not** be used for nominal data.

**Code Example:**

```python
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder(categories=[['Small', 'Medium', 'Large']])
df['size_encoded'] = encoder.fit_transform(df[['size']])
```

---

### 2.4 Target Encoding

Replaces each category with a summary statistic of the target variable (usually the mean).

**Example:**
If predicting whether a customer will buy (1) or not (0):

| City   | Mean Target |
| ------ | ----------- |
| Paris  | 0.8         |
| London | 0.6         |
| NYC    | 0.3         |

**Pros:**

* Handles high-cardinality variables well.
* Encodes useful information about the target.

**Cons:**

* Risk of **data leakage** (use CV-based encoding to prevent this).

**Code Example:**

```python
import category_encoders as ce

encoder = ce.TargetEncoder(cols=['city'])
df['city_encoded'] = encoder.fit_transform(df['city'], df['target'])
```

---

### 2.5 Frequency Encoding

Replaces categories with their frequency or count in the dataset.

**Example:**

| City   | Frequency |
| ------ | --------- |
| Paris  | 500       |
| London | 300       |
| NYC    | 200       |

**Pros:**

* Simple and works for tree-based models.
* No increase in dimensionality.

**Cons:**

* May lose meaning if frequencies are similar for different categories.

---

### 2.6 Hash Encoding

Maps categories to a fixed number of columns using a hash function.

**Pros:**

* Handles very high-cardinality features efficiently.
* Avoids large memory use from one-hot encoding.

**Cons:**

* Collisions (different categories mapped to same column).

---

## 3. Choosing the Right Encoding Method

| Feature Type                       | Best Options                                 |
| ---------------------------------- | -------------------------------------------- |
| Nominal, low cardinality           | One-Hot Encoding                             |
| Nominal, high cardinality          | Target Encoding, Frequency Encoding, Hashing |
| Ordinal                            | Ordinal Encoding, Label Encoding             |
| Sensitive to linearity assumptions | Avoid Label Encoding for nominal features    |

---

## 4. Best Practices

* For **linear models**, avoid label encoding for nominal data (can distort relationships).
* For **tree-based models** (Decision Trees, Random Forest, XGBoost), label encoding can work fine because they split on feature values.
* Be mindful of **data leakage**—fit encoders only on the training set and then transform the test set.
* Consider reducing dimensions after one-hot encoding if the feature space explodes.

---

## Visual: Encoding Overview

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/encoding-methods.png" alt="Encoding Methods Overview" width="500"/>
</p>

---

## References

* [Scikit-learn: Preprocessing Categorical Features](https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features)
* [Category Encoders Library](https://contrib.scikit-learn.org/category_encoders/)
* [Kaggle: Encoding Categorical Variables](https://www.kaggle.com/code/dansbecker/using-categorical-data-with-one-hot-encoding)
* [Analytics Vidhya: Categorical Encoding Guide](https://www.analyticsvidhya.com/blog/2020/08/types-of-categorical-data-encoding/)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>