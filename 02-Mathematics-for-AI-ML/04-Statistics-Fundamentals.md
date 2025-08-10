![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# Statistics Fundamentals for Machine Learning

---

## Introduction

**Statistics** is the science of collecting, analyzing, and interpreting data.  
In machine learning, statistics helps us summarize data, estimate relationships, and make predictions with confidence.  
Understanding the basics of statistics is essential for feature engineering, model evaluation, and communicating results in ML projects.

---

## Why Statistics Matters in ML

- **Summarizing data:** Quickly understand large datasets using key numbers.
- **Spotting patterns:** Discover relationships between variables.
- **Testing hypotheses:** Check if results are due to chance.
- **Evaluating models:** Use statistical measures to judge performance.
- **Communicating findings:** Visualize and report insights clearly.

---

## Key Statistical Concepts

### 1. Descriptive Statistics

#### a. Mean, Median, and Mode

- **Mean:** The “average.” Add up all values, divide by count.
- **Median:** The middle value when data is sorted.
- **Mode:** The most frequently occurring value.

**Example:**

```python
import numpy as np
data = [3, 5, 5, 7, 9]
print("Mean:", np.mean(data))
print("Median:", np.median(data))
print("Mode:", max(set(data), key=data.count))
````

#### b. Range, Variance, and Standard Deviation

* **Range:** Difference between max and min.
* **Variance:** Average squared difference from the mean (spread).
* **Standard Deviation:** Square root of variance (how much values typically differ from the mean).

---

### 2. Data Visualization

* **Histograms:** Show distribution of values.
* **Boxplots:** Show median, quartiles, and outliers.
* **Scatter plots:** Show relationships between two variables.

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/statistics-histogram.png" alt="Histogram Example" width="420"/>
</p>

---

### 3. Correlation and Covariance

* **Correlation:** Measures how two variables change together, from -1 (perfect negative) to +1 (perfect positive), 0 = none.
* **Covariance:** Similar, but not standardized.

**Example:**

```python
import numpy as np
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
print("Correlation:", np.corrcoef(x, y)[0, 1])
print("Covariance:", np.cov(x, y)[0, 1])
```

---

### 4. Probability Distributions

* **Normal (Gaussian):** Bell curve, many real-world things (heights, test scores).
* **Uniform:** All values equally likely.
* **Binomial:** Number of successes in fixed trials.

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/normal-vs-uniform.png" alt="Normal vs Uniform Distribution" width="420"/>
</p>

---

### 5. Inferential Statistics

Helps us make predictions about a population based on a sample.

#### a. Sampling

* Select a smaller group (sample) from a larger group (population).
* **Random sampling** is best for unbiased results.

#### b. Confidence Intervals

* Range of values that likely includes the true population parameter.
* “We are 95% confident that the average height is between 160–170 cm.”

#### c. Hypothesis Testing

* **Null hypothesis (H₀):** No effect or difference.
* **Alternative hypothesis (H₁):** There is an effect or difference.
* Use tests (t-test, chi-square, etc.) to check significance.

---

## Real-World Example: Are Students Taller in City A or City B?

Suppose you collect heights from students in two cities and want to know if one city’s students are taller **on average**.

**Step 1:** Summarize the data (mean, std).
**Step 2:** Plot histograms and boxplots.
**Step 3:** Run a t-test to see if the means are significantly different.

**Code Example:**

```python
import numpy as np
from scipy.stats import ttest_ind
city_a = [160, 165, 168, 170, 175]
city_b = [155, 159, 163, 166, 168]
print("Mean A:", np.mean(city_a), "Mean B:", np.mean(city_b))
stat, pval = ttest_ind(city_a, city_b)
print("p-value:", pval)
if pval < 0.05:
    print("Difference is significant!")
else:
    print("No significant difference.")
```

---

### Visual: Boxplot Comparison

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/boxplot-city-heights.png" alt="Boxplot of City Heights" width="420"/>
</p>

---

## Statistics in ML Model Evaluation

* **Accuracy, Precision, Recall, F1-score:** For classification models
* **Mean Squared Error (MSE):** For regression
* **ROC Curve, AUC:** For classifier quality
* **Confusion Matrix:** Summarizes prediction errors

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/confusion-matrix.png" alt="Confusion Matrix" width="320"/>
</p>

---

## Best Practices

* Always visualize data before modeling.
* Use summary stats to check for outliers or errors.
* Report confidence intervals, not just point estimates.
* Choose the right statistical test for your question.

---

## Further Reading & References

* [Khan Academy: Statistics and Probability](https://www.khanacademy.org/math/statistics-probability)
* [Wikipedia: Statistical Hypothesis Testing](https://en.wikipedia.org/wiki/Statistical_hypothesis_testing)
* [StatQuest: Statistical Concepts (YouTube)](https://www.youtube.com/c/joshstarmer)
* [Scikit-learn Metrics & Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>