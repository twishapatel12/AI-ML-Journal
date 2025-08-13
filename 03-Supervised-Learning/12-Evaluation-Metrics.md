![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# Evaluation Metrics in Machine Learning

---

## Introduction

**Evaluation metrics** help us measure how well a machine learning model performs.  
They guide us in:
- Comparing different models
- Selecting the best model
- Understanding model strengths and weaknesses

The choice of metric depends on:
- **Problem type** (classification, regression, ranking, etc.)
- **Data characteristics** (imbalanced classes, noise, etc.)
- **Business goals** (prioritizing certain errors over others)

---

## 1. Classification Metrics

Used when the target variable is categorical.

### 1.1 Confusion Matrix

A table showing the counts of **predicted vs. actual** classes.

|                | Predicted Positive | Predicted Negative |
|----------------|--------------------|--------------------|
| **Actual Positive** | True Positive (TP)   | False Negative (FN)  |
| **Actual Negative** | False Positive (FP)  | True Negative (TN)   |

**Code Example:**
```python
from sklearn.metrics import confusion_matrix
y_true = [1, 0, 1, 1, 0]
y_pred = [1, 0, 1, 0, 0]
print(confusion_matrix(y_true, y_pred))
````

---

### 1.2 Accuracy

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

* Fraction of correct predictions.
* **Limitation:** Misleading for imbalanced datasets.

---

### 1.3 Precision

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

* Out of predicted positives, how many are correct.
* Important when **false positives** are costly (e.g., spam detection).

---

### 1.4 Recall (Sensitivity or True Positive Rate)

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

* Out of actual positives, how many did we correctly predict.
* Important when **false negatives** are costly (e.g., cancer detection).

---

### 1.5 F1 Score

$$
\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

* Harmonic mean of precision and recall.
* Useful when you need a balance between precision and recall.

---

### 1.6 ROC Curve & AUC

* **ROC Curve:** Plots True Positive Rate (Recall) vs. False Positive Rate.
* **AUC (Area Under Curve):** Probability that the model ranks a random positive higher than a random negative.

**Code Example:**

```python
from sklearn.metrics import roc_auc_score
y_true = [0, 0, 1, 1]
y_scores = [0.1, 0.4, 0.35, 0.8]
print("AUC:", roc_auc_score(y_true, y_scores))
```

---

### 1.7 PR Curve (Precision-Recall Curve)

* Plots precision vs. recall at various thresholds.
* More informative than ROC when classes are imbalanced.

---

## 2. Regression Metrics

Used when the target variable is continuous.

### 2.1 Mean Absolute Error (MAE)

$$
\text{MAE} = \frac{1}{n} \sum |y_i - \hat{y}_i|
$$

* Average absolute difference between actual and predicted values.
* Easy to interpret.

---

### 2.2 Mean Squared Error (MSE)

$$
\text{MSE} = \frac{1}{n} \sum (y_i - \hat{y}_i)^2
$$

* Squares errors, penalizing larger errors more.
* Sensitive to outliers.

---

### 2.3 Root Mean Squared Error (RMSE)

$$
\text{RMSE} = \sqrt{\text{MSE}}
$$

* Same units as the target variable.
* Easier to interpret than MSE.

---

### 2.4 $R^2$ Score (Coefficient of Determination)

$$
R^2 = 1 - \frac{\text{SS}_{res}}{\text{SS}_{tot}}
$$

* Proportion of variance in the target explained by the model.
* 1 = perfect fit, 0 = no better than mean prediction.

---

## 3. Choosing the Right Metric

| Problem Type               | Recommended Metrics                        |
| -------------------------- | ------------------------------------------ |
| Binary classification      | Accuracy, Precision, Recall, F1, ROC-AUC   |
| Multi-class classification | Macro/micro averaged Precision, Recall, F1 |
| Imbalanced classification  | Precision, Recall, F1, PR-AUC              |
| Regression                 | MAE, MSE, RMSE, $R^2$                      |

---

## Visual: Classification Metrics Summary

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/evaluation-metrics-classification.png" alt="Classification Metrics Summary" width="500"/>
</p>

*Diagram showing how Accuracy, Precision, Recall, and F1 relate to confusion matrix outcomes.*

---

## References

* [Scikit-learn: Model Evaluation Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
* [Wikipedia: Precision and Recall](https://en.wikipedia.org/wiki/Precision_and_recall)
* [Analytics Vidhya: Evaluation Metrics](https://www.analyticsvidhya.com/blog/2020/07/20-evaluation-metrics-machine-learning-models/)
* [Khan Academy: ROC and AUC](https://www.khanacademy.org/math/statistics-probability)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>