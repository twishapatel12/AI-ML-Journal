![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# Support Vector Machines (SVMs)

---

## Introduction

**Support Vector Machines (SVMs)** are powerful and versatile machine learning algorithms used for both classification and regression tasks.  
They are especially known for their ability to find robust decision boundaries—even in high-dimensional spaces—and are effective on both linear and non-linear problems.

---

## What is an SVM?

SVMs aim to find the **best possible separating line (or hyperplane)** between two classes.  
The best hyperplane is the one that **maximizes the margin**—the distance between the nearest points (support vectors) of each class and the hyperplane itself.

- **Margin:** The space between the hyperplane and the closest data points from either class.
- **Support vectors:** The data points that “support” the optimal margin; they are the most critical samples in the dataset.

---

## Visual: SVM Decision Boundary

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/svm-decision-boundary.png" alt="SVM Decision Boundary" width="480"/>
</p>

*Shows two classes of points, the separating hyperplane, margins, and support vectors.*

---

## Linear vs. Non-Linear SVMs

- **Linear SVM:** Works when data is linearly separable (can be split with a straight line/plane).
- **Non-linear SVM:** Uses the “kernel trick” to map data to a higher-dimensional space, making it possible to separate with a hyperplane even if data isn’t linearly separable.

**Common kernels:**  
- Linear
- Polynomial
- Radial Basis Function (RBF, or Gaussian)

---

## How Does an SVM Work?

1. **Find the optimal hyperplane** that separates classes with the largest margin.
2. **Use support vectors** to define this margin.
3. **For non-linear data,** transform input space using a kernel function so a linear separator is possible in the new space.

---

## SVM for Classification

- SVM outputs +1 for one class and -1 for the other.
- For probabilities/confidence, SVMs can be calibrated.

---

## Code Example: SVM in Python (Classification)

```python
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

# Simple example: points and classes
X = np.array([[2,2], [1,3], [2,0], [0,1], [3,1], [4,4]])
y = np.array([1, 1, 0, 0, 0, 1])

model = svm.SVC(kernel='linear', C=1.0)
model.fit(X, y)

# Visualize decision boundary
w = model.coef_[0]
b = model.intercept_[0]
x_vals = np.linspace(0, 5, 100)
y_vals = -(w[0]/w[1])*x_vals - b/w[1]

plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr', s=60)
plt.plot(x_vals, y_vals, 'k-')
plt.title('SVM Linear Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
````

---

## SVM for Regression (SVR)

SVMs can also be used for regression problems (predicting continuous values) using a variant called **Support Vector Regression (SVR)**.

---

## Advantages and Limitations

**Advantages:**

* Effective in high-dimensional spaces.
* Works well even when the number of features exceeds the number of samples.
* Robust to outliers (with proper parameter tuning).
* Can model complex boundaries with kernel trick.

**Limitations:**

* Less effective on very large datasets (training time increases quickly).
* Requires careful tuning of hyperparameters (kernel, C, gamma).
* Output is less interpretable than simple linear models.

---

## Real-World Applications

* Text classification (spam detection, sentiment analysis)
* Image classification (handwriting recognition)
* Bioinformatics (protein classification)
* Face recognition

---

## Further Reading & References

* [Scikit-learn: SVM Documentation](https://scikit-learn.org/stable/modules/svm.html)
* [Wikipedia: Support Vector Machine](https://en.wikipedia.org/wiki/Support_vector_machine)
* [Analytics Vidhya: SVM Guide](https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/)
* [Khan Academy: SVMs](https://www.khanacademy.org/math/statistics-probability)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>