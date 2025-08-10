![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# Probability for Machine Learning

---

## Introduction

Probability is the mathematics of uncertainty.  
In machine learning (ML), probability helps us handle noisy data, make predictions under uncertainty, and build models that can learn from patterns, not just rules.  
Understanding probability is essential for nearly every area of AI/ML, from basic classifiers to deep learning and reinforcement learning.

---

## Why is Probability Important in ML?

- **Data is noisy and incomplete:** Probability lets us model the likelihood of different outcomes, even when the data is not perfect.
- **Predictions are uncertain:** Models predict not just “yes/no,” but how confident they are.
- **Probabilistic models power ML:** Algorithms like Naive Bayes, logistic regression, and Bayesian networks rely on probability theory.
- **Helps prevent overfitting:** Probability theory underlies regularization, Bayesian inference, and much more.

---

## Key Concepts

### 1. Random Variables

A **random variable** is a variable whose possible values are outcomes of a random phenomenon.

- **Discrete random variable:** Takes on countable values (e.g., rolling a die: 1–6)
- **Continuous random variable:** Takes on any value in a range (e.g., height, weight)

---

### 2. Probability Distributions

- **Probability distribution:** Tells us the likelihood of each possible value of a random variable.

| Type         | Example                    | Description                  |
|--------------|----------------------------|------------------------------|
| Discrete     | Binomial, Bernoulli        | Yes/no, heads/tails, counts  |
| Continuous   | Gaussian (Normal), Exponential | Heights, scores, times  |

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/gaussian-vs-bernoulli.png" alt="Gaussian vs Bernoulli Distribution" width="400"/>
</p>

---

### 3. Mean, Variance, and Standard Deviation

- **Mean (expected value):** Average outcome.
- **Variance:** How spread out the outcomes are.
- **Standard deviation:** Square root of variance (measures “typical” deviation).

**Example:**

```python
import numpy as np

data = [1, 2, 3, 4, 5]
print("Mean:", np.mean(data))
print("Variance:", np.var(data))
print("Standard deviation:", np.std(data))
````

---

### 4. Conditional Probability

* **Conditional probability:** Probability of event A given that B has occurred:
  `P(A|B) = P(A and B) / P(B)`
* **Used for:** Prediction, recommendation, generative models.

---

### 5. Bayes’ Theorem

One of the most important formulas in ML:

```
P(A|B) = [P(B|A) * P(A)] / P(B)
```

* Lets you update your belief about A after observing B.
* Foundation of **Naive Bayes classifiers** and Bayesian ML.

**Example: Spam Classification**

If “free money” appears, what’s the probability an email is spam?

---

### 6. Independence

Two events A and B are **independent** if knowing A does not change the probability of B:

* `P(A and B) = P(A) * P(B)`

This concept is essential for simple models like Naive Bayes, which assumes features are independent.

---

### 7. Likelihood and Maximum Likelihood Estimation (MLE)

* **Likelihood:** Probability of the observed data given model parameters.
* **MLE:** Find parameters that maximize the likelihood.

**Example:**
Fitting a normal distribution to observed data by maximizing the likelihood.

---

## How Probability is Used in ML

* **Classification:** Predicts probabilities for each class (e.g., spam vs not spam)
* **Regression:** Predicts probability distributions for outputs, not just single values
* **Regularization:** Uses probability to prevent overfitting
* **Uncertainty estimation:** Deep learning models output not just predictions, but their confidence

---

## Code Example: Naive Bayes Classifier

```python
from sklearn.naive_bayes import GaussianNB

X = [[1.8, 6.2], [1.7, 5.8], [3.2, 7.3], [3.0, 6.9]]  # [feature1, feature2]
y = [0, 0, 1, 1]  # Classes: 0 or 1

model = GaussianNB()
model.fit(X, y)
print("Predicted class:", model.predict([[2.5, 7.0]]))
print("Probabilities:", model.predict_proba([[2.5, 7.0]]))
```

---

## Visual: Probability in ML

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/probability-ml-visual.png" alt="Probability in Machine Learning Diagram" width="500"/>
</p>

---

## Common Probability Distributions in ML

* **Bernoulli:** Binary outcomes (success/failure)
* **Binomial:** Number of successes in multiple trials
* **Gaussian/Normal:** Many natural processes, used for continuous data
* **Poisson:** Counting rare events
* **Multinomial:** Multi-class categorical data

---

## Further Reading & References

* [Khan Academy: Probability and Statistics](https://www.khanacademy.org/math/statistics-probability)
* [Stanford CS229: Probability Review](https://see.stanford.edu/materials/aimlcs229/cs229-prob.pdf)
* [DeepLearning.ai: Probability for Machine Learning](https://www.deeplearning.ai/resources/probability-for-machine-learning/)
* [Wikipedia: Probability Distribution](https://en.wikipedia.org/wiki/Probability_distribution)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>
