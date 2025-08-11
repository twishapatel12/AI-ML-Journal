![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# Bayes’ Theorem in Machine Learning

---

## Introduction

**Bayes’ Theorem** is a fundamental concept in probability and statistics, with powerful applications in machine learning (ML) and artificial intelligence (AI).  
It allows us to update our beliefs about the world as we observe new evidence—making it the foundation for probabilistic modeling, classification, and inference in AI/ML.

---

## What is Bayes’ Theorem?

Bayes’ Theorem relates the conditional and marginal probabilities of random events.  
It is mathematically expressed as:

\[
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
\]

Where:
- \( P(A|B) \): Probability of event A, given that B has occurred (posterior)
- \( P(B|A) \): Probability of event B, given that A has occurred (likelihood)
- \( P(A) \): Probability of event A (prior)
- \( P(B) \): Probability of event B (evidence)

---

## Why is Bayes’ Theorem Useful in ML?

- **Learning from new data:** Updates model beliefs as new data arrives.
- **Classification:** Powers Naive Bayes classifiers and Bayesian inference.
- **Combining expert knowledge and data:** Prior knowledge + evidence = better predictions.

---

## Simple Example: Medical Diagnosis

Suppose a disease affects 1% of people.  
A test for the disease is 99% accurate (both for positive and negative).  
What is the chance someone really has the disease, given a positive test?

Let:
- \( D \): Has disease
- \( T \): Test is positive

Given:
- \( P(D) = 0.01 \) (prior)
- \( P(T|D) = 0.99 \) (true positive rate)
- \( P(T|\neg D) = 0.01 \) (false positive rate)

By Bayes’ Theorem:

\[
P(D|T) = \frac{P(T|D) \cdot P(D)}{P(T|D) \cdot P(D) + P(T|\neg D) \cdot P(\neg D)}
\]

Plug in numbers:

\[
P(D|T) = \frac{0.99 \times 0.01}{0.99 \times 0.01 + 0.01 \times 0.99} = \frac{0.0099}{0.0099 + 0.0099} = 0.5
\]

So, despite the accurate test, the probability the person really has the disease after a positive test is only **50%**.

---

### Visual: Bayes’ Theorem in Action

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/bayes-theorem-visual.png" alt="Bayes Theorem Visual" width="500"/>
</p>

---

## Bayes’ Theorem in Machine Learning

### 1. Naive Bayes Classifier

A fast and simple ML algorithm used for text classification, spam filtering, sentiment analysis, and more.

- **Assumes features are independent** (which is “naive” but works surprisingly well).
- Calculates the posterior probability of each class given observed features.

**Code Example: Email Spam Classification**

```python
from sklearn.naive_bayes import MultinomialNB

X = [[2, 1], [1, 2], [2, 3], [3, 1]]  # Example word counts [“free”, “offer”]
y = [1, 0, 1, 0]  # 1 = spam, 0 = not spam

model = MultinomialNB()
model.fit(X, y)
print("Spam probability for [2,2]:", model.predict_proba([[2,2]]))
````

---

### 2. Bayesian Networks and Inference

* Used for modeling complex dependencies and uncertainty in ML.
* Examples: medical diagnosis, recommendation systems, sensor fusion.

---

### 3. Bayesian Deep Learning

* Models not just predictions, but uncertainty in those predictions.
* Useful for tasks like medical AI, autonomous vehicles, and any critical system.

---

## Key Concepts

| Term       | Meaning                                     |
| ---------- | ------------------------------------------- |
| Prior      | What you believed before seeing the data    |
| Likelihood | How probable is the data, given your belief |
| Evidence   | Overall probability of seeing the data      |
| Posterior  | Updated belief after seeing the data        |

---

## Practical Applications

* **Spam detection**
* **Medical diagnosis**
* **Predictive text and language models**
* **Fraud detection**
* **Recommender systems**

---

## Further Reading & References

* [Khan Academy: Bayes’ Theorem](https://www.khanacademy.org/math/statistics-probability/probability-library)
* [Wikipedia: Bayes’ Theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem)
* [DeepLearning.ai: Bayes’ Theorem for ML](https://www.deeplearning.ai/resources/bayes-theorem-for-machine-learning/)
* [StatQuest: Bayes’ Theorem (YouTube)](https://www.youtube.com/watch?v=HZGCoVF3YvM)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>