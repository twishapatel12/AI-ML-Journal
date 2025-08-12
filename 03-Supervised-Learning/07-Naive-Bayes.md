![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# Naive Bayes Classifier

---

## Introduction

**Naive Bayes** is a family of simple but surprisingly powerful algorithms for classification problems in machine learning.  
It’s especially popular for tasks involving text (like spam detection and sentiment analysis) because it’s fast, requires little data, and can handle lots of features.

The “naive” part? It assumes all features are independent—even if they aren’t in reality. This assumption makes the math easy, and in practice, Naive Bayes often works very well.

---

## The Core Idea: Bayes’ Theorem

Naive Bayes classifiers use **Bayes’ Theorem** to calculate the probability of each class given the input features:

$$
P(\text{Class} | \text{Features}) = \frac{P(\text{Features} | \text{Class}) \cdot P(\text{Class})}{P(\text{Features})}
$$

- $P(\text{Class} | \text{Features})$: Probability that the example is in a class, given its features.
- $P(\text{Features} | \text{Class})$: Likelihood of seeing those features if you’re in that class.
- $P(\text{Class})$: Prior probability of the class.
- $P(\text{Features})$: Probability of the features (same for all classes, can be ignored for comparison).

---

## Why “Naive”?

- Naive Bayes assumes all features are **independent** given the class.
- This simplifies calculations but rarely holds perfectly in the real world.
- Despite the “naive” assumption, it works well in many cases—especially with lots of features and not much data.

---

## Types of Naive Bayes

- **Gaussian Naive Bayes:** For continuous features (assumes they follow a normal distribution).
- **Multinomial Naive Bayes:** For counts (e.g., word counts in text).
- **Bernoulli Naive Bayes:** For binary/boolean features (e.g., word appears/doesn’t appear).

---

## Real-World Example: Spam Detection

Suppose you want to classify emails as spam or not spam, based on the words they contain.

- For each word, Naive Bayes learns the probability of the word appearing in spam and in non-spam emails.
- For a new email, it multiplies the probabilities for all words and picks the class with the highest final probability.

---

## Visual: Naive Bayes for Email Classification

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/naive-bayes-diagram.png" alt="Naive Bayes Email Classification" width="480"/>
</p>

*Shows email features going into a Naive Bayes classifier, producing probabilities for spam/not spam.*

---

## Code Example: Naive Bayes in Python

```python
from sklearn.naive_bayes import MultinomialNB

# Toy example: [count of "free", count of "offer"]
X = [[2, 0], [0, 1], [1, 1], [0, 0]]
y = [1, 0, 1, 0]  # 1 = spam, 0 = not spam

model = MultinomialNB()
model.fit(X, y)
print("Prediction for [1, 0]:", model.predict([[1, 0]]))         # Expected: spam
print("Class probabilities for [1, 0]:", model.predict_proba([[1, 0]]))
````

---

## Advantages and Limitations

**Advantages:**

* Extremely fast to train and predict.
* Works well with high-dimensional, sparse data (like text).
* Performs surprisingly well even with the independence assumption.

**Limitations:**

* Assumes features are independent (may not hold in practice).
* Can be outperformed by more complex models for some problems.
* Probability outputs can be unreliable for correlated features.

---

## Use Cases

* Email spam filtering
* Sentiment analysis
* News/article classification
* Medical diagnosis (symptom presence/absence)

---

## Best Practices

* Preprocess text data carefully (tokenization, stop-word removal, etc.).
* Try different variants (Multinomial, Gaussian, Bernoulli) based on feature type.
* For unbalanced datasets, consider adjusting priors or using class weights.

---

## Further Reading & References

* [Scikit-learn: Naive Bayes Documentation](https://scikit-learn.org/stable/modules/naive_bayes.html)
* [Wikipedia: Naive Bayes Classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
* [Analytics Vidhya: Naive Bayes Guide](https://www.analyticsvidhya.com/blog/2017/09/naive-bayes-explained/)
* [Khan Academy: Bayes’ Theorem](https://www.khanacademy.org/math/statistics-probability)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>