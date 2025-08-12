![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# Softmax Function and Log Loss in Machine Learning

---

## Introduction

The **softmax function** and **log loss** (also called cross-entropy loss) are foundational tools in machine learning—especially for classification problems with more than two classes (multiclass classification).

Softmax turns raw model scores (logits) into probabilities for each class.  
Log loss measures how well these probabilities match the true class labels.

---

## What is the Softmax Function?

Softmax is a function that takes a vector of numbers (logits) and “squashes” them into probabilities that sum to 1.

- **Use case:** Final layer of a neural network for multiclass classification.
- **Why?** It helps interpret model outputs as probabilities—so you can say “the model is 80% sure this is a cat, 15% dog, 5% rabbit.”

### Softmax Formula

Given a vector of logits \( z = [z_1, z_2, ..., z_K] \), softmax outputs \( p_i \):

\[
p_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
\]

- \( p_i \) is the predicted probability of class \( i \).
- All \( p_i \) values are between 0 and 1, and sum to 1.

---

### Softmax in Practice

**Example:**

Suppose a model outputs logits `[2.0, 1.0, 0.1]` for classes “cat”, “dog”, “rabbit”.

**Code:**

```python
import numpy as np

logits = np.array([2.0, 1.0, 0.1])
exps = np.exp(logits)
probs = exps / np.sum(exps)
print("Probabilities:", probs)
# Output: [0.659, 0.242, 0.099]  (sums to 1)
````

---

## Visual: Softmax Function

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/softmax-output-diagram.png" alt="Softmax Output Diagram" width="480"/>
</p>

*Shows how raw logits are transformed into class probabilities by softmax.*

---

## Why Softmax?

* Makes outputs comparable as probabilities.
* Useful for multiclass problems where only one class is correct per example (e.g., image classification: cat, dog, rabbit).
* Allows calculation of log loss.

---

## What is Log Loss (Cross-Entropy Loss)?

**Log loss** measures the “distance” between the predicted probability distribution (from softmax) and the true distribution (the correct class label).

* Lower log loss = better model.
* High log loss means the model is confident but often wrong (bad for classification).

### Log Loss Formula

For one example with K classes:

$$
\text{Log Loss} = -\sum_{i=1}^{K} y_i \log(p_i)
$$

* $y_i$ is 1 if class $i$ is correct, 0 otherwise.
* $p_i$ is the predicted probability for class $i$.

**For a batch of examples:** Take the average log loss over all examples.

---

### Example Calculation

Suppose the true class is “dog” (index 1):
Softmax outputs: `[0.659, 0.242, 0.099]`

Log loss for this example:

$$
\text{Log Loss} = -\log(0.242) \approx 1.42
$$

---

### Code Example: Softmax + Log Loss

```python
import numpy as np

def softmax(x):
    exps = np.exp(x)
    return exps / np.sum(exps)

def log_loss(y_true, y_pred_probs):
    # y_true: index of true class
    return -np.log(y_pred_probs[y_true])

logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)
y_true = 1  # (e.g., class "dog")
loss = log_loss(y_true, probs)
print("Predicted probabilities:", probs)
print("Log loss:", loss)
```

---

## Why Log Loss is Used

* Penalizes confident but wrong predictions harshly.
* Encourages the model to output probabilities that match the true likelihood of each class.
* Used to train classifiers—especially neural networks—via gradient descent.

---

## Visual: Log Loss Behavior

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/log-loss-curve.png" alt="Log Loss Curve" width="480"/>
</p>

*Shows how log loss increases sharply as predicted probability for the true class decreases.*

---

## Best Practices

* Always use softmax in the last layer of a neural network for multiclass classification.
* Use log loss (cross-entropy) as the objective/loss function for training.
* For binary classification, use a sigmoid instead of softmax (but log loss still applies).

---

## Further Reading & References

* [Stanford CS231n: Softmax and Cross-Entropy](https://cs231n.github.io/linear-classify/#softmax)
* [DeepLearning.ai: Softmax Explained](https://www.deeplearning.ai/resources/softmax-in-machine-learning/)
* [Wikipedia: Softmax Function](https://en.wikipedia.org/wiki/Softmax_function)
* [Wikipedia: Cross Entropy](https://en.wikipedia.org/wiki/Cross_entropy)
* [Analytics Vidhya: Cross-Entropy Loss](https://www.analyticsvidhya.com/blog/2021/03/softmax-function-and-cross-entropy-loss/)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>