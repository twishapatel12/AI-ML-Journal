![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# What is Machine Learning?

---

## Introduction

Machine Learning (ML) is a subset of artificial intelligence focused on building systems that can learn from data, identify patterns, and make decisions with minimal human intervention. Instead of being explicitly programmed for each task, machine learning models infer rules and relationships directly from examples.

---

## What is Machine Learning?

Machine learning enables computers to “learn” from past data and improve their performance over time. This means that, rather than relying only on hand-crafted rules, a machine learning model discovers structure in data and uses it to make predictions or decisions.

---

## How Is ML Different from Traditional Programming?

- **Traditional programming:** Humans write the explicit rules.  
- **Machine learning:** The system learns rules and patterns from data.

For example, a traditional spam filter might block emails containing certain keywords.  
A machine learning spam filter learns subtle (and changing) signals from thousands of real examples, spotting spam even when the keywords change.

---

## Types of Machine Learning

| Type           | Description                         | Example                               |
|----------------|-------------------------------------|---------------------------------------|
| Supervised     | Learns from labeled data            | Email spam detection, price prediction|
| Unsupervised   | Finds patterns in unlabeled data    | Customer segmentation, clustering     |
| Reinforcement  | Learns by trial and error           | Game-playing agents, robotics         |

---

## Typical Machine Learning Workflow

1. **Data Collection:** Gather raw data (images, text, numbers, etc.)
2. **Preprocessing:** Clean and transform the data (handling missing values, normalization, etc.)
3. **Model Selection:** Choose an appropriate algorithm (e.g., decision tree, linear regression, neural network).
4. **Training:** Fit the model to the data.
5. **Evaluation:** Test the model’s performance on unseen data.
6. **Deployment:** Use the trained model in real-world applications.

---

## Minimal Example: Predicting House Prices

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Example data: size (sq.ft.) and price
X = np.array([[1000], [1500], [2000], [2500]])  # Features
y = np.array([200000, 250000, 300000, 350000])  # Labels

model = LinearRegression()
model.fit(X, y)

# Predict price for a 1800 sq.ft. house
predicted_price = model.predict([[1800]])
print(f"Predicted price: ${predicted_price[0]:,.0f}")
````

*This simple example demonstrates supervised learning using linear regression.*

---

## Real-World Applications

* **Product Recommendations:** Netflix, Amazon, and Spotify use ML to personalize what you see and hear.
* **Fraud Detection:** Banks use ML to catch unusual and suspicious activity.
* **Speech and Image Recognition:** ML powers voice assistants and helps classify objects in images.
* **Healthcare:** Diagnosing diseases from medical images or predicting patient risks.
* **Autonomous Vehicles:** Self-driving cars use ML to interpret their environment and make decisions.

---

## Visual: Machine Learning Pipeline

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/ml-workflow-diagram.png" alt="Machine Learning Workflow Diagram" width="420">
</p>

**Figure:**
Data is collected, preprocessed, and used to train a machine learning model, which can then make predictions on new, unseen data.

---

## Machine Learning vs. Artificial Intelligence

* **Artificial Intelligence (AI):** The broader field, including any method to simulate human-like intelligence.
* **Machine Learning (ML):** A key subfield, focused on data-driven learning and prediction.
* **Deep Learning:** A specialized ML approach using large neural networks for vision, speech, and language tasks.

See also: [What is Artificial Intelligence?](https://github.com/twishapatel12/AI-ML-Journal/blob/main/foundations-of-ai-ml/02-what-is-ai.md)

---

## Further Reading and References

* [Stanford: Machine Learning (Course)](https://online.stanford.edu/courses/xine229-machine-learning)
* [IEEE: The Real-World Impact of Machine Learning](https://spectrum.ieee.org/tag/machine-learning)
* [MIT OpenCourseWare: Machine Learning](https://ocw.mit.edu/courses/6-036-introduction-to-machine-learning-fall-2020/)
* [A Visual Introduction to Machine Learning](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/)
* [Scikit-learn: Machine Learning in Python](https://scikit-learn.org/stable/tutorial/basic/tutorial.html)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>