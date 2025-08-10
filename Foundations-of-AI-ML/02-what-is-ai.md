![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# What is Artificial Intelligence?

---

## Introduction

Artificial Intelligence (AI) is a branch of computer science concerned with building smart machines capable of performing tasks that typically require human intelligence.  
AI systems are designed to reason, learn, perceive, communicate, and make decisions.

The term "Artificial Intelligence" was first introduced by John McCarthy in 1956 at the Dartmouth Conference. Since then, the field has evolved from simple rule-based systems to modern machine learning, deep learning, and generative models.

---

## Definition

**Artificial Intelligence** refers to the simulation of human intelligence in machines, allowing them to perform tasks such as learning, reasoning, problem-solving, perception, and language understanding.

---

## Types of Artificial Intelligence

| Category              | Description                                               | Example                 |
|-----------------------|-----------------------------------------------------------|-------------------------|
| Narrow AI (Weak AI)   | Specialized in one task; cannot generalize.               | Siri, Google Maps, Chess AI  |
| General AI (Strong AI)| Human-level, can understand or learn any task.            | *Theoretical only*      |
| Superintelligent AI   | Surpasses human intelligence across all fields.           | *Theoretical/future*    |

---

## Key Approaches in AI

- **Rule-Based Systems**: If-then logic (used in early AI and expert systems).
- **Search and Optimization**: Pathfinding, decision trees.
- **Machine Learning**: Learning patterns from data (supervised, unsupervised, reinforcement learning).
- **Neural Networks / Deep Learning**: Multi-layered algorithms inspired by the human brain.

---

## Real-World Examples & How They Work

### 1. AI in Everyday Life

| Application                  | How AI Works (Technical)                                                    |
|------------------------------|-----------------------------------------------------------------------------|
| Google Maps & Navigation     | Uses search algorithms, real-time data, and shortest-path optimization.     |
| Voice Assistants (Alexa/Siri)| Speech recognition, natural language processing, and intent classification. |
| Email Spam Filters           | Rule-based and ML classifiers (e.g., Naive Bayes, SVM).                    |
| Image Recognition            | Convolutional Neural Networks (CNNs) for object detection/classification.   |
| Recommendation Engines       | Collaborative filtering, matrix factorization, deep learning.               |

---

### 2. Detailed Example: Email Spam Detection

AI-powered spam filters combine rule-based systems (e.g., block emails with certain words) and machine learning (e.g., learn patterns in spam content):

```python
# Simple spam detection with rule-based logic
def is_spam_rule_based(email_text):
    spam_keywords = ['lottery', 'free money', 'click here']
    return any(keyword in email_text.lower() for keyword in spam_keywords)

# Modern spam detection (conceptual, using ML model)
def is_spam_ml(email_text, ml_model):
    features = extract_features(email_text)
    prediction = ml_model.predict(features)
    return prediction == 'spam'
````

*In practice, spam filters use a mix of both methods for higher accuracy.*

---

## Visual: How AI Systems Work

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/ai-workflow-diagram.png" alt="How AI Works Diagram" width="420">
</p>

**Figure:**
A simplified AI workflow. Data (images, text, signals) is collected, preprocessed, and then used by an AI algorithm to make decisions or predictions. In modern AI, this often involves training a machine learning model with labeled data, followed by real-time prediction.

---

## Real-World AI: Case Study - Self-Driving Cars

* **Perception:** Uses computer vision to detect lanes, vehicles, pedestrians (via CNNs and LIDAR sensor fusion).
* **Planning:** Predicts the future position of objects and plans routes using search algorithms (A\*, Dijkstra).
* **Decision Making:** Chooses actions based on real-time data (e.g., when to stop or accelerate).
* **Control:** Executes the chosen actions by controlling the car’s hardware.

Self-driving cars combine multiple branches of AI: perception (deep learning), reasoning (search and optimization), and control theory.

---

## How is AI Different from Machine Learning?

* **AI** is the goal: building systems that act intelligently.
* **Machine Learning** is a subset: a way for machines to learn from data.
* **Deep Learning** is a further subset: using large neural networks for tasks like vision and language.

See [Day 01: AI vs ML vs DS](./ai-vs-ml-vs-ds.md) for a detailed breakdown.

---

## Academic & Technical References

* [IEEE: What is Artificial Intelligence?](https://spectrum.ieee.org/what-is-artificial-intelligence)
* [MIT CSAIL: AI Explained](https://csail.mit.edu/research/artificial-intelligence)
* [Russell, S., & Norvig, P. (2016). Artificial Intelligence: A Modern Approach. Pearson.](http://aima.cs.berkeley.edu/)
* [IBM: Artificial Intelligence Explained](https://www.ibm.com/cloud/learn/what-is-artificial-intelligence)
* [Stanford AI Research](https://ai.stanford.edu/)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>