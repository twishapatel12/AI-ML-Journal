![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# Markov Chains in Machine Learning

---

## Introduction

**Markov chains** are mathematical models used to describe systems that move between different states, where the probability of moving to the next state depends only on the current state—not on how you got there.  
They are fundamental in probability theory and have important applications in machine learning, natural language processing, and even computer vision.

---

## What is a Markov Chain?

A **Markov chain** is a sequence of random events (states), where the probability of moving to the next state depends only on the present state.

- **States:** The different conditions or positions the system can be in.
- **Transitions:** The moves between states, each with a certain probability.

**Markov property:**  
The future is independent of the past, given the present.

---

## Real-World Example: Weather

Suppose the weather can be **Sunny**, **Cloudy**, or **Rainy**.

- If today is Sunny, there is a 60% chance tomorrow is Sunny, 30% Cloudy, 10% Rainy.
- These probabilities depend only on today's weather—not last week’s.

---

## Transition Matrix

A **transition matrix** encodes the probabilities of moving from each state to every other state.

**Example:**

|      | Sunny | Cloudy | Rainy |
|------|-------|--------|-------|
| Sunny  | 0.6   | 0.3    | 0.1   |
| Cloudy | 0.3   | 0.4    | 0.3   |
| Rainy  | 0.2   | 0.5    | 0.3   |

- Each row sums to 1 (total probability for all next states).

---

## Markov Chain Diagram

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/markov-chain-weather.png" alt="Markov Chain Diagram" width="440"/>
</p>

*Arrows show transitions between Sunny, Cloudy, and Rainy, each labeled with probabilities.*

---

## How Does a Markov Chain Evolve?

Given a **starting state**, you can use the transition matrix to predict the likelihood of each possible next state—and continue for many steps into the future.

**Code Example: Simulating a Markov Chain**

```python
import numpy as np

states = ['Sunny', 'Cloudy', 'Rainy']
transition_matrix = np.array([
    [0.6, 0.3, 0.1],
    [0.3, 0.4, 0.3],
    [0.2, 0.5, 0.3]
])

current_state = 0  # Start at Sunny
steps = 10
history = [states[current_state]]

for _ in range(steps):
    current_state = np.random.choice([0, 1, 2], p=transition_matrix[current_state])
    history.append(states[current_state])

print("Weather over 10 days:", history)
````

---

## Stationary Distribution

After many steps, the system may settle into a **stationary distribution**—the long-term probabilities of being in each state, regardless of where you started.

This can be found by solving:

$$
\pi = \pi P
$$

Where:

* $\pi$ is the stationary distribution (row vector).
* $P$ is the transition matrix.

---

## Applications of Markov Chains in ML

* **Text generation and language models:**
  Predict next word or character based only on the current one (n-gram models).
* **Speech recognition and Hidden Markov Models (HMMs):**
  Model sequences of sounds or events.
* **Reinforcement learning:**
  Many environments are modeled as Markov Decision Processes (MDPs).
* **Google PageRank:**
  Web navigation as a Markov process.
* **Computer vision:**
  Model pixel transitions, motion, and more.

---

## Visual: Markov Chain for Text Generation

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/markov-text-gen.png" alt="Markov Chain for Text Generation" width="440"/>
</p>

*Shows transitions between words, illustrating how simple models can create text by following word-to-word probabilities.*

---

## Summary

* Markov chains are models for sequences where the next step depends only on the current step.
* Transition matrices represent the probability of moving between states.
* Used in text, speech, ranking, and many sequence-based ML problems.

---

## Further Reading & References

* [Khan Academy: Introduction to Markov Chains](https://www.khanacademy.org/math/statistics-probability/probability-library)
* [Wikipedia: Markov Chain](https://en.wikipedia.org/wiki/Markov_chain)
* [StatQuest: Markov Chains (YouTube)](https://www.youtube.com/watch?v=uvYTGEZQTEs)
* [Stanford CS229: Markov Models](https://cs229.stanford.edu/notes2022fall/cs229-notes8b.pdf)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>