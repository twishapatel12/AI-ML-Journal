![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# Gaussian Mixture Models (GMM)

---

## Introduction

A **Gaussian Mixture Model (GMM)** is a **probabilistic model** that assumes data points are generated from a mixture of several Gaussian distributions (also called **normal distributions**) with unknown parameters.

GMM is a **soft clustering** method:
- Each point is assigned a **probability** of belonging to each cluster (rather than being assigned to exactly one cluster as in K-Means).
- This makes it more flexible for overlapping clusters and clusters of different shapes and sizes.

---

## Why Use GMM Instead of K-Means?

- **K-Means** assumes clusters are spherical and equally sized.
- **GMM** allows clusters to have different shapes, sizes, and orientations.
- K-Means assigns points to exactly one cluster (**hard clustering**), whereas GMM gives probabilities (**soft clustering**).

---

## Mathematical Formulation

The probability density function of a GMM with $K$ components is:

$$
p(x) = \sum_{k=1}^K \pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k)
$$

Where:
- $\pi_k$: weight (mixing coefficient) of the $k$-th Gaussian ($\sum_{k=1}^K \pi_k = 1$)
- $\mu_k$: mean vector of the $k$-th Gaussian
- $\Sigma_k$: covariance matrix of the $k$-th Gaussian
- $\mathcal{N}$: Gaussian probability density function

---

## How GMM Learns: The Expectation-Maximization (EM) Algorithm

1. **Initialization**  
   - Start with initial guesses for means ($\mu_k$), covariances ($\Sigma_k$), and mixing coefficients ($ \pi_k$).

2. **E-step (Expectation)**  
   - Calculate the probability that each data point belongs to each Gaussian component.

3. **M-step (Maximization)**  
   - Update parameters ($\mu_k, \Sigma_k, \pi_k$) to maximize the likelihood given these probabilities.

4. **Repeat** until convergence.

---

## Code Example: GMM in Python

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# Create synthetic data
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

# Fit Gaussian Mixture Model
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X)

# Predict probabilities and cluster assignments
labels = gmm.predict(X)
probs = gmm.predict_proba(X)

# Plot results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30)
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', marker='x', s=200, label='Centroids')
plt.title("Gaussian Mixture Model Clustering")
plt.legend()
plt.show()
````

---

## Choosing the Number of Components

Two common criteria:

* **BIC (Bayesian Information Criterion)**
* **AIC (Akaike Information Criterion)**

Lower values indicate a better fit (with a penalty for complexity).

```python
bic_scores = []
aic_scores = []
for k in range(1, 7):
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(X)
    bic_scores.append(gmm.bic(X))
    aic_scores.append(gmm.aic(X))

plt.plot(range(1, 7), bic_scores, label='BIC')
plt.plot(range(1, 7), aic_scores, label='AIC')
plt.xlabel("Number of Components")
plt.ylabel("Score")
plt.legend()
plt.show()
```

---

## Visual: GMM Concept

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/gmm-concept.png" alt="GMM Concept Diagram" width="500"/>
</p>

*Shows overlapping Gaussian distributions representing different clusters, with probabilities for each point.*

---

## Advantages and Limitations

**Advantages:**

* Can model clusters of different shapes, sizes, and orientations.
* Provides soft assignments (probabilities), which is useful for uncertainty estimation.
* Flexible and probabilistic approach.

**Limitations:**

* Assumes data is generated from Gaussian distributions (may not hold true).
* Sensitive to initialization; may converge to local optima.
* Computationally more expensive than K-Means.

---

## Best Practices

* Standardize data before fitting GMM.
* Try different `covariance_type` options (`full`, `tied`, `diag`, `spherical`).
* Use AIC/BIC to choose the number of components.
* Run with multiple random seeds to avoid local minima.

---

## Real-World Applications

* **Speech recognition**: Modeling acoustic features.
* **Image segmentation**: Grouping pixels by color/texture.
* **Astronomy**: Classifying celestial objects.
* **Finance**: Modeling asset returns with multiple regimes.

---

## References

* [Scikit-learn: Gaussian Mixture Models](https://scikit-learn.org/stable/modules/mixture.html#mixture)
* [Wikipedia: Gaussian Mixture Model](https://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model)
* [Bishop, C. M. - Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/)
* [Original EM Algorithm Paper (Dempster et al., 1977)](https://www.jstor.org/stable/2984875)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>