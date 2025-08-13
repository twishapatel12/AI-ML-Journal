![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# Independent Component Analysis (ICA)

---

## Introduction

**Independent Component Analysis (ICA)** is an **unsupervised dimensionality reduction** technique used to separate a multivariate signal into **statistically independent components**.  

While **PCA** finds **uncorrelated components** (maximizing variance), ICA goes further by finding **independent** components, which can capture **non-Gaussian structure** in the data.

ICA is widely used in:
- **Signal processing** (e.g., separating mixed audio signals)
- **Neuroimaging** (e.g., analyzing fMRI or EEG data)
- **Financial data analysis** (identifying independent market factors)

---

## ICA vs. PCA

| Feature            | PCA                                  | ICA                                          |
|--------------------|--------------------------------------|----------------------------------------------|
| Goal               | Maximize variance                    | Maximize statistical independence           |
| Components         | Uncorrelated                         | Independent                                  |
| Distribution       | Works best with Gaussian data        | Works best with non-Gaussian data            |
| Output             | Orthogonal components                | Non-orthogonal components                    |
| Common Use Case    | Data compression, noise reduction    | Blind source separation (e.g., “cocktail party problem”) |

---

## The Cocktail Party Problem

Imagine recording two people speaking simultaneously using two microphones. Each microphone captures a **mixture** of the two voices.  
ICA can separate the recordings into **two independent speech signals** without knowing the mixing process.

---

## Mathematical Formulation

We assume:

$$
X = AS
$$

Where:
- $X$ = observed data matrix (mixtures)
- $A$ = unknown mixing matrix
- $S$ = source signals (independent components)

Goal: Estimate both $A$ and $S$ given only $X$, under the assumption that the components of $S$ are statistically independent and non-Gaussian.

---

## How ICA Works

1. **Centering and Whitening**  
   - Remove the mean and decorrelate variables.
   - Whitening makes covariance matrix the identity.

2. **Maximizing Non-Gaussianity**  
   - Independent signals are more non-Gaussian than mixtures (Central Limit Theorem).
   - Use measures like **kurtosis** or **negentropy** to find directions that maximize non-Gaussianity.

3. **Recover Independent Components**  
   - Iteratively estimate unmixing matrix $W$ so that:

$$
S = WX
$$

---

## Code Example: ICA in Python

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

# Generate synthetic mixed signals
np.random.seed(42)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

s1 = np.sin(2 * time)          # Signal 1 : sine wave
s2 = np.sign(np.sin(3 * time)) # Signal 2 : square wave
s3 = np.random.rand(n_samples) # Signal 3 : noise

S = np.c_[s1, s2, s3]
S /= S.std(axis=0)  # Standardize

# Mix data
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])
X = np.dot(S, A.T)

# Apply ICA
ica = FastICA(n_components=3, random_state=42)
S_ica = ica.fit_transform(X)  # Estimated independent sources

# Plot results
plt.figure(figsize=(9, 6))
models = [X, S, S_ica]
names = ['Mixed signals', 'True sources', 'ICA recovered signals']
for i, (model, name) in enumerate(zip(models, names), start=1):
    plt.subplot(3, 1, i)
    plt.title(name)
    for sig in model.T:
        plt.plot(sig)
plt.tight_layout()
plt.show()
````

---

## Visual: ICA Concept

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/ica-concept.png" alt="ICA Concept Diagram" width="500"/>
</p>

*Shows how mixed signals can be separated into independent sources using ICA.*

---

## Advantages and Limitations

**Advantages:**

* Can separate mixed signals without prior knowledge.
* Works well for non-Gaussian independent sources.
* Useful in many real-world signal separation tasks.

**Limitations:**

* The number of sources must not exceed the number of observations.
* Sensitive to noise.
* Scaling and order of recovered components are indeterminate.

---

## Best Practices

* Standardize/whiten data before ICA.
* Choose the number of components carefully.
* Use **FastICA** for efficient computation.
* Validate independence using statistical measures.

---

## Real-World Applications

* **EEG analysis**: Separating brain activity from eye blinks and noise.
* **Audio processing**: Isolating individual voices or instruments.
* **Image processing**: Extracting independent texture components.
* **Finance**: Identifying independent drivers of asset returns.

---

## References

* [Scikit-learn: FastICA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html)
* [Wikipedia: Independent Component Analysis](https://en.wikipedia.org/wiki/Independent_component_analysis)
* [Hyvärinen & Oja, Independent Component Analysis: Algorithms and Applications (2000)](https://www.cs.helsinki.fi/u/ahyvarin/papers/NN00new.pdf)
* [The Cocktail Party Problem - Stanford CS229 Notes](https://cs229.stanford.edu/notes2021fall/cs229-notes10.pdf)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>