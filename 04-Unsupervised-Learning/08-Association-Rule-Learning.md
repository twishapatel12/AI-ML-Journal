![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# Association Rule Learning

---

## Introduction

**Association Rule Learning** is an unsupervised learning technique used to **find interesting relationships (patterns, correlations, or associations)** between variables in large datasets.  

It’s widely used in **market basket analysis** to discover which products are frequently bought together.

---

## Key Terms

- **Itemset**: A collection of one or more items (e.g., `{milk, bread}`).
- **Transaction**: A set of items purchased together in one event.
- **Support**: Frequency of occurrence of an itemset in the dataset.

$$
\text{Support}(A) = \frac{\text{Number of transactions containing } A}{\text{Total number of transactions}}
$$

- **Confidence**: Likelihood that item B is purchased when item A is purchased.

$$
\text{Confidence}(A \rightarrow B) = \frac{\text{Support}(A \cup B)}{\text{Support}(A)}
$$

- **Lift**: How much more likely A and B occur together than if they were independent.

$$
\text{Lift}(A \rightarrow B) = \frac{\text{Confidence}(A \rightarrow B)}{\text{Support}(B)}
$$

  - Lift > 1: Positive correlation  
  - Lift = 1: Independent  
  - Lift < 1: Negative correlation

---

## Popular Algorithms

### 1. Apriori Algorithm
- Works by identifying **frequent itemsets** and building rules from them.
- Uses the property: *If an itemset is frequent, all of its subsets must also be frequent*.
- Prunes search space to improve efficiency.

---

### 2. FP-Growth Algorithm
- Uses a **Frequent Pattern Tree** structure.
- Faster than Apriori for large datasets because it avoids generating all candidate itemsets.

---

## Example: Market Basket Analysis

If:
- Support({milk}) = 40%
- Support({milk, bread}) = 20%

Then:
- **Confidence**({milk} → {bread}) = 20% / 40% = 50%
- **Lift** = 0.5 / Support({bread})

---

## Code Example: Association Rules in Python

```python
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Example dataset: transactions
dataset = [
    ['milk', 'bread', 'eggs'],
    ['bread', 'butter'],
    ['milk', 'bread', 'butter', 'eggs'],
    ['bread', 'butter'],
    ['milk', 'bread', 'butter']
]

# Convert to one-hot encoded DataFrame
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_array = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_array, columns=te.columns_)

# Apply Apriori
frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)

# Generate rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
````

---

## Visual: Association Rule Learning Concept

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/association-rules-concept.png" alt="Association Rules Concept Diagram" width="500"/>
</p>

*Shows how frequent itemsets lead to rule generation, evaluated by support, confidence, and lift.*

---

## Advantages and Limitations

**Advantages:**

* Easy to understand and implement.
* Works well for discovering interesting patterns in transactions.
* Applicable in many fields beyond retail (e.g., healthcare, web usage mining).

**Limitations:**

* Can generate a very large number of rules (need filtering).
* Not suitable for predicting exact quantities or continuous outcomes.
* Computationally expensive for very large datasets without optimization.

---

## Best Practices

* Use **min\_support** and **min\_confidence** thresholds to reduce the number of unimportant rules.
* Interpret **lift** to ensure rules are meaningful.
* Combine with domain knowledge to filter out irrelevant patterns.

---

## Real-World Applications

* **Retail**: Product bundling, cross-selling strategies.
* **Healthcare**: Finding symptom-disease correlations.
* **Web Analytics**: Discovering common click paths.
* **Banking**: Identifying patterns in fraudulent transactions.

---

## References

* [mlxtend: Frequent Pattern Mining](http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/)
* [Wikipedia: Association Rule Learning](https://en.wikipedia.org/wiki/Association_rule_learning)
* [Agrawal et al., 1993: Original Apriori Paper](https://rakesh.agrawal-family.com/papers/vldb93apriori.pdf)
* [FP-Growth Paper (Han et al., 2000)](https://ieeexplore.ieee.org/document/892129)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>