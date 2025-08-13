![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# Latent Dirichlet Allocation (LDA) - Topic Modeling

---

## Introduction

**Latent Dirichlet Allocation (LDA)** is a **generative probabilistic model** used for **topic modeling**—discovering hidden thematic structures in a collection of documents.

Key idea:
- Each document is a **mixture of topics**.
- Each topic is a **distribution over words**.
- LDA tries to infer:
  - Which topics are present in each document
  - Which words are associated with each topic

---

## How LDA Works

1. **Assumptions**:
   - There are `K` topics in total.
   - Each document is a probabilistic combination of these topics.
   - Each topic is a probabilistic distribution over the vocabulary.

2. **Generative Process**:
   For each document:
   1. Randomly choose topic proportions ($\theta_d$) from a Dirichlet distribution.
   2. For each word in the document:
      - Randomly choose a topic from $\theta_d$.
      - Randomly choose a word from the topic’s word distribution ($\phi_k$).

3. **Goal of LDA**:
   - Given the observed words, infer:
     - Topic distribution for each document ($\theta_d$)
     - Word distribution for each topic ($\phi_k$)
     - Topic assignment for each word

---

## LDA Graphical Model

$$
\alpha, \beta \quad \text{(Dirichlet priors)}
$$

$$
\theta_d \sim \text{Dir}(\alpha), \quad \phi_k \sim \text{Dir}(\beta)
$$

$$
z_{dn} \sim \text{Multinomial}(\theta_d), \quad w_{dn} \sim \text{Multinomial}(\phi_{z_{dn}})
$$

Where:
- $d$ = document index
- $n$ = word position in document
- $z_{dn}$ = topic assignment for word $n$ in document $d$

---

## Code Example: LDA in Python (Gensim)

```python
import gensim
import gensim.corpora as corpora
from gensim.models import LdaModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Sample documents
documents = [
    "Machine learning is amazing for text classification",
    "Deep learning advances computer vision and NLP",
    "Topic modeling extracts hidden themes from documents",
    "Natural language processing is part of AI",
    "AI and ML are revolutionizing many industries"
]

# Preprocess text
stop_words = set(stopwords.words('english'))
texts = [[word for word in word_tokenize(doc.lower()) if word.isalpha() and word not in stop_words]
         for doc in documents]

# Create dictionary and corpus
id2word = corpora.Dictionary(texts)
corpus = [id2word.doc2bow(text) for text in texts]

# Build LDA model
lda_model = LdaModel(corpus=corpus, id2word=id2word, num_topics=2, random_state=42, passes=10)

# Print topics
for idx, topic in lda_model.print_topics():
    print(f"Topic {idx}: {topic}")
````

---

## Choosing the Number of Topics (K)

* **Coherence Score**: Measures semantic similarity between top words in topics.
* **Perplexity**: Lower values indicate better generalization (but less interpretable than coherence).
* Use domain knowledge to guide K.

---

## Visual: LDA Concept

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/lda-concept.png" alt="LDA Concept Diagram" width="500"/>
</p>

*Shows documents containing multiple topics, with topics being distributions over words.*

---

## Advantages and Limitations

**Advantages:**

* Unsupervised—no labeled data needed.
* Provides interpretable topics.
* Handles large document collections.

**Limitations:**

* Bag-of-words assumption ignores word order.
* Needs predefined number of topics.
* Can produce incoherent topics if parameters are not tuned.

---

## Best Practices

* Remove stop words and rare words before training.
* Use **lemmatization** to group word forms.
* Tune hyperparameters:

  * `alpha`: Document-topic density
  * `beta`: Topic-word density
* Evaluate with coherence and human judgment.

---

## Real-World Applications

* News categorization.
* Customer feedback analysis.
* Academic research trend discovery.
* Legal document classification.
* Social media trend analysis.

---

## References

* [Gensim: LDA Model](https://radimrehurek.com/gensim/models/ldamodel.html)
* [Blei, Ng, Jordan (2003) - LDA Original Paper](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)
* [Scikit-learn: Topic Extraction with LDA](https://scikit-learn.org/stable/modules/decomposition.html#latentdirichletallocation)
* [Wikipedia: Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>