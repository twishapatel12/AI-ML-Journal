![Banner](https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/aiml-banner.png)

# Recommendation Systems: The Basics

---

## Introduction

Recommendation systems are a key technology behind modern platforms like Amazon, Netflix, Spotify, and YouTube.  
They help users discover new products, movies, songs, or friends by making smart, personalized suggestions.

In this guide, we’ll cover how recommendation systems work, their main types, core concepts, and practical code examples.

---

## What is a Recommendation System?

A **recommendation system** is an algorithm that predicts what a user might like or find useful based on their past behavior, preferences, or similarities to other users or items.

- **Why important?**  
  They increase engagement, help users find relevant content, and boost business revenue.

---

## Main Types of Recommendation Systems

### 1. Content-Based Filtering

- **How it works:**  
  Recommends items similar to what the user has liked before, based on item features (e.g., genre, keywords, tags).
- **Example:**  
  If you like action movies, it suggests more action movies.

### 2. Collaborative Filtering

- **How it works:**  
  Recommends items that users with similar tastes enjoyed (user-user or item-item similarity).
- **Example:**  
  If people who like the same books as you also liked a new book, you’ll get it recommended.

### 3. Hybrid Methods

- Combine content-based and collaborative filtering for better results and fewer limitations.

---

## Visual: Recommendation System Types

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/recommendation-types-diagram.png" alt="Types of Recommendation Systems" width="500"/>
</p>

---

## Content-Based Filtering Example

**Key idea:**  
Match item features to user preferences.

**Example:**  
User likes movies with “sci-fi” and “adventure” tags.

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.DataFrame({
    'title': ['Inception', 'The Matrix', 'Toy Story', 'Interstellar'],
    'tags': ['sci-fi action', 'sci-fi adventure', 'animation kids', 'sci-fi space']
})
user_profile = "sci-fi adventure"

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies['tags'] + " " + user_profile)
cos_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
recommend_idx = cos_sim.argsort()[::-1]
print("Recommended:", movies.iloc[recommend_idx]['title'].values)
````

---

## Collaborative Filtering Example

**Key idea:**
Leverage the wisdom of the crowd—users who liked similar items get similar recommendations.

**User-Item Matrix Example:**

|       | Movie A | Movie B | Movie C | Movie D |
| ----- | ------- | ------- | ------- | ------- |
| User1 | 5       | 4       | ?       | 2       |
| User2 | 4       | ?       | 5       | 1       |
| User3 | 2       | 2       | 1       | 5       |
| User4 | ?       | 4       | 4       | 3       |

* The `?` means the user hasn't rated the item. The system tries to predict the missing values.

**Simplified code example (using NearestNeighbors):**

```python
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Rows = users, columns = items, values = ratings
ratings = np.array([
    [5, 4, np.nan, 2],
    [4, np.nan, 5, 1],
    [2, 2, 1, 5],
    [np.nan, 4, 4, 3]
])

# For simplicity, replace nans with 0
filled = np.nan_to_num(ratings)
model = NearestNeighbors(metric='cosine')
model.fit(filled)

# Find similar users to User1
distances, indices = model.kneighbors([filled[0]], n_neighbors=2)
print("Most similar users to User1:", indices)
```

---

## Hybrid Recommendation Systems

Many real-world platforms use **hybrid systems** that combine both content-based and collaborative filtering.
This helps reduce their individual weaknesses and gives more robust recommendations.

---

## Challenges in Recommendation Systems

* **Cold Start Problem:**
  New users or items have little or no data, making it hard to give accurate suggestions.
* **Scalability:**
  Recommending from millions of items/users requires efficient algorithms.
* **Diversity & Bias:**
  Systems may reinforce user “filter bubbles” or miss new trends.

---

## Visual: How Netflix Uses Recommendations

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/netflix-recommendation-architecture.png" alt="Netflix Recommendation Architecture" width="600"/>
</p>

*Netflix uses multiple algorithms—collaborative, content, and even context (time of day, device)—to personalize each row of your homepage.*

---

## Where You See Recommendation Systems

* **E-commerce:** Amazon, Flipkart (product suggestions)
* **Streaming:** Netflix, YouTube, Spotify (movies, videos, songs)
* **Social Media:** Facebook, Twitter, Instagram (news feed, friend suggestions)
* **News & Content:** Google News, Medium, Quora

---

## Further Reading & References

* [Netflix Tech Blog: Personalization](https://netflixtechblog.com/netflix-recommendations-beyond-the-5-stars-part-1-55838468f429)
* [Google AI Blog: Recommender Systems](https://ai.googleblog.com/2021/06/scalable-deep-learning-recommendation.html)
* [Wikipedia: Recommender system](https://en.wikipedia.org/wiki/Recommender_system)
* [Analytics Vidhya: Building Recommendation Engines](https://www.analyticsvidhya.com/blog/2021/06/the-most-comprehensive-guide-to-recommendation-engine-types-techniques-and-real-life-examples/)

---

<p align="center">
  <img src="https://github.com/twishapatel12/AI-ML-Journal/blob/main/assets/twisha-patel-logo.png" alt="Twisha Patel Logo" width="80"/>
</p>
<p align="center">
  Created and maintained by Twisha Patel  
  <br>
  <a href="https://github.com/twishapatel12/AI-ML-Journal">GitHub Repo</a>
</p>
