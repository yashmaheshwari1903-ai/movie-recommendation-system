# Movie Recommendation System: Project Report

**Student:** Yash maheshwari
**Programme:** Integrated M.Tech Artificial Intelligence, VIT Bhopal University
**Platform:** VITyarthi  **Date:** September 2026

---

## Abstract

This project implements a movie recommendation system that combines content-based filtering and item-based collaborative filtering into a hybrid model. Content similarity is computed from TF-IDF vectors of genres and keywords; collaborative predictions come from cosine similarity between mean-centred item rating vectors. On a held-out test split, the collaborative model achieves an RMSE of 0.880, compared with 1.126 for a global-mean baseline. The system is packaged as a tested Python module with a command-line interface and continuous integration.

## 1. Introduction

### 1.1 Motivation
Streaming services offer far more titles than anyone can browse. Recommender systems reduce this overload and are central to platforms such as Netflix and YouTube. Building one is also a practical way to apply linear algebra, information retrieval and evaluation methodology from the AI curriculum.

### 1.2 Problem statement
Given movies with descriptive metadata and a set of user-movie ratings, produce a ranked list of unseen movies for a user, and support new users who have almost no history.

### 1.3 Objectives
1. Build content-based and collaborative recommenders.
2. Combine them in a hybrid model.
3. Support cold-start users.
4. Evaluate with standard metrics.
5. Provide reproducible, tested, documented code.

## 2. Background

**Content-based filtering** recommends items similar to those a user liked, using item attributes. It works for new items but tends to over-specialise.

**Collaborative filtering** uses patterns in many users' ratings and needs no item metadata. It suffers from the *cold-start problem* for new users or items, and from data sparsity.

**Hybrid systems** blend both to offset each other's weaknesses.

**TF-IDF** weights a term by how often it appears in a document, discounted by how common it is across all documents, so distinctive terms such as "wizard" count more than common ones.

**Cosine similarity** measures the angle between two vectors: `cos(a, b) = (a . b) / (|a| |b|)`, ranging from 0 (unrelated) to 1 (identical direction) for non-negative vectors.

## 3. System Design

### 3.1 Architecture

```
 movies.csv ──► ContentRecommender (TF-IDF + cosine) ──┐
                                                        ├─► HybridRecommender ─► CLI
 ratings.csv ─► CollaborativeRecommender (item-item) ──┘          │
                                                             evaluate.py
```

### 3.2 Modules

| Module | Responsibility |
|---|---|
| `data_loader.py` | Read CSVs, validate columns and rating range, title lookup |
| `content_based.py` | TF-IDF matrix, movie-to-movie similarity, user-profile scoring |
| `collaborative.py` | User-item matrix, mean-centring, item similarity, rating prediction |
| `hybrid.py` | Normalisation, weighted blend, popularity ranking |
| `evaluate.py` | Train/test split, RMSE, MAE, Precision@K |
| `cli.py` | Commands and interactive mode |

### 3.3 Dataset
- `movies.csv`: 40 movies (Bollywood and Hollywood) with year, pipe-separated genres and keywords.
- `ratings.csv`: 960 ratings by 60 users on a 1-5 scale. The matrix is 40% filled (960 of 2,400 possible ratings).
- Ratings are **synthetic**: five taste profiles (Bollywood drama, thriller fan, sci-fi/action, family/animation, mixed) are assigned to users in rotation. A rating is `2.0 + 2.8 x (genre match) + user bias + Gaussian noise`, rounded and clipped to 1-5, using seed 42 for reproducibility.

## 4. Methodology

### 4.1 Content-based model
For each movie, text = genres (repeated twice) + keywords. A `TfidfVectorizer` with English stop-words produces the feature matrix, and pairwise cosine similarity gives a 40x40 similarity matrix. For a user profile, every movie's score is the sum of similarities to the user's rated movies weighted by `(rating - 2.5)`, so liked movies pull similar items up and disliked ones push them down.

### 4.2 Collaborative model
1. Build a user-item matrix.
2. Subtract each user's mean rating (removes generous/harsh bias), fill missing with 0.
3. Compute item-item cosine similarity.
4. To predict user *u* on item *i*, take the *k* = 20 most similar items that *u* rated (positive similarity only) and compute
   `r_hat(u,i) = sum(sim(i,j) * r(u,j)) / sum(sim(i,j))`,
   clipped to [1, 5].
5. Fallbacks: global mean if the user has no history; the user's own mean if no positively similar item exists.

### 4.3 Hybrid model
Both score sets are min-max scaled to [0, 1] and combined:
`score = alpha * CF + (1 - alpha) * Content`, with default `alpha = 0.6`. Movies the user has already rated are excluded.

### 4.4 Evaluation protocol
Per user, 20% of ratings (at least one) are held out; users with fewer than 3 ratings are skipped. Models are fitted on the remainder.
- **RMSE / MAE** on held-out ratings, versus a baseline predicting the training mean.
- **Precision@5:** fraction of the top-5 recommended movies that appear in the user's held-out set with rating >= 4.

## 5. Implementation

- **Language and libraries:** Python 3.10+, pandas, NumPy, scikit-learn.
- **Interface:** `argparse` CLI (`recommend`, `similar`, `popular`, `evaluate`, `interactive`) with input validation and non-zero exit codes on errors.
- **Testing:** 16 pytest tests covering data validation, title lookup, similarity ordering and exclusion of self, prediction range, cold-start fallback, exclusion of seen movies, invalid inputs, split integrity, CF-beats-baseline and CLI behaviour.
- **CI:** GitHub Actions runs the suite on Python 3.10, 3.11 and 3.12 for every push and pull request.

## 6. Results

### 6.1 Metrics

| Metric | Item-based CF | Global-mean baseline |
|---|---|---|
| RMSE | 0.880 | 1.126 |
| MAE | 0.714 | n/a |
| Precision@5 | 0.067 | n/a |

The collaborative model reduces RMSE by roughly 22%, showing that other users' ratings carry real signal.

### 6.2 Sample outputs

Recommendations for user 1 (a Bollywood-drama profile): *Dil Chahta Hai, Barfi, Forrest Gump, Titanic, Dangal*. All are drama, comedy or romance titles, matching the profile.

Movies similar to *Inception*: *The Matrix, Iron Man, Avengers: Endgame, The Prestige, The Dark Knight*. These are sci-fi, action and thriller titles, which is sensible.

### 6.3 Discussion
Precision@5 is low mainly because of the evaluation setup: each user has about 16 ratings, so only around 3 are held out and typically about one of them is rated 4 or higher. Even a perfect model can find at most a handful of hits, and the top-5 list is drawn from all unseen movies, including many the user never rated. Precision@K is therefore best read as a relative metric for comparing models on the same split, not as an absolute quality score. RMSE is the more informative number here.

Recommendations are also strongly shaped by genre because the synthetic data was built from genre profiles. This makes the results tidy but easier than real data.

## 7. Limitations

1. Synthetic ratings and only 40 movies; performance on real data may differ.
2. Cold-start users must rate a few movies first.
3. Content features are limited to genres and keywords; no cast, director or plot text.
4. No hyper-parameter search for `alpha` or `k`; defaults were chosen by reasoning, not tuning.
5. Single train/test split with one seed; no cross-validation or confidence intervals.

## 8. Future Work

- Use the MovieLens dataset and tune `alpha` and `k` with cross-validation.
- Add matrix factorisation (SVD/ALS) and compare against the current models.
- Add diversity and novelty metrics alongside accuracy.
- Build a Streamlit or Flask front end and fetch posters via an API.

## 9. Conclusion

The project delivers a working hybrid movie recommender that outperforms a naive baseline on rating prediction, handles new users through an interactive mode, and is packaged with tests, CI and documentation for easy reuse. It demonstrates the core ideas behind real-world recommender systems in a small, understandable codebase.

## 10. References

1. Ricci, F., Rokach, L., Shapira, B. (eds.), *Recommender Systems Handbook*, Springer.
2. Sarwar, B. et al., "Item-based collaborative filtering recommendation algorithms," *WWW*, 2001.
3. Harper, F. M., Konstan, J. A., "The MovieLens Datasets: History and Context," *ACM TiiS*, 2015.
4. scikit-learn documentation: https://scikit-learn.org
5. pandas documentation: https://pandas.pydata.org
