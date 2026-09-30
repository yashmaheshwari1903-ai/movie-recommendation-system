# Project Description

**Title:** Movie Recommendation System (Hybrid Content-Based and Collaborative Filtering)
**Student:** Raj Verma | **Reg. No.:** 25MIM10221
**Programme:** Integrated M.Tech Artificial Intelligence, VIT Bhopal University
**Platform:** VITyarthi

## Overview

With thousands of films available across streaming platforms, viewers spend more time choosing than watching. This project builds a recommendation engine that suggests movies a user is likely to enjoy, using two complementary techniques combined into one hybrid model. It is written in Python with pandas and scikit-learn, and is used through a command-line interface.

## Problem Statement

Given a catalogue of movies and a history of user ratings, recommend unseen movies that match each user's taste, including for new users who have rated only a few titles.

## Objectives

1. Represent each movie by its genres and keywords and find similar movies using TF-IDF and cosine similarity.
2. Predict user ratings from other users' behaviour using item-based collaborative filtering.
3. Blend both approaches into a hybrid recommender with an adjustable weight.
4. Handle new users through an interactive rate-and-recommend mode.
5. Measure quality objectively with RMSE, MAE and Precision@K.
6. Deliver clean, tested, GitHub-ready code with documentation.

## Approach

- **Data:** 40 popular Bollywood and Hollywood movies and 960 ratings (1-5) from 60 simulated users with distinct genre tastes.
- **Content model:** TF-IDF vectors on genres (double-weighted) and keywords; cosine similarity between movies.
- **Collaborative model:** mean-centred user ratings, item-item cosine similarity, weighted average over the top-k similar rated items.
- **Hybrid model:** min-max normalise both score sets, then blend with `alpha` (default 0.6).
- **Evaluation:** per-user 80/20 split, compared against a global-mean baseline.

## Tools and Technologies

Python 3.10+, pandas, NumPy, scikit-learn, pytest, Git and GitHub Actions (CI).

## Outcomes

- Working CLI with five commands: `recommend`, `similar`, `popular`, `evaluate`, `interactive`.
- Collaborative RMSE of 0.880 against 1.126 for the baseline (about 22% lower error).
- 16 automated tests passing on Python 3.10, 3.11 and 3.12.
- Modular design where each model can be used independently as a library.

## Limitations

The dataset is small and synthetic, and the system uses no demographic or time information. Results should be re-validated on real data such as MovieLens before drawing conclusions about real-world performance.
