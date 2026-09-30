"""Offline evaluation: RMSE and Precision@K on a held-out test split."""
import numpy as np
import pandas as pd

from .collaborative import CollaborativeRecommender


def train_test_split(ratings, test_frac=0.2, seed=42):
    """Hold out test_frac of each user's ratings (at least 1 stays in train)."""
    rng = np.random.default_rng(seed)
    test_idx = []
    for _, grp in ratings.groupby("user_id"):
        if len(grp) < 3:
            continue
        n_test = max(1, int(len(grp) * test_frac))
        test_idx.extend(rng.choice(grp.index, size=n_test, replace=False))
    test = ratings.loc[test_idx]
    return ratings.drop(test.index).reset_index(drop=True), test.reset_index(drop=True)


def evaluate(ratings, k=5, threshold=4, seed=42):
    """Return {'rmse', 'mae', 'baseline_rmse', 'precision_at_k'}."""
    train, test = train_test_split(ratings, seed=seed)
    cf = CollaborativeRecommender(train)
    train_by_user = {u: dict(zip(g["movie_id"], g["rating"])) for u, g in train.groupby("user_id")}

    preds = [
        cf.predict(train_by_user.get(u, {}), m)
        for u, m in zip(test["user_id"], test["movie_id"])
    ]
    err = np.array(preds) - test["rating"].values
    baseline = test["rating"].values - train["rating"].mean()

    hits, total = 0, 0
    for u, grp in test.groupby("user_id"):
        ur = train_by_user.get(u, {})
        if not ur:
            continue
        scores = cf.scores_for_user(ur)
        top = sorted(scores, key=scores.get, reverse=True)[:k]
        relevant = set(grp.loc[grp["rating"] >= threshold, "movie_id"])
        hits += len(set(top) & relevant)
        total += k

    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "baseline_rmse": float(np.sqrt(np.mean(baseline**2))),
        "precision_at_k": hits / total if total else 0.0,
    }
