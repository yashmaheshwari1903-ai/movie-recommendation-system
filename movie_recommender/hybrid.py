"""Hybrid recommender: weighted blend of collaborative and content scores."""
import numpy as np

from .collaborative import CollaborativeRecommender
from .content_based import ContentRecommender


def _minmax(d):
    if not d:
        return d
    lo, hi = min(d.values()), max(d.values())
    if hi == lo:
        return {k: 0.5 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


class HybridRecommender:
    def __init__(self, movies, ratings, alpha=0.6, k=20):
        """alpha = weight of collaborative score (1-alpha goes to content)."""
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = alpha
        self.movies = movies
        self.ratings = ratings
        self.content = ContentRecommender(movies)
        self.cf = CollaborativeRecommender(ratings, k=k)
        self._titles = dict(zip(movies["movie_id"], movies["title"]))

    def user_ratings(self, user_id):
        rows = self.ratings[self.ratings["user_id"] == user_id]
        return dict(zip(rows["movie_id"], rows["rating"]))

    def recommend(self, user_id, n=5):
        """Top-n unseen movies for an existing user: [(movie_id, title, score)]."""
        ur = self.user_ratings(user_id)
        if not ur:
            raise KeyError(f"User {user_id} has no ratings")
        return self.recommend_from_ratings(ur, n)

    def recommend_from_ratings(self, ur, n=5):
        cf = _minmax(self.cf.scores_for_user(ur))
        ct = _minmax(
            {m: s for m, s in self.content.scores_for_user(ur).items() if m not in ur}
        )
        blended = {
            m: self.alpha * cf.get(m, 0.0) + (1 - self.alpha) * ct.get(m, 0.0)
            for m in set(cf) | set(ct)
        }
        top = sorted(blended.items(), key=lambda x: -x[1])[:n]
        return [(int(m), self._titles[m], round(float(s), 3)) for m, s in top]

    def similar_movies(self, movie_id, n=5):
        return [
            (m, self._titles[m], round(s, 3))
            for m, s in self.content.similar_to(movie_id, n)
        ]

    def popular(self, n=5, min_ratings=5):
        g = self.ratings.groupby("movie_id")["rating"].agg(["mean", "count"])
        g = g[g["count"] >= min_ratings].sort_values("mean", ascending=False).head(n)
        return [(int(m), self._titles[m], round(float(r["mean"]), 2)) for m, r in g.iterrows()]
