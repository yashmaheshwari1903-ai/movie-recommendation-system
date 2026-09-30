"""Item-based collaborative filtering with cosine similarity."""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeRecommender:
    def __init__(self, ratings, k=20):
        self.k = k
        self.matrix = ratings.pivot_table(
            index="user_id", columns="movie_id", values="rating"
        )
        self.global_mean = float(ratings["rating"].mean())
        # Mean-centre each user's ratings so generous/harsh raters compare fairly.
        centred = self.matrix.sub(self.matrix.mean(axis=1), axis=0).fillna(0.0)
        self.item_sim = pd.DataFrame(
            cosine_similarity(centred.T.values),
            index=self.matrix.columns,
            columns=self.matrix.columns,
        )

    def predict(self, user_ratings, movie_id):
        """Predict the rating a user would give to movie_id."""
        if movie_id not in self.item_sim.index or not user_ratings:
            return self.global_mean
        rated = [m for m in user_ratings if m in self.item_sim.index and m != movie_id]
        if not rated:
            return self.global_mean
        sims = self.item_sim.loc[movie_id, rated]
        top = sims[sims > 0].nlargest(self.k)
        if top.empty:
            return float(np.mean(list(user_ratings.values())))
        vals = np.array([user_ratings[m] for m in top.index])
        return float(np.clip(np.dot(top.values, vals) / top.values.sum(), 1, 5))

    def scores_for_user(self, user_ratings):
        """Predicted rating for every movie the user has NOT rated."""
        return {
            int(m): self.predict(user_ratings, m)
            for m in self.matrix.columns
            if m not in user_ratings
        }
