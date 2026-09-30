"""Content-based filtering using TF-IDF over genres and keywords."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentRecommender:
    def __init__(self, movies):
        self.movies = movies.reset_index(drop=True)
        # Genres are repeated so they carry more weight than keywords.
        text = (
            self.movies["genres"].str.replace("|", " ", regex=False).str.replace("-", "")
            + " "
            + self.movies["genres"].str.replace("|", " ", regex=False).str.replace("-", "")
            + " "
            + self.movies["keywords"]
        )
        self.matrix = TfidfVectorizer(stop_words="english").fit_transform(text)
        self.sim = cosine_similarity(self.matrix)
        self._idx = {mid: i for i, mid in enumerate(self.movies["movie_id"])}

    def similar_to(self, movie_id, n=5):
        """Top-n movies most similar to the given movie."""
        if movie_id not in self._idx:
            raise KeyError(f"Unknown movie_id {movie_id}")
        i = self._idx[movie_id]
        order = np.argsort(-self.sim[i])
        result = [
            (int(self.movies.iloc[j]["movie_id"]), float(self.sim[i, j]))
            for j in order
            if j != i
        ]
        return result[:n]

    def scores_for_user(self, user_ratings):
        """Score every movie against a user profile.

        user_ratings: dict {movie_id: rating}. Movies the user liked (>= 3)
        contribute their similarity, weighted by (rating - 2.5).
        Returns dict {movie_id: score}.
        """
        scores = np.zeros(len(self.movies))
        for mid, r in user_ratings.items():
            if mid in self._idx:
                scores += (r - 2.5) * self.sim[self._idx[mid]]
        return {int(m): float(s) for m, s in zip(self.movies["movie_id"], scores)}
