"""Load and validate the movies and ratings CSV files."""
from pathlib import Path

import pandas as pd

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "data"


def load_data(data_dir=None):
    """Return (movies_df, ratings_df). Raises ValueError on bad data."""
    data_dir = Path(data_dir) if data_dir else DEFAULT_DIR
    movies = pd.read_csv(data_dir / "movies.csv")
    ratings = pd.read_csv(data_dir / "ratings.csv")

    for col in ("movie_id", "title", "genres", "keywords"):
        if col not in movies.columns:
            raise ValueError(f"movies.csv is missing column '{col}'")
    for col in ("user_id", "movie_id", "rating"):
        if col not in ratings.columns:
            raise ValueError(f"ratings.csv is missing column '{col}'")
    if not ratings["rating"].between(1, 5).all():
        raise ValueError("Ratings must be between 1 and 5")

    movies["keywords"] = movies["keywords"].fillna("")
    ratings = ratings[ratings["movie_id"].isin(movies["movie_id"])]
    return movies.reset_index(drop=True), ratings.reset_index(drop=True)


def find_movie_id(movies, title):
    """Case-insensitive title lookup (exact, then partial). Returns id or None."""
    t = title.strip().lower()
    exact = movies[movies["title"].str.lower() == t]
    if not exact.empty:
        return int(exact.iloc[0]["movie_id"])
    part = movies[movies["title"].str.lower().str.contains(t, regex=False)]
    if not part.empty:
        return int(part.iloc[0]["movie_id"])
    return None
