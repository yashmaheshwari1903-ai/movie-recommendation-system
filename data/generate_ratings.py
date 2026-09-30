"""Generate a reproducible synthetic ratings file (data/ratings.csv).

Each simulated user has a taste profile (favourite genres). Ratings depend on
how well a movie's genres match that profile, plus random noise. Users rate
only ~40% of movies, so the matrix is sparse like real-world data.

Run:  python data/generate_ratings.py
"""
import csv
import random
from pathlib import Path

SEED = 42
N_USERS = 60
HERE = Path(__file__).parent

PROFILES = {
    "bollywood_drama": {"Drama", "Comedy", "Romance", "Sport", "Musical"},
    "thriller_fan": {"Thriller", "Crime", "Drama"},
    "scifi_action": {"Sci-Fi", "Action", "Adventure"},
    "family_animation": {"Animation", "Family", "Fantasy", "Adventure"},
    "mixed": {"Drama", "Thriller", "Sci-Fi", "Comedy"},
}


def main():
    rng = random.Random(SEED)
    with open(HERE / "movies.csv", newline="", encoding="utf-8") as f:
        movies = list(csv.DictReader(f))
    names = list(PROFILES)
    rows = []
    for uid in range(1, N_USERS + 1):
        liked = PROFILES[names[(uid - 1) % len(names)]]
        bias = rng.uniform(-0.4, 0.4)
        for m in movies:
            if rng.random() > 0.40:
                continue
            genres = set(m["genres"].split("|"))
            match = len(genres & liked) / len(genres)
            score = 2.0 + 2.8 * match + bias + rng.gauss(0, 0.5)
            rating = int(min(5, max(1, round(score))))
            rows.append((uid, int(m["movie_id"]), rating))
    with open(HERE / "ratings.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "movie_id", "rating"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} ratings for {N_USERS} users.")


if __name__ == "__main__":
    main()
