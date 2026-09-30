"""Command-line interface.

Examples:
  python -m movie_recommender recommend --user 1 --n 5
  python -m movie_recommender similar --title "Inception"
  python -m movie_recommender popular
  python -m movie_recommender evaluate
  python -m movie_recommender interactive
"""
import argparse
import sys

from .data_loader import find_movie_id, load_data
from .evaluate import evaluate
from .hybrid import HybridRecommender


def _print(rows, label):
    for i, (_, title, score) in enumerate(rows, 1):
        print(f"  {i}. {title}  ({label}: {score})")


def _interactive(rec, movies):
    print("Rate movies to get recommendations. Type 'done' when finished.\n")
    print("Sample titles:", ", ".join(movies["title"].sample(8, random_state=1)))
    ratings = {}
    while True:
        title = input("\nMovie title (or 'done'): ").strip()
        if title.lower() == "done":
            break
        mid = find_movie_id(movies, title)
        if mid is None:
            print("  Not found, try another title.")
            continue
        try:
            r = float(input("  Your rating (1-5): "))
        except ValueError:
            print("  Please enter a number.")
            continue
        if not 1 <= r <= 5:
            print("  Rating must be between 1 and 5.")
            continue
        ratings[mid] = r
    if not ratings:
        print("No ratings given.")
        return
    print("\nRecommended for you:")
    _print(rec.recommend_from_ratings(ratings, 5), "score")


def build_parser():
    p = argparse.ArgumentParser(prog="movie_recommender", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", help="folder containing movies.csv and ratings.csv")
    p.add_argument("--alpha", type=float, default=0.6,
                   help="weight for collaborative score, 0-1 (default 0.6)")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("recommend", help="recommend movies for an existing user")
    r.add_argument("--user", type=int, required=True)
    r.add_argument("--n", type=int, default=5)

    s = sub.add_parser("similar", help="movies similar to a title")
    s.add_argument("--title", required=True)
    s.add_argument("--n", type=int, default=5)

    pop = sub.add_parser("popular", help="highest-rated movies")
    pop.add_argument("--n", type=int, default=5)

    sub.add_parser("evaluate", help="RMSE / Precision@5 on a held-out split")
    sub.add_parser("interactive", help="rate movies and get recommendations")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    movies, ratings = load_data(args.data_dir)
    rec = HybridRecommender(movies, ratings, alpha=args.alpha)

    if args.command == "recommend":
        try:
            rows = rec.recommend(args.user, args.n)
        except KeyError as e:
            print(f"Error: {e.args[0]}", file=sys.stderr)
            return 1
        print(f"Top {args.n} recommendations for user {args.user}:")
        _print(rows, "score")
    elif args.command == "similar":
        mid = find_movie_id(movies, args.title)
        if mid is None:
            print(f"Error: no movie matching '{args.title}'", file=sys.stderr)
            return 1
        print(f"Movies similar to '{args.title}':")
        _print(rec.similar_movies(mid, args.n), "similarity")
    elif args.command == "popular":
        print("Most popular movies:")
        _print(rec.popular(args.n), "avg rating")
    elif args.command == "evaluate":
        m = evaluate(ratings)
        print(f"RMSE (item-based CF):   {m['rmse']:.3f}")
        print(f"MAE  (item-based CF):   {m['mae']:.3f}")
        print(f"RMSE (global-mean base): {m['baseline_rmse']:.3f}")
        print(f"Precision@5:            {m['precision_at_k']:.3f}")
    elif args.command == "interactive":
        _interactive(rec, movies)
    return 0


if __name__ == "__main__":
    sys.exit(main())
