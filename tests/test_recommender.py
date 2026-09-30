import pytest

from movie_recommender import (CollaborativeRecommender, ContentRecommender,
                               HybridRecommender, load_data)
from movie_recommender.data_loader import find_movie_id
from movie_recommender.evaluate import evaluate, train_test_split
from movie_recommender.cli import main


@pytest.fixture(scope="module")
def data():
    return load_data()


@pytest.fixture(scope="module")
def hybrid(data):
    return HybridRecommender(*data)


def test_load_data_shapes(data):
    movies, ratings = data
    assert len(movies) == 40
    assert ratings["rating"].between(1, 5).all()


def test_find_movie_id_case_insensitive(data):
    movies, _ = data
    assert find_movie_id(movies, "inception") == 16
    assert find_movie_id(movies, "dark knight") == 15
    assert find_movie_id(movies, "no such film") is None


def test_content_similar_excludes_self(data):
    movies, _ = data
    c = ContentRecommender(movies)
    res = c.similar_to(16, n=5)
    assert len(res) == 5
    assert all(mid != 16 for mid, _ in res)
    assert res == sorted(res, key=lambda x: -x[1])


def test_content_sci_fi_neighbours(data):
    movies, _ = data
    c = ContentRecommender(movies)
    top_ids = [m for m, _ in c.similar_to(18, n=6)]  # The Matrix
    assert 16 in top_ids  # Inception shares Sci-Fi/Action


def test_content_unknown_movie(data):
    movies, _ = data
    with pytest.raises(KeyError):
        ContentRecommender(movies).similar_to(9999)


def test_cf_prediction_in_range(data):
    _, ratings = data
    cf = CollaborativeRecommender(ratings)
    p = cf.predict({1: 5, 2: 4}, 3)
    assert 1 <= p <= 5


def test_cf_cold_start_returns_global_mean(data):
    _, ratings = data
    cf = CollaborativeRecommender(ratings)
    assert cf.predict({}, 3) == pytest.approx(cf.global_mean)


def test_recommend_excludes_seen_movies(hybrid):
    seen = set(hybrid.user_ratings(1))
    recs = hybrid.recommend(1, n=10)
    assert len(recs) == 10
    assert not seen & {m for m, _, _ in recs}


def test_recommend_unknown_user(hybrid):
    with pytest.raises(KeyError):
        hybrid.recommend(99999)


def test_invalid_alpha(data):
    with pytest.raises(ValueError):
        HybridRecommender(*data, alpha=1.5)


def test_new_user_from_ratings(hybrid):
    recs = hybrid.recommend_from_ratings({16: 5, 18: 5, 24: 4}, n=3)
    assert len(recs) == 3
    assert all(m not in (16, 18, 24) for m, _, _ in recs)


def test_popular_sorted(hybrid):
    pop = hybrid.popular(5)
    scores = [s for _, _, s in pop]
    assert scores == sorted(scores, reverse=True)


def test_split_no_overlap(data):
    _, ratings = data
    train, test = train_test_split(ratings)
    keys_train = set(zip(train.user_id, train.movie_id))
    keys_test = set(zip(test.user_id, test.movie_id))
    assert not keys_train & keys_test


def test_cf_beats_baseline(data):
    _, ratings = data
    m = evaluate(ratings)
    assert m["rmse"] < m["baseline_rmse"]
    assert 0 <= m["precision_at_k"] <= 1


def test_cli_recommend(capsys):
    assert main(["recommend", "--user", "1", "--n", "3"]) == 0
    assert "recommendations for user 1" in capsys.readouterr().out


def test_cli_similar_bad_title(capsys):
    assert main(["similar", "--title", "zzzz"]) == 1
