"""
Unit tests for Recommendation Models (Version 1, Version 2, Version 3, and integrated Recommender).
Verifies recommendation count, self-exclusion, score validity, and cold-start handling.
"""

import pytest
from src.config import (
    V1_MODEL_PATH,
    V2_MODEL_PATH,
    V3_ITEM_MODEL_PATH,
    V3_USER_MODEL_PATH,
    TOP_K,
)
from src.version1.basic_content import BasicContentRecommender
from src.version2.advanced_content import AdvancedContentRecommender
from src.version3.collaborative import CollaborativeRecommender
from src.recommender import MovieRecommenderSystem


@pytest.fixture(scope="module")
def recommender_system():
    system = MovieRecommenderSystem()
    system.load_models(
        v1_path=V1_MODEL_PATH,
        v2_path=V2_MODEL_PATH,
        v3_item_path=V3_ITEM_MODEL_PATH,
        v3_user_path=V3_USER_MODEL_PATH,
    )
    return system


def test_v1_basic_content(recommender_system):
    """Test Version 1 outputs, score bounds, self-removal, and top_k limit."""
    query_movie_id = 1  # Toy Story
    recs = recommender_system.v1.recommend(query_movie_id, top_k=10)

    assert len(recs) == 10
    # Crucial rule: Self-recommendation removal
    rec_ids = [r["movie_id"] for r in recs]
    assert query_movie_id not in rec_ids

    # Similarity scores in [0.0, 1.0]
    for r in recs:
        assert 0.0 <= r["similarity_score"] <= 1.0
        assert "rank" in r
        assert "title" in r
        assert "genres" in r


def test_v2_advanced_content(recommender_system):
    """Test Version 2 outputs, score bounds, self-removal, and top_k limit."""
    query_movie_id = 2571  # The Matrix
    recs = recommender_system.v2.recommend(query_movie_id, top_k=10)

    assert len(recs) == 10
    rec_ids = [r["movie_id"] for r in recs]
    assert query_movie_id not in rec_ids

    for r in recs:
        assert 0.0 <= r["similarity_score"] <= 1.0
        assert "rank" in r
        assert "title" in r
        assert "genres" in r


def test_v3_collaborative_filtering(recommender_system):
    """Test Version 3 Item-Based CF outputs, self-removal, and top_k limit."""
    query_movie_id = 260  # Star Wars: Episode IV
    recs, cold_start_msg = recommender_system.v3.recommend(query_movie_id, top_k=10)

    assert len(recs) == 10
    rec_ids = [r["movie_id"] for r in recs]
    assert query_movie_id not in rec_ids
    assert cold_start_msg is None  # Star Wars has thousands of ratings

    for r in recs:
        assert "collaborative_score" in r
        assert 0.0 <= r["collaborative_score"] <= 1.0


def test_v3_cold_start_handling(recommender_system):
    """Test cold-start handling on movies with very few or no ratings."""
    # Find a movie with < MIN_MOVIE_RATINGS
    item_model = recommender_system.v3.item_model
    rare_movie_id = None
    for mid, count in item_model.movie_rating_counts.items():
        if count < item_model.min_ratings:
            rare_movie_id = mid
            break

    if rare_movie_id is None:
        # Check any movie in movies_df not in rating counts
        for mid in item_model.movie_id_to_idx.keys():
            if mid not in item_model.movie_rating_counts:
                rare_movie_id = mid
                break

    assert rare_movie_id is not None
    recs, cold_start_msg = recommender_system.v3.recommend(rare_movie_id, top_k=10)
    assert len(recs) <= 10
    assert cold_start_msg is not None
    assert "insufficient rating data" in cold_start_msg.lower()
    # Ensure target movie itself is excluded
    assert rare_movie_id not in [r["movie_id"] for r in recs]


def test_integrated_recommendation(recommender_system):
    """Test the full 4-version integrated flow."""
    res = recommender_system.recommend(1, top_k=10)
    assert "movie" in res
    assert "v1" in res
    assert "v2" in res
    assert "v3" in res
    assert "v4" in res

    assert len(res["v1"]) <= 10
    assert len(res["v2"]) <= 10
    assert len(res["v3"]) <= 10
    assert len(res["v4"]) <= 10

    # Self movie excluded across all versions
    for v_key in ["v1", "v2", "v3", "v4"]:
        rec_ids = [r["movie_id"] for r in res[v_key]]
        assert 1 not in rec_ids

    # Formatted terminal string
    formatted_str = recommender_system.format_terminal_output(res)
    assert "VERSION 1" in formatted_str
    assert "VERSION 2" in formatted_str
    assert "VERSION 3" in formatted_str
    assert "VERSION 4" in formatted_str
