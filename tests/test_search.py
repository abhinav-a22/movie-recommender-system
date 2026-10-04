"""
Unit tests for MovieSearchEngine:
Exact matching, case-insensitivity, partial matching,
article normalization, duplicate resolution, and ID lookup.
"""

import pytest
import pandas as pd
from src.data_loader import load_movies
from src.preprocessing import preprocess_movies
from src.search import MovieSearchEngine, normalize_string


@pytest.fixture(scope="module")
def search_engine():
    movies = load_movies()
    clean_movies = preprocess_movies(movies)
    return MovieSearchEngine(clean_movies)


def test_normalize_string():
    """Verify string normalization for search indexing."""
    assert normalize_string("Star Wars: Episode IV!") == "star wars episode iv"
    assert normalize_string("   The Matrix   ") == "the matrix"
    assert normalize_string("") == ""


def test_exact_and_case_insensitive_match(search_engine):
    """Test exact and case-insensitive queries."""
    res1 = search_engine.search("Toy Story")
    assert len(res1) > 0
    assert res1[0]["CleanTitle"] == "Toy Story"

    res2 = search_engine.search("toy story")
    assert len(res2) > 0
    assert res2[0]["CleanTitle"] == "Toy Story"

    res3 = search_engine.search("TOY STORY")
    assert len(res3) > 0
    assert res3[0]["CleanTitle"] == "Toy Story"


def test_article_normalization_search(search_engine):
    """Verify searching 'The Matrix' matches MovieLens 'Matrix, The (1999)'."""
    results = search_engine.search("The Matrix")
    assert len(results) > 0
    assert results[0]["CleanTitle"] == "Matrix, The" or results[0]["NormalizedTitle"] == "The Matrix"
    assert results[0]["Year"] == 1999


def test_partial_and_token_match(search_engine):
    """Verify partial matching for franchise titles."""
    results = search_engine.search("star wars")
    assert len(results) >= 3
    titles = [r["DisplayTitle"] for r in results]
    assert any("A New Hope" in t for t in titles)
    assert any("Empire Strikes Back" in t for t in titles)


def test_movie_id_lookup(search_engine):
    """Verify direct lookup by MovieID."""
    movie = search_engine.get_movie_by_id(1)
    assert movie is not None
    assert movie["MovieID"] == 1
    assert "Toy Story" in movie["Title"]

    # Non-existent ID
    none_movie = search_engine.get_movie_by_id(999999)
    assert none_movie is None


def test_unknown_movie_search(search_engine):
    """Verify searching for non-existent movies returns empty list without error."""
    results = search_engine.search("NonExistentMovieRandomXYZ123")
    assert len(results) == 0


def test_empty_or_whitespace_search(search_engine):
    """Verify blank search returns empty list."""
    assert search_engine.search("") == []
    assert search_engine.search("   ") == []
