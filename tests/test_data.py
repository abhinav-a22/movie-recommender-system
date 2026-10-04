"""
Unit tests for data loading, parsing, and preprocessing logic.
"""

import pytest
import pandas as pd
from pathlib import Path

from src.data_loader import load_users, load_movies, load_ratings, validate_datasets
from src.preprocessing import (
    extract_year_and_clean_title,
    normalize_title_articles,
    get_decade,
    preprocess_movies,
    preprocess_ratings,
)
from src.config import DATA_DIR


@pytest.fixture(scope="module")
def loaded_data():
    users = load_users()
    movies = load_movies()
    ratings = load_ratings()
    return users, movies, ratings


def test_data_loading_and_counts(loaded_data):
    """Verify that dataset files can be parsed and match expected row minimums."""
    users, movies, ratings = loaded_data

    assert len(users) >= 6000, f"Expected >= 6000 users, got {len(users)}"
    assert len(movies) >= 3800, f"Expected >= 3800 movies, got {len(movies)}"
    assert len(ratings) >= 1000000, f"Expected >= 1M ratings, got {len(ratings)}"


def test_validation_rules(loaded_data):
    """Test dataframe structure and value range validation."""
    users, movies, ratings = loaded_data

    assert validate_datasets(users, movies, ratings) is True

    # Test failure on invalid ratings (using integer 9 outside [1, 5] range)
    bad_ratings = ratings.head(10).copy()
    bad_ratings.loc[0, "Rating"] = 9
    with pytest.raises(ValueError):
        validate_datasets(users, movies, bad_ratings)


def test_year_and_title_extraction():
    """Test title cleaning and release year extraction regex."""
    clean, year = extract_year_and_clean_title("Toy Story (1995)")
    assert clean == "Toy Story"
    assert year == 1995

    clean, year = extract_year_and_clean_title("The Matrix (1999)")
    assert clean == "The Matrix"
    assert year == 1999

    clean, year = extract_year_and_clean_title("Postman, The (Il Postino) (1994)")
    assert clean == "Postman, The (Il Postino)"
    assert year == 1994


def test_article_inversion_normalization():
    """Test conversion of inverted articles like 'Matrix, The' -> 'The Matrix'."""
    assert normalize_title_articles("Matrix, The") == "The Matrix"
    assert normalize_title_articles("Lion King, The") == "The Lion King"
    assert normalize_title_articles("Clockwork Orange, A") == "A Clockwork Orange"
    assert normalize_title_articles("Affair to Remember, An") == "An Affair to Remember"
    assert normalize_title_articles("Toy Story") == "Toy Story"


def test_decade_calculation():
    """Test decade string calculation."""
    assert get_decade(1995) == "1990s"
    assert get_decade(1977) == "1970s"
    assert get_decade(2000) == "2000s"
    assert get_decade(None) == "unknown_decade"


def test_movie_preprocessing_pipeline(loaded_data):
    """Test complete movie feature engineering."""
    _, movies, _ = loaded_data
    sample_movies = movies.head(20)
    proc_df = preprocess_movies(sample_movies)

    assert "Year" in proc_df.columns
    assert "NormalizedTitle" in proc_df.columns
    assert "DisplayTitle" in proc_df.columns
    assert "V1_GenreFeatures" in proc_df.columns
    assert "V2_CombinedFeatures" in proc_df.columns
    assert proc_df["Year"].notna().all()


def test_ratings_preprocessing_and_types():
    """Test ratings data cleaning and types."""
    raw_ratings = pd.DataFrame(
        {
            "UserID": [1, 2, 3, None],
            "MovieID": [10, 20, 30, 40],
            "Rating": [4.0, 5.0, 6.5, 3.0],  # 6.5 is invalid
            "Timestamp": [100, 200, 300, 400],
        }
    )
    clean_ratings = preprocess_ratings(raw_ratings)
    # Row with NaN UserID and row with 6.5 rating should be dropped
    assert len(clean_ratings) == 2
    assert (clean_ratings["Rating"] <= 5.0).all()
    assert (clean_ratings["Rating"] >= 1.0).all()
