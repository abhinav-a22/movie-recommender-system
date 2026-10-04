"""
Data Loader module for MovieLens 1M dataset.
Handles loading users.dat, movies.dat, and ratings.dat with validation and encoding support.
"""

from pathlib import Path
from typing import Tuple, Optional
import pandas as pd

from src.config import (
    DATA_DIR,
    USERS_FILE,
    MOVIES_FILE,
    RATINGS_FILE,
    REQUIRED_DATA_FILES,
    DATASET_ENCODING,
)


def check_dataset_exists(data_dir: Optional[Path] = None) -> bool:
    """Verify that all required .dat files exist and are non-empty."""
    base_dir = data_dir or DATA_DIR
    for filename in ["users.dat", "movies.dat", "ratings.dat"]:
        filepath = base_dir / filename
        if not filepath.exists() or filepath.stat().st_size == 0:
            return False
    return True


def load_users(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load users.dat into a pandas DataFrame.
    Format: UserID::Gender::Age::Occupation::Zip-code
    """
    path = filepath or USERS_FILE
    if not path.exists():
        raise FileNotFoundError(f"Users dataset not found at: {path}")

    users = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["UserID", "Gender", "Age", "Occupation", "Zip-code"],
        encoding=DATASET_ENCODING,
    )
    return users


def load_movies(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load movies.dat into a pandas DataFrame.
    Format: MovieID::Title::Genres
    """
    path = filepath or MOVIES_FILE
    if not path.exists():
        raise FileNotFoundError(f"Movies dataset not found at: {path}")

    movies = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["MovieID", "Title", "Genres"],
        encoding=DATASET_ENCODING,
    )
    return movies


def load_ratings(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load ratings.dat into a pandas DataFrame.
    Format: UserID::MovieID::Rating::Timestamp
    """
    path = filepath or RATINGS_FILE
    if not path.exists():
        raise FileNotFoundError(f"Ratings dataset not found at: {path}")

    ratings = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["UserID", "MovieID", "Rating", "Timestamp"],
        encoding=DATASET_ENCODING,
    )
    return ratings


def validate_datasets(
    users_df: pd.DataFrame, movies_df: pd.DataFrame, ratings_df: pd.DataFrame
) -> bool:
    """
    Validates structural integrity and constraints of MovieLens 1M dataframes:
    - Expected columns present
    - Non-empty records
    - Rating bounds (1 to 5)
    - Valid IDs
    """
    # Check users
    expected_user_cols = {"UserID", "Gender", "Age", "Occupation", "Zip-code"}
    if not expected_user_cols.issubset(users_df.columns) or len(users_df) == 0:
        raise ValueError("Invalid users dataset structure or empty dataframe.")

    # Check movies
    expected_movie_cols = {"MovieID", "Title", "Genres"}
    if not expected_movie_cols.issubset(movies_df.columns) or len(movies_df) == 0:
        raise ValueError("Invalid movies dataset structure or empty dataframe.")

    # Check ratings
    expected_rating_cols = {"UserID", "MovieID", "Rating", "Timestamp"}
    if not expected_rating_cols.issubset(ratings_df.columns) or len(ratings_df) == 0:
        raise ValueError("Invalid ratings dataset structure or empty dataframe.")

    if not ratings_df["Rating"].between(1, 5).all():
        raise ValueError("Ratings contains values outside the [1, 5] range.")

    return True


def load_all_data(
    data_dir: Optional[Path] = None, validate: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and return (users_df, movies_df, ratings_df).
    Optionally runs validation.
    """
    base_dir = data_dir or DATA_DIR
    users_df = load_users(base_dir / "users.dat")
    movies_df = load_movies(base_dir / "movies.dat")
    ratings_df = load_ratings(base_dir / "ratings.dat")

    if validate:
        validate_datasets(users_df, movies_df, ratings_df)

    return users_df, movies_df, ratings_df

