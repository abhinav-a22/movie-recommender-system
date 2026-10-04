"""
Preprocessing and feature engineering module for MovieLens 1M dataset.
Handles title cleaning, article inversion normalization, release year and decade extraction,
genre tokenization, text representation engineering for Version 2, and rating validation.
"""

import re
from typing import Tuple, Optional
import pandas as pd
import numpy as np


def extract_year_and_clean_title(title_raw: str) -> Tuple[str, Optional[int]]:
    """
    Extracts 4-digit release year from a title like 'Toy Story (1995)' or
    'Postman, The (Il Postino) (1994)' and returns (clean_title, year).
    If no year is found, returns (title_raw, None).
    """
    if not isinstance(title_raw, str):
        return str(title_raw), None

    # Search for (YYYY) typically at the end of the title
    match = re.search(r"\((\d{4})\)\s*$", title_raw.strip())
    if match:
        year = int(match.group(1))
        clean_title = re.sub(r"\s*\(\d{4}\)\s*$", "", title_raw.strip()).strip()
        return clean_title, year

    # Secondary check anywhere in title
    match_any = re.search(r"\((\d{4})\)", title_raw)
    if match_any:
        year = int(match_any.group(1))
        clean_title = re.sub(r"\s*\(\d{4}\)", "", title_raw).strip()
        return clean_title, year

    return title_raw.strip(), None


def normalize_title_articles(title: str) -> str:
    """
    Normalizes inverted article titles from MovieLens:
    'Matrix, The' -> 'The Matrix'
    'Lion King, The' -> 'The Lion King'
    'Clockwork Orange, A' -> 'A Clockwork Orange'
    'Affair to Remember, An' -> 'An Affair to Remember'
    """
    if not isinstance(title, str):
        return ""

    title = title.strip()
    if title.endswith(", The"):
        return f"The {title[:-5]}".strip()
    elif title.endswith(", the"):
        return f"the {title[:-5]}".strip()
    elif title.endswith(", A"):
        return f"A {title[:-3]}".strip()
    elif title.endswith(", a"):
        return f"a {title[:-3]}".strip()
    elif title.endswith(", An"):
        return f"An {title[:-4]}".strip()
    elif title.endswith(", an"):
        return f"an {title[:-4]}".strip()
    return title


def get_decade(year: Optional[int]) -> str:
    """Returns decade string representation, e.g., 1995 -> '1990s'."""
    if year is None or pd.isna(year) or year <= 0:
        return "unknown_decade"
    decade = (int(year) // 10) * 10
    return f"{decade}s"


def clean_genres_for_v1(genres_raw: str) -> str:
    """
    Formats genres for Version 1 Basic Content-Based vectorization:
    'Animation|Children's|Comedy' -> 'Animation Children's Comedy'
    """
    if not isinstance(genres_raw, str) or not genres_raw:
        return "Unknown"
    # Replace pipe delimiter with space
    return " ".join(genres_raw.split("|"))


def create_v2_combined_features(
    clean_title: str,
    normalized_title: str,
    genres_raw: str,
    year: Optional[int],
    decade: str,
) -> str:
    """
    Engineers rich combined textual representation for Version 2:
    Clean Title + Normalized Title + Genres + Release Year + Decade.
    Example: 'toy story the toy story animation children comedy 1995 1990s'
    """
    clean_t = clean_title.lower() if clean_title else ""
    norm_t = normalized_title.lower() if normalized_title else ""
    # Remove apostrophes in genres for clean tokenization (e.g., children's -> childrens)
    genres_clean = genres_raw.lower().replace("|", " ").replace("'", "")
    year_str = str(year) if year is not None and not pd.isna(year) else ""
    decade_str = decade.lower() if decade else ""

    parts = [clean_t, norm_t, genres_clean, year_str, decade_str]
    return " ".join(p for p in parts if p).strip()


def preprocess_movies(movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes the raw movies dataframe:
    - Extracts year
    - Extracts clean title & normalized title
    - Formats genres
    - Creates combined metadata string for Version 2
    """
    df = movies_df.copy()

    # Extract year and clean title
    extracted = df["Title"].apply(extract_year_and_clean_title)
    df["CleanTitle"] = [item[0] for item in extracted]
    df["Year"] = [item[1] for item in extracted]

    # Normalize inverted articles
    df["NormalizedTitle"] = df["CleanTitle"].apply(normalize_title_articles)

    # Calculate decade
    df["Decade"] = df["Year"].apply(get_decade)

    # Process genres for V1
    df["V1_GenreFeatures"] = df["Genres"].apply(clean_genres_for_v1)

    # Display genres (formatted with pipe and space)
    df["DisplayGenres"] = df["Genres"].apply(
        lambda g: " | ".join(g.split("|")) if isinstance(g, str) else "Unknown"
    )

    # Format Display Title: Normalized Title (Year) or Clean Title (Year)
    df["DisplayTitle"] = df.apply(
        lambda row: f"{row['NormalizedTitle']} ({int(row['Year'])})"
        if pd.notna(row["Year"])
        else row["NormalizedTitle"],
        axis=1,
    )

    # Create combined text for V2
    df["V2_CombinedFeatures"] = df.apply(
        lambda row: create_v2_combined_features(
            row["CleanTitle"],
            row["NormalizedTitle"],
            row["Genres"],
            row["Year"],
            row["Decade"],
        ),
        axis=1,
    )

    return df


def preprocess_ratings(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and validates the ratings dataframe:
    - Drops nulls
    - Validates rating range [1.0, 5.0]
    - Ensures integer data types
    """
    df = ratings_df.dropna(subset=["UserID", "MovieID", "Rating"]).copy()
    df["UserID"] = df["UserID"].astype(int)
    df["MovieID"] = df["MovieID"].astype(int)
    df["Rating"] = df["Rating"].astype(float)
    df = df[df["Rating"].between(1.0, 5.0)]
    return df


def train_test_split_ratings(
    ratings_df: pd.DataFrame, test_ratio: float = 0.20, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits ratings into train and test sets reproducibly.
    Stratifies or samples randomly based on random_state.
    """
    rng = np.random.RandomState(random_state)
    shuffled_indices = rng.permutation(len(ratings_df))
    test_size = int(len(ratings_df) * test_ratio)

    test_idx = shuffled_indices[:test_size]
    train_idx = shuffled_indices[test_size:]

    train_df = ratings_df.iloc[train_idx].copy()
    test_df = ratings_df.iloc[test_idx].copy()

    return train_df, test_df

