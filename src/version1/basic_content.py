"""
Version 1 — Basic Content-Based Recommendation Model.
Recommends movies using genre-based vectorization and cosine similarity.
"""

from pathlib import Path
from typing import List, Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

from src.config import TOP_K


class BasicContentRecommender:
    """
    Basic Content-Based Recommender using multi-hot genre vectors
    and cosine similarity.
    """

    def __init__(self):
        self.vectorizer = CountVectorizer(
            binary=True, token_pattern=r"(?u)\b[\w\'-]+\b"
        )
        self.genre_matrix = None
        self.movies_df = None
        self.movie_id_to_idx = {}
        self.idx_to_movie_id = {}

    def fit(self, movies_df: pd.DataFrame) -> "BasicContentRecommender":
        """
        Builds the genre feature matrix and movie lookup index.
        """
        self.movies_df = movies_df.copy().reset_index(drop=True)
        # Ensure V1_GenreFeatures exists
        if "V1_GenreFeatures" not in self.movies_df.columns:
            self.movies_df["V1_GenreFeatures"] = self.movies_df["Genres"].apply(
                lambda g: " ".join(g.split("|")) if isinstance(g, str) else "Unknown"
            )

        genre_texts = self.movies_df["V1_GenreFeatures"].tolist()
        self.genre_matrix = self.vectorizer.fit_transform(genre_texts)

        # Build index mappings
        self.movie_id_to_idx = {
            int(mid): idx for idx, mid in enumerate(self.movies_df["MovieID"])
        }
        self.idx_to_movie_id = {
            idx: int(mid) for idx, mid in enumerate(self.movies_df["MovieID"])
        }
        return self

    def recommend(
        self, movie_id: int, top_k: int = TOP_K
    ) -> List[Dict[str, Any]]:
        """
        Generates top-K recommendations for a given MovieID.
        Excludes the query movie itself.
        """
        if self.genre_matrix is None or self.movies_df is None:
            raise ValueError("Model is not fitted. Call fit() or load() first.")

        if movie_id not in self.movie_id_to_idx:
            raise KeyError(f"MovieID {movie_id} not found in model index.")

        target_idx = self.movie_id_to_idx[movie_id]
        target_vector = self.genre_matrix[target_idx]

        # Compute cosine similarity between target movie and all candidate movies
        sim_scores = cosine_similarity(target_vector, self.genre_matrix).flatten()

        # Build candidate indices excluding target movie
        candidate_indices = np.arange(len(self.movies_df))
        candidate_indices = candidate_indices[candidate_indices != target_idx]

        # Candidate scores
        candidate_scores = sim_scores[candidate_indices]

        # Candidate titles and years for deterministic tie-breaking:
        # 1. Similarity score descending (-score)
        # 2. Release Year descending (newer movies if tied)
        # 3. Alphabetical title ascending
        cand_years = np.nan_to_num(
            self.movies_df.iloc[candidate_indices]["Year"].fillna(0).values,
            nan=0.0,
        )
        cand_titles = self.movies_df.iloc[candidate_indices]["DisplayTitle"].values

        # Sort indices
        sorted_candidates = sorted(
            range(len(candidate_indices)),
            key=lambda i: (
                -candidate_scores[i],
                -cand_years[i],
                cand_titles[i],
            ),
        )

        results = []
        for rank_num, idx_in_cand in enumerate(sorted_candidates[:top_k], start=1):
            actual_idx = candidate_indices[idx_in_cand]
            row = self.movies_df.iloc[actual_idx]
            results.append(
                {
                    "rank": rank_num,
                    "movie_id": int(row["MovieID"]),
                    "title": str(row.get("DisplayTitle", row["Title"])),
                    "year": row.get("Year", None),
                    "genres": str(row.get("DisplayGenres", row["Genres"])),
                    "similarity_score": round(float(candidate_scores[idx_in_cand]), 4),
                }
            )

        return results

    def save(self, filepath: Union[str, Path]):
        """Persists model artifact to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "vectorizer": self.vectorizer,
            "genre_matrix": self.genre_matrix,
            "movies_df": self.movies_df,
            "movie_id_to_idx": self.movie_id_to_idx,
            "idx_to_movie_id": self.idx_to_movie_id,
        }
        joblib.dump(artifact, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "BasicContentRecommender":
        """Loads model artifact from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {path}")
        artifact = joblib.load(path)
        instance = cls()
        instance.vectorizer = artifact["vectorizer"]
        instance.genre_matrix = artifact["genre_matrix"]
        instance.movies_df = artifact["movies_df"]
        instance.movie_id_to_idx = artifact["movie_id_to_idx"]
        instance.idx_to_movie_id = artifact["idx_to_movie_id"]
        return instance

