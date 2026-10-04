"""
Version 2 — Advanced Content-Based Recommendation Model.
Recommends movies using TF-IDF on engineered metadata (title, normalized title,
genres, release year, decade) and cosine similarity.
"""

from pathlib import Path
from typing import List, Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

from src.config import TOP_K


class AdvancedContentRecommender:
    """
    Advanced Content-Based Recommender using TF-IDF vectorization
    over engineered movie textual metadata and cosine similarity.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            min_df=1,
            ngram_range=(1, 2),
            sublinear_tf=True,
            token_pattern=r"(?u)\b[\w\'-]+\b",
        )
        self.tfidf_matrix = None
        self.movies_df = None
        self.movie_id_to_idx = {}
        self.idx_to_movie_id = {}

    def fit(self, movies_df: pd.DataFrame) -> "AdvancedContentRecommender":
        """
        Builds the TF-IDF feature matrix and movie lookup index.
        """
        self.movies_df = movies_df.copy().reset_index(drop=True)

        if "V2_CombinedFeatures" not in self.movies_df.columns:
            raise KeyError(
                "Expected column 'V2_CombinedFeatures' in movies_df. Run preprocessing first."
            )

        text_corpus = self.movies_df["V2_CombinedFeatures"].fillna("").tolist()
        self.tfidf_matrix = self.vectorizer.fit_transform(text_corpus)

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
        if self.tfidf_matrix is None or self.movies_df is None:
            raise ValueError("Model is not fitted. Call fit() or load() first.")

        if movie_id not in self.movie_id_to_idx:
            raise KeyError(f"MovieID {movie_id} not found in model index.")

        target_idx = self.movie_id_to_idx[movie_id]
        target_vector = self.tfidf_matrix[target_idx]

        # Compute cosine similarity
        sim_scores = cosine_similarity(target_vector, self.tfidf_matrix).flatten()

        # Exclude target movie
        candidate_indices = np.arange(len(self.movies_df))
        candidate_indices = candidate_indices[candidate_indices != target_idx]

        candidate_scores = sim_scores[candidate_indices]

        # Multi-key deterministic tie breaking:
        # 1. Similarity score descending (-score)
        # 2. Release Year descending (-year)
        # 3. Alphabetical title ascending
        cand_years = np.nan_to_num(
            self.movies_df.iloc[candidate_indices]["Year"].fillna(0).values,
            nan=0.0,
        )
        cand_titles = self.movies_df.iloc[candidate_indices]["DisplayTitle"].values

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
            "tfidf_matrix": self.tfidf_matrix,
            "movies_df": self.movies_df,
            "movie_id_to_idx": self.movie_id_to_idx,
            "idx_to_movie_id": self.idx_to_movie_id,
        }
        joblib.dump(artifact, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "AdvancedContentRecommender":
        """Loads model artifact from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {path}")
        artifact = joblib.load(path)
        instance = cls()
        instance.vectorizer = artifact["vectorizer"]
        instance.tfidf_matrix = artifact["tfidf_matrix"]
        instance.movies_df = artifact["movies_df"]
        instance.movie_id_to_idx = artifact["movie_id_to_idx"]
        instance.idx_to_movie_id = artifact["idx_to_movie_id"]
        return instance

