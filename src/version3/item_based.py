"""
Version 3B — Item-Based Collaborative Filtering.
Primary collaborative filtering model for movie-to-movie queries.
Learns from user rating behavior using sparse interaction matrices and cosine similarity.
Includes cold-start handling with configurable minimum rating thresholds.
"""

from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import joblib

from src.config import TOP_K, MIN_MOVIE_RATINGS


class ItemBasedCF:
    """
    Item-Based Collaborative Filtering using item-item rating pattern similarity.
    Calculates normalized cosine similarity between movie rating vectors in sparse space.
    """

    def __init__(self, min_ratings: int = MIN_MOVIE_RATINGS):
        self.min_ratings = min_ratings
        self.movie_user_sparse = None  # Normalized CSR matrix (movies x users)
        self.raw_matrix = None
        self.movies_df = None
        self.movie_id_to_idx = {}
        self.idx_to_movie_id = {}
        self.movie_rating_counts = {}
        self.movie_avg_ratings = {}
        self.fallback_popular_ids = []

    def fit(
        self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame
    ) -> "ItemBasedCF":
        """
        Builds sparse Movie-User matrix and precalculates rating statistics.
        """
        self.movies_df = movies_df.copy().reset_index(drop=True)

        # Mappings
        self.movie_id_to_idx = {
            int(mid): idx for idx, mid in enumerate(self.movies_df["MovieID"])
        }
        self.idx_to_movie_id = {
            idx: int(mid) for idx, mid in enumerate(self.movies_df["MovieID"])
        }

        # Filter ratings to movies known in movies_df
        valid_ratings = ratings_df[
            ratings_df["MovieID"].isin(self.movie_id_to_idx)
        ].copy()

        # Map UserID to contiguous 0-indexed values
        unique_users = sorted(valid_ratings["UserID"].unique())
        user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}

        row_indices = [
            self.movie_id_to_idx[mid] for mid in valid_ratings["MovieID"]
        ]
        col_indices = [user_to_idx[uid] for uid in valid_ratings["UserID"]]
        data_values = valid_ratings["Rating"].values.astype(np.float32)

        n_movies = len(self.movies_df)
        n_users = len(unique_users)

        # Build raw sparse matrix (movies x users)
        self.raw_matrix = csr_matrix(
            (data_values, (row_indices, col_indices)),
            shape=(n_movies, n_users),
            dtype=np.float32,
        )

        # Calculate rating counts and average rating per movie
        counts_series = valid_ratings.groupby("MovieID")["Rating"].agg(
            ["count", "mean"]
        )
        self.movie_rating_counts = {
            int(mid): int(row["count"]) for mid, row in counts_series.iterrows()
        }
        self.movie_avg_ratings = {
            int(mid): float(row["mean"]) for mid, row in counts_series.iterrows()
        }

        # Precompute fallback popular movies (high rating count and high average rating)
        popular_candidates = counts_series[
            counts_series["count"] >= self.min_ratings
        ].copy()
        # Weighted rating / bayesian average for fallback
        c_mean = valid_ratings["Rating"].mean()
        m_thresh = self.min_ratings
        popular_candidates["score"] = (
            (popular_candidates["count"] / (popular_candidates["count"] + m_thresh))
            * popular_candidates["mean"]
        ) + (
            (m_thresh / (popular_candidates["count"] + m_thresh)) * c_mean
        )
        popular_candidates = popular_candidates.sort_values(
            by="score", ascending=False
        )
        self.fallback_popular_ids = [
            int(mid) for mid in popular_candidates.index.tolist()
        ]

        # L2-normalize rows of movie_user matrix for fast cosine similarity:
        # dot product of row i with all rows equals cosine similarity!
        self.movie_user_sparse = normalize(self.raw_matrix, norm="l2", axis=1)

        return self

    def recommend(
        self, movie_id: int, top_k: int = TOP_K
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Recommends top-K movies based on rating pattern similarity.
        Returns (recommendations_list, cold_start_message_or_None).
        """
        if self.movie_user_sparse is None or self.movies_df is None:
            raise ValueError("Model is not fitted. Call fit() or load() first.")

        if movie_id not in self.movie_id_to_idx:
            raise KeyError(f"MovieID {movie_id} not found in model index.")

        rating_count = self.movie_rating_counts.get(movie_id, 0)
        target_idx = self.movie_id_to_idx[movie_id]

        cold_start_warning = None
        # Cold start condition: fewer ratings than min_ratings threshold
        if rating_count < self.min_ratings:
            cold_start_warning = (
                f"Collaborative filtering has insufficient rating data for this movie "
                f"({rating_count} ratings < {self.min_ratings} threshold). "
                f"Using the strongest available collaborative recommendations."
            )

        # Compute cosine similarity using sparse matrix dot product
        target_vector = self.movie_user_sparse[target_idx]
        sim_scores = (self.movie_user_sparse.dot(target_vector.T)).toarray().flatten()

        candidate_indices = np.arange(len(self.movies_df))
        candidate_indices = candidate_indices[candidate_indices != target_idx]

        candidate_scores = sim_scores[candidate_indices]

        cand_counts = np.array(
            [
                self.movie_rating_counts.get(self.idx_to_movie_id[idx], 0)
                for idx in candidate_indices
            ]
        )
        cand_avgs = np.array(
            [
                self.movie_avg_ratings.get(self.idx_to_movie_id[idx], 0.0)
                for idx in candidate_indices
            ]
        )
        cand_titles = self.movies_df.iloc[candidate_indices]["DisplayTitle"].values

        # If movie had zero ratings or strictly zero similarity across all candidates, fallback
        if np.max(candidate_scores) <= 0.0:
            results = []
            for rank_num, fb_id in enumerate(self.fallback_popular_ids, start=1):
                if fb_id == movie_id:
                    continue
                fb_idx = self.movie_id_to_idx[fb_id]
                row = self.movies_df.iloc[fb_idx]
                results.append(
                    {
                        "rank": len(results) + 1,
                        "movie_id": fb_id,
                        "title": str(row.get("DisplayTitle", row["Title"])),
                        "year": row.get("Year", None),
                        "genres": str(row.get("DisplayGenres", row["Genres"])),
                        "collaborative_score": round(
                            float(self.movie_avg_ratings.get(fb_id, 3.0) / 5.0),
                            4,
                        ),
                    }
                )
                if len(results) >= top_k:
                    break
            return results, cold_start_warning

        qualifying_mask = cand_counts >= self.min_ratings
        if np.sum(qualifying_mask) >= top_k:
            eligible_pos = np.where(qualifying_mask)[0]
        else:
            eligible_pos = np.arange(len(candidate_indices))

        sorted_pos = sorted(
            eligible_pos,
            key=lambda p: (
                -candidate_scores[p],
                -cand_counts[p],
                -cand_avgs[p],
                cand_titles[p],
            ),
        )

        results = []
        for rank_num, pos in enumerate(sorted_pos[:top_k], start=1):
            actual_idx = candidate_indices[pos]
            row = self.movies_df.iloc[actual_idx]
            mid = int(row["MovieID"])
            results.append(
                {
                    "rank": rank_num,
                    "movie_id": mid,
                    "title": str(row.get("DisplayTitle", row["Title"])),
                    "year": row.get("Year", None),
                    "genres": str(row.get("DisplayGenres", row["Genres"])),
                    "collaborative_score": round(
                        float(candidate_scores[pos]), 4
                    ),
                }
            )

        return results, cold_start_warning

    def save(self, filepath: Union[str, Path]):
        """Persists model artifact to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "min_ratings": self.min_ratings,
            "movie_user_sparse": self.movie_user_sparse,
            "raw_matrix": self.raw_matrix,
            "movies_df": self.movies_df,
            "movie_id_to_idx": self.movie_id_to_idx,
            "idx_to_movie_id": self.idx_to_movie_id,
            "movie_rating_counts": self.movie_rating_counts,
            "movie_avg_ratings": self.movie_avg_ratings,
            "fallback_popular_ids": self.fallback_popular_ids,
        }
        joblib.dump(artifact, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "ItemBasedCF":
        """Loads model artifact from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {path}")
        artifact = joblib.load(path)
        instance = cls(min_ratings=artifact.get("min_ratings", MIN_MOVIE_RATINGS))
        instance.movie_user_sparse = artifact["movie_user_sparse"]
        instance.raw_matrix = artifact["raw_matrix"]
        instance.movies_df = artifact["movies_df"]
        instance.movie_id_to_idx = artifact["movie_id_to_idx"]
        instance.idx_to_movie_id = artifact["idx_to_movie_id"]
        instance.movie_rating_counts = artifact["movie_rating_counts"]
        instance.movie_avg_ratings = artifact["movie_avg_ratings"]
        instance.fallback_popular_ids = artifact["fallback_popular_ids"]
        return instance

