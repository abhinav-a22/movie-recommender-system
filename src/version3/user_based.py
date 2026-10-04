"""
Version 3A — User-Based Collaborative Filtering.
Implements user-user neighborhood collaborative filtering using sparse representations.
Supports predicting user ratings for offline evaluation (RMSE, MAE) and ranking recommendations.
"""

from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import joblib

from src.config import TOP_K


class UserBasedCF:
    """
    User-Based Collaborative Filtering using user-user cosine similarity
    on mean-centered rating vectors with sparse CSC/CSR indexing.
    """

    def __init__(self, k_neighbors: int = 40):
        self.k_neighbors = k_neighbors
        self.user_movie_sparse = None  # (n_users x n_movies)
        self.user_movie_norm = None    # normalized for cosine similarity
        self.user_movie_csc = None     # CSC format for fast column slicing
        self.user_means = None         # mean rating per user
        self.movies_df = None
        self.user_id_to_idx = {}
        self.idx_to_user_id = {}
        self.movie_id_to_idx = {}
        self.idx_to_movie_id = {}

    def fit(
        self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame
    ) -> "UserBasedCF":
        """
        Builds sparse User-Movie matrix, user index mappings, and user rating averages.
        """
        self.movies_df = movies_df.copy().reset_index(drop=True)

        self.movie_id_to_idx = {
            int(mid): idx for idx, mid in enumerate(self.movies_df["MovieID"])
        }
        self.idx_to_movie_id = {
            idx: int(mid) for idx, mid in enumerate(self.movies_df["MovieID"])
        }

        valid_ratings = ratings_df[
            ratings_df["MovieID"].isin(self.movie_id_to_idx)
        ].copy()

        unique_users = sorted(valid_ratings["UserID"].unique())
        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.idx_to_user_id = {idx: uid for idx, uid in enumerate(unique_users)}

        n_users = len(unique_users)
        n_movies = len(self.movies_df)

        row_indices = [self.user_id_to_idx[uid] for uid in valid_ratings["UserID"]]
        col_indices = [self.movie_id_to_idx[mid] for mid in valid_ratings["MovieID"]]
        ratings_data = valid_ratings["Rating"].values.astype(np.float32)

        self.user_movie_sparse = csr_matrix(
            (ratings_data, (row_indices, col_indices)),
            shape=(n_users, n_movies),
            dtype=np.float32,
        )

        user_sums = np.array(self.user_movie_sparse.sum(axis=1)).flatten()
        user_counts = np.diff(self.user_movie_sparse.indptr)
        self.user_means = np.divide(
            user_sums,
            user_counts,
            out=np.zeros_like(user_sums, dtype=np.float32),
            where=user_counts > 0,
        )

        self.user_movie_norm = normalize(self.user_movie_sparse, norm="l2", axis=1)
        self.user_movie_csc = self.user_movie_sparse.tocsc()
        return self

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """
        Predicts the rating for a specific (user_id, movie_id) pair in O(1) time.
        Used for RMSE / MAE evaluation.
        """
        if user_id not in self.user_id_to_idx:
            return 3.0
        if movie_id not in self.movie_id_to_idx:
            return float(self.user_means[self.user_id_to_idx[user_id]])

        u_idx = self.user_id_to_idx[user_id]
        m_idx = self.movie_id_to_idx[movie_id]
        u_mean = float(self.user_means[u_idx])

        existing_val = self.user_movie_sparse[u_idx, m_idx]
        if existing_val > 0:
            return float(existing_val)

        col_start = self.user_movie_csc.indptr[m_idx]
        col_end = self.user_movie_csc.indptr[m_idx + 1]
        if col_start == col_end:
            return u_mean

        qual_users = self.user_movie_csc.indices[col_start:col_end]
        qual_ratings = self.user_movie_csc.data[col_start:col_end]

        other_mask = qual_users != u_idx
        if not np.any(other_mask):
            return u_mean

        qual_users = qual_users[other_mask]
        qual_ratings = qual_ratings[other_mask]

        u_vector = self.user_movie_norm[u_idx]
        sims = (self.user_movie_norm[qual_users].dot(u_vector.T)).toarray().flatten()

        positive_mask = sims > 0
        if not np.any(positive_mask):
            return u_mean

        top_indices = qual_users[positive_mask]
        top_sims = sims[positive_mask]
        top_ratings = qual_ratings[positive_mask]

        if len(top_sims) > self.k_neighbors:
            top_k_order = np.argsort(-top_sims)[: self.k_neighbors]
            top_indices = top_indices[top_k_order]
            top_sims = top_sims[top_k_order]
            top_ratings = top_ratings[top_k_order]

        neighbor_means = self.user_means[top_indices]
        deviations = top_ratings - neighbor_means

        sim_sum = np.sum(top_sims)
        if sim_sum == 0:
            return u_mean

        predicted = u_mean + (np.sum(top_sims * deviations) / sim_sum)
        return float(np.clip(predicted, 1.0, 5.0))

    def recommend_for_user(
        self, user_id: int, top_k: int = TOP_K
    ) -> List[Dict[str, Any]]:
        """
        Recommends top-K unrated movies for a given user using vectorized sparse operations.
        """
        if user_id not in self.user_id_to_idx:
            raise KeyError(f"UserID {user_id} not found in model index.")

        u_idx = self.user_id_to_idx[user_id]
        u_mean = float(self.user_means[u_idx])

        user_row = self.user_movie_sparse.getrow(u_idx).toarray().flatten()
        unrated_movie_indices = np.where(user_row == 0)[0]

        u_vector = self.user_movie_norm[u_idx]
        sims = (self.user_movie_norm.dot(u_vector.T)).toarray().flatten()
        sims[u_idx] = 0.0

        top_neighbors = np.argsort(-sims)[: self.k_neighbors]
        top_sims = sims[top_neighbors]
        valid_mask = top_sims > 0
        valid_neighbors = top_neighbors[valid_mask]
        valid_sims = top_sims[valid_mask]

        if len(valid_neighbors) == 0:
            return []

        neighbor_matrix = self.user_movie_sparse[valid_neighbors]
        indicator_matrix = neighbor_matrix.copy()
        indicator_matrix.data = np.ones_like(indicator_matrix.data, dtype=np.float32)

        means_k = self.user_means[valid_neighbors]
        sim_sums = indicator_matrix.T.dot(valid_sims)

        dev_matrix = neighbor_matrix.copy()
        row_indices = np.repeat(np.arange(len(valid_neighbors)), np.diff(neighbor_matrix.indptr))
        dev_matrix.data = dev_matrix.data - means_k[row_indices]

        weighted_dev = dev_matrix.T.dot(valid_sims)

        pred_scores = np.zeros(len(unrated_movie_indices), dtype=np.float32)
        valid_m_mask = sim_sums[unrated_movie_indices] > 0

        pred_scores[valid_m_mask] = u_mean + (
            weighted_dev[unrated_movie_indices[valid_m_mask]]
            / sim_sums[unrated_movie_indices[valid_m_mask]]
        )

        sorted_pos = np.argsort(-pred_scores)[:top_k]
        results = []
        for rank_num, pos in enumerate(sorted_pos, start=1):
            if pred_scores[pos] <= 0:
                continue
            m_idx = unrated_movie_indices[pos]
            row = self.movies_df.iloc[m_idx]
            results.append(
                {
                    "rank": rank_num,
                    "movie_id": int(row["MovieID"]),
                    "title": str(row.get("DisplayTitle", row["Title"])),
                    "year": row.get("Year", None),
                    "genres": str(row.get("DisplayGenres", row["Genres"])),
                    "predicted_rating": round(float(np.clip(pred_scores[pos], 1.0, 5.0)), 2),
                }
            )
        return results

    def save(self, filepath: Union[str, Path]):
        """Persists model artifact to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "k_neighbors": self.k_neighbors,
            "user_movie_sparse": self.user_movie_sparse,
            "user_movie_norm": self.user_movie_norm,
            "user_movie_csc": self.user_movie_csc,
            "user_means": self.user_means,
            "movies_df": self.movies_df,
            "user_id_to_idx": self.user_id_to_idx,
            "idx_to_user_id": self.idx_to_user_id,
            "movie_id_to_idx": self.movie_id_to_idx,
            "idx_to_movie_id": self.idx_to_movie_id,
        }
        joblib.dump(artifact, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "UserBasedCF":
        """Loads model artifact from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {path}")
        artifact = joblib.load(path)
        instance = cls(k_neighbors=artifact.get("k_neighbors", 40))
        instance.user_movie_sparse = artifact["user_movie_sparse"]
        instance.user_movie_norm = artifact["user_movie_norm"]
        instance.user_movie_csc = artifact.get("user_movie_csc", instance.user_movie_sparse.tocsc())
        instance.user_means = artifact["user_means"]
        instance.movies_df = artifact["movies_df"]
        instance.user_id_to_idx = artifact["user_id_to_idx"]
        instance.idx_to_user_id = artifact["idx_to_user_id"]
        instance.movie_id_to_idx = artifact["movie_id_to_idx"]
        instance.idx_to_movie_id = artifact["idx_to_movie_id"]
        return instance

