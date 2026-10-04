"""
Version 3 — Collaborative Filtering Facade.
Combines Item-Based CF (primary for movie query recommendations)
and User-Based CF (for user queries and evaluation).
"""

from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple
import pandas as pd

from src.config import (
    TOP_K,
    MIN_MOVIE_RATINGS,
    V3_ITEM_MODEL_PATH,
    V3_USER_MODEL_PATH,
)
from .item_based import ItemBasedCF
from .user_based import UserBasedCF


class CollaborativeRecommender:
    """
    Facade managing collaborative filtering models.
    Delegates movie queries to ItemBasedCF and user queries to UserBasedCF.
    """

    def __init__(self, min_ratings: int = MIN_MOVIE_RATINGS, k_neighbors: int = 40):
        self.item_model = ItemBasedCF(min_ratings=min_ratings)
        self.user_model = UserBasedCF(k_neighbors=k_neighbors)

    def fit(
        self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame
    ) -> "CollaborativeRecommender":
        """Fits both Item-Based and User-Based CF models."""
        self.item_model.fit(movies_df, ratings_df)
        self.user_model.fit(movies_df, ratings_df)
        return self

    def recommend(
        self, movie_id: int, top_k: int = TOP_K
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Primary interface for terminal movie recommendation.
        Returns item-item collaborative recommendations and cold-start warning (if any).
        """
        return self.item_model.recommend(movie_id, top_k=top_k)

    def recommend_for_user(
        self, user_id: int, top_k: int = TOP_K
    ) -> List[Dict[str, Any]]:
        """Recommends movies for a specific user ID."""
        return self.user_model.recommend_for_user(user_id, top_k=top_k)

    def save(
        self,
        item_path: Union[str, Path] = V3_ITEM_MODEL_PATH,
        user_path: Union[str, Path] = V3_USER_MODEL_PATH,
    ):
        """Saves item and user CF models."""
        self.item_model.save(item_path)
        self.user_model.save(user_path)

    @classmethod
    def load(
        cls,
        item_path: Union[str, Path] = V3_ITEM_MODEL_PATH,
        user_path: Union[str, Path] = V3_USER_MODEL_PATH,
    ) -> "CollaborativeRecommender":
        """Loads collaborative filtering models from disk."""
        instance = cls()
        instance.item_model = ItemBasedCF.load(item_path)
        if Path(user_path).exists():
            instance.user_model = UserBasedCF.load(user_path)
        return instance

