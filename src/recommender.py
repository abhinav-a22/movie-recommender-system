"""
Integrated Recommendation System orchestrator.
Manages V1, V2, V3, and V4 ensemble pipelines, search lookup,
and formatted terminal displays.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

from src.config import (
    TOP_K,
    V1_MODEL_PATH,
    V2_MODEL_PATH,
    V3_ITEM_MODEL_PATH,
    V3_USER_MODEL_PATH,
)
from src.version1.basic_content import BasicContentRecommender
from src.version2.advanced_content import AdvancedContentRecommender
from src.version3.collaborative import CollaborativeRecommender
from src.version4.union_ranker import UnionRanker
from src.search import MovieSearchEngine


class MovieRecommenderSystem:
    """
    Main integrated recommender application facade.
    Loads and serves all four recommendation outputs.
    """

    def __init__(self):
        self.v1 = None
        self.v2 = None
        self.v3 = None
        self.v4 = UnionRanker()
        self.search_engine = None
        self.movies_df = None

    def is_ready(self) -> bool:
        """Checks if all required models and indices are loaded."""
        return (
            self.v1 is not None
            and self.v2 is not None
            and self.v3 is not None
            and self.search_engine is not None
        )

    def load_models(
        self,
        v1_path: Path = V1_MODEL_PATH,
        v2_path: Path = V2_MODEL_PATH,
        v3_item_path: Path = V3_ITEM_MODEL_PATH,
        v3_user_path: Path = V3_USER_MODEL_PATH,
    ):
        """Loads precomputed model artifacts from disk."""
        self.v1 = BasicContentRecommender.load(v1_path)
        self.v2 = AdvancedContentRecommender.load(v2_path)
        self.v3 = CollaborativeRecommender.load(
            item_path=v3_item_path, user_path=v3_user_path
        )
        self.movies_df = self.v1.movies_df
        self.search_engine = MovieSearchEngine(self.movies_df)

    def recommend(
        self, movie_id: int, top_k: int = TOP_K
    ) -> Dict[str, Any]:
        """
        Generates full 4-version recommendation dictionary for a selected movie ID.
        """
        if not self.is_ready():
            raise RuntimeError("Recommendation system is not loaded.")

        movie_info = self.search_engine.get_movie_by_id(movie_id)
        if not movie_info:
            raise KeyError(f"MovieID {movie_id} not found in database.")

        # Version 1: Basic Content-Based
        v1_recs = self.v1.recommend(movie_id, top_k=top_k)

        # Version 2: Advanced Content-Based
        v2_recs = self.v2.recommend(movie_id, top_k=top_k)

        # Version 3: Collaborative Filtering
        v3_recs, cold_start_warning = self.v3.recommend(movie_id, top_k=top_k)

        # Version 4: Combined / Union Ranking
        v4_recs = self.v4.rank(
            v1_recs=v1_recs,
            v2_recs=v2_recs,
            v3_recs=v3_recs,
            query_movie_id=movie_id,
            top_k=top_k,
        )

        return {
            "movie": movie_info,
            "v1": v1_recs,
            "v2": v2_recs,
            "v3": v3_recs,
            "v4": v4_recs,
            "cold_start_warning": cold_start_warning,
        }

    def format_terminal_output(self, result: Dict[str, Any]) -> str:
        """
        Formats all 4 recommendation outputs for terminal display.
        """
        lines = []
        movie = result["movie"]
        cold_start = result.get("cold_start_warning")

        lines.append("=" * 60)
        lines.append("                      SELECTED MOVIE")
        lines.append("=" * 60)
        lines.append(f"{movie['DisplayTitle']}")
        lines.append(f"Genres: {movie['DisplayGenres']}")
        lines.append("")

        # Version 1
        lines.append("=" * 60)
        lines.append("VERSION 1 — BASIC CONTENT-BASED RECOMMENDATIONS")
        lines.append("=" * 60)
        if not result["v1"]:
            lines.append("No recommendations available.")
        else:
            for rec in result["v1"]:
                lines.append(f"{rec['rank']}. {rec['title']}")
                lines.append(f"   Genres: {rec['genres']}")
                lines.append(f"   Similarity Score: {rec['similarity_score']:.4f}")
                lines.append("")

        # Version 2
        lines.append("=" * 60)
        lines.append("VERSION 2 — ADVANCED CONTENT-BASED RECOMMENDATIONS")
        lines.append("=" * 60)
        if not result["v2"]:
            lines.append("No recommendations available.")
        else:
            for rec in result["v2"]:
                lines.append(f"{rec['rank']}. {rec['title']}")
                lines.append(f"   Genres: {rec['genres']}")
                lines.append(f"   Similarity Score: {rec['similarity_score']:.4f}")
                lines.append("")

        # Version 3
        lines.append("=" * 60)
        lines.append("VERSION 3 — COLLABORATIVE FILTERING RECOMMENDATIONS")
        lines.append("=" * 60)
        if cold_start:
            lines.append(f"[NOTE] {cold_start}")
            lines.append("")
        if not result["v3"]:
            lines.append("No recommendations available.")
        else:
            for rec in result["v3"]:
                lines.append(f"{rec['rank']}. {rec['title']}")
                lines.append(f"   Genres: {rec['genres']}")
                lines.append(
                    f"   Collaborative Similarity Score: {rec['collaborative_score']:.4f}"
                )
                lines.append("")

        # Version 4
        lines.append("=" * 60)
        lines.append("VERSION 4 — COMBINED / UNION RECOMMENDATIONS")
        lines.append("=" * 60)
        if not result["v4"]:
            lines.append("No combined recommendations available.")
        else:
            for rec in result["v4"]:
                lines.append(f"{rec['rank']}. {rec['title']}")
                lines.append(f"   Recommended by: {rec['recommended_by']}")
                lines.append(f"   Model Agreement: {rec['model_agreement']}")
                lines.append(f"   Combined Rank Score: {rec['combined_score']:.4f}")
                lines.append("")

        lines.append("=" * 60)
        lines.append("                           END")
        lines.append("=" * 60)

        return "\n".join(lines)

