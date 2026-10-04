"""
Version 4 — Combined / Union Recommendation Ranker.
Ensemble ranking layer consuming recommendations from V1, V2, and V3.
Strictly implements the 7-tier priority hierarchy, normalized rank scores,
and deterministic tie-breaking.
"""

from typing import List, Dict, Any, Optional
from src.config import PRIORITY_ORDER, PRIORITY_DISPLAY_NAMES, TOP_K


def calculate_normalized_rank_score(rank: int, n: int) -> float:
    """
    Calculates normalized rank score for a movie at rank (1-indexed)
    in a list of size N:
    rank_score = (N - rank + 1) / N
    """
    if n <= 0:
        return 0.0
    return (n - rank + 1) / n


def determine_priority_group(in_v1: bool, in_v2: bool, in_v3: bool) -> str:
    """
    Maps version membership boolean flags to one of the 7 exact priority groups:
    1. V1 + V2 + V3
    2. V2 + V3
    3. V1 + V3
    4. V1 + V2
    5. V3 only
    6. V2 only
    7. V1 only
    """
    if in_v1 and in_v2 and in_v3:
        return "V1_V2_V3"
    elif in_v2 and in_v3:
        return "V2_V3"
    elif in_v1 and in_v3:
        return "V1_V3"
    elif in_v1 and in_v2:
        return "V1_V2"
    elif in_v3:
        return "V3"
    elif in_v2:
        return "V2"
    elif in_v1:
        return "V1"
    else:
        return "NONE"


class UnionRanker:
    """
    Ensemble Ranker for Version 4.
    Fuses top recommendations from V1, V2, and V3 according to strict priority rules.
    """

    def __init__(self, priority_order: Optional[List[str]] = None):
        self.priority_order = priority_order or list(PRIORITY_ORDER)
        self.priority_map = {grp: idx for idx, grp in enumerate(self.priority_order)}

    def rank(
        self,
        v1_recs: List[Dict[str, Any]],
        v2_recs: List[Dict[str, Any]],
        v3_recs: List[Dict[str, Any]],
        query_movie_id: Optional[int] = None,
        top_k: Optional[int] = TOP_K,
    ) -> List[Dict[str, Any]]:
        """
        Consumes recommendations from V1, V2, and V3 and produces the combined ranking.

        Sorting logic (Requirement 23):
        1. Exact priority group (V1_V2_V3 > V2_V3 > V1_V3 > V1_V2 > V3 > V2 > V1)
        2. Higher combined normalized rank score
        3. Better average original rank (smaller numerical value)
        4. Higher number of model appearances
        5. Alphabetical movie title (deterministic tie-breaker)
        """
        n_v1 = len(v1_recs) if v1_recs else TOP_K
        n_v2 = len(v2_recs) if v2_recs else TOP_K
        n_v3 = len(v3_recs) if v3_recs else TOP_K

        v1_map = {int(rec["movie_id"]): rec for rec in v1_recs}
        v2_map = {int(rec["movie_id"]): rec for rec in v2_recs}
        v3_map = {int(rec["movie_id"]): rec for rec in v3_recs}

        all_movie_ids = set(v1_map.keys()) | set(v2_map.keys()) | set(v3_map.keys())

        if query_movie_id is not None and query_movie_id in all_movie_ids:
            all_movie_ids.remove(query_movie_id)

        entries = []
        for mid in all_movie_ids:
            rec_v1 = v1_map.get(mid)
            rec_v2 = v2_map.get(mid)
            rec_v3 = v3_map.get(mid)

            in_v1 = rec_v1 is not None
            in_v2 = rec_v2 is not None
            in_v3 = rec_v3 is not None

            sample_rec = rec_v1 or rec_v2 or rec_v3
            title = sample_rec["title"]
            year = sample_rec.get("year", None)
            genres = sample_rec.get("genres", "")

            score_v1 = (
                calculate_normalized_rank_score(rec_v1["rank"], n_v1)
                if in_v1
                else 0.0
            )
            score_v2 = (
                calculate_normalized_rank_score(rec_v2["rank"], n_v2)
                if in_v2
                else 0.0
            )
            score_v3 = (
                calculate_normalized_rank_score(rec_v3["rank"], n_v3)
                if in_v3
                else 0.0
            )

            combined_score = score_v1 + score_v2 + score_v3

            original_ranks = []
            if in_v1:
                original_ranks.append(rec_v1["rank"])
            if in_v2:
                original_ranks.append(rec_v2["rank"])
            if in_v3:
                original_ranks.append(rec_v3["rank"])

            avg_original_rank = (
                sum(original_ranks) / len(original_ranks) if original_ranks else 999.0
            )
            num_appearances = len(original_ranks)

            group = determine_priority_group(in_v1, in_v2, in_v3)
            group_priority = self.priority_map.get(group, 999)

            entries.append(
                {
                    "movie_id": mid,
                    "title": title,
                    "year": year,
                    "genres": genres,
                    "in_v1": in_v1,
                    "in_v2": in_v2,
                    "in_v3": in_v3,
                    "v1_rank": rec_v1["rank"] if in_v1 else None,
                    "v2_rank": rec_v2["rank"] if in_v2 else None,
                    "v3_rank": rec_v3["rank"] if in_v3 else None,
                    "v1_score": score_v1,
                    "v2_score": score_v2,
                    "v3_score": score_v3,
                    "combined_score": round(combined_score, 4),
                    "avg_original_rank": round(avg_original_rank, 2),
                    "num_appearances": num_appearances,
                    "priority_group": group,
                    "group_priority": group_priority,
                    "recommended_by": PRIORITY_DISPLAY_NAMES.get(group, group),
                    "model_agreement": f"{num_appearances}/3",
                }
            )

        sorted_entries = sorted(
            entries,
            key=lambda e: (
                e["group_priority"],
                -e["combined_score"],
                e["avg_original_rank"],
                -e["num_appearances"],
                e["title"].strip().lower(),
            ),
        )

        final_results = []
        for final_rank, entry in enumerate(sorted_entries, start=1):
            entry["rank"] = final_rank
            final_results.append(entry)

        if top_k is not None:
            return final_results[:top_k]
        return final_results

