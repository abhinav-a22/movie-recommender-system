"""Tests for Version 4 Union / Combined Recommendation Ranker."""

from __future__ import annotations

import pytest

from src.models import RecommendationResult
from src.version4.union_ranker import PriorityGroup, UnionRanker


def make_dummy_rec(
    movie_id: int, title: str, rank: int, score: float = 0.8
) -> RecommendationResult:
    return RecommendationResult(
        movie_id=movie_id,
        title=title,
        year=2000,
        genres="Action",
        similarity_score=score,
        rank=rank,
        clean_title=title,
    )


def test_priority_tier_dominance():
    ranker = UnionRanker()

    # Movie 1 is in all three (V1, V2, V3) but ranked 10 in all
    v1_recs = [make_dummy_rec(1, "Three Systems Movie", 10, 0.1)]
    v2_recs = [make_dummy_rec(1, "Three Systems Movie", 10, 0.1)]
    v3_recs = [make_dummy_rec(1, "Three Systems Movie", 10, 0.1)]

    # Movie 2 is only in V1, ranked #1 with score 1.0
    v1_recs.append(make_dummy_rec(2, "Single System Movie", 1, 1.0))

    union_results = ranker.rank(v1_recs, v2_recs, v3_recs, top_n=5)

    assert len(union_results) == 2
    # Tier 1 (V1_V2_V3) MUST beat Tier 7 (V1) despite lower individual rank/score
    assert union_results[0].movie_id == 1
    assert union_results[0].priority_group == PriorityGroup.V1_V2_V3.value
    assert union_results[1].movie_id == 2
    assert union_results[1].priority_group == PriorityGroup.V1_ONLY.value


def test_exact_seven_tier_ordering():
    ranker = UnionRanker()

    # 1: V1+V2+V3
    # 2: V2+V3
    # 3: V1+V3
    # 4: V1+V2
    # 5: V3 only
    # 6: V2 only
    # 7: V1 only
    v1 = [
        make_dummy_rec(1, "M1", 1),
        make_dummy_rec(3, "M3", 2),
        make_dummy_rec(4, "M4", 3),
        make_dummy_rec(7, "M7", 4),
    ]
    v2 = [
        make_dummy_rec(1, "M1", 1),
        make_dummy_rec(2, "M2", 2),
        make_dummy_rec(4, "M4", 3),
        make_dummy_rec(6, "M6", 4),
    ]
    v3 = [
        make_dummy_rec(1, "M1", 1),
        make_dummy_rec(2, "M2", 2),
        make_dummy_rec(3, "M3", 3),
        make_dummy_rec(5, "M5", 4),
    ]

    results = ranker.rank(v1, v2, v3, top_n=10)
    result_ids = [r.movie_id for r in results]

    assert result_ids == [1, 2, 3, 4, 5, 6, 7]
    assert results[0].priority_group == PriorityGroup.V1_V2_V3.value
    assert results[1].priority_group == PriorityGroup.V2_V3.value
    assert results[2].priority_group == PriorityGroup.V1_V3.value
    assert results[3].priority_group == PriorityGroup.V1_V2.value
    assert results[4].priority_group == PriorityGroup.V3_ONLY.value
    assert results[5].priority_group == PriorityGroup.V2_ONLY.value
    assert results[6].priority_group == PriorityGroup.V1_ONLY.value


def test_normalized_rank_scoring_formula():
    ranker = UnionRanker()
    # N = 10
    # rank 1 score = (10 - 1 + 1) / 10 = 1.0
    # rank 5 score = (10 - 5 + 1) / 10 = 0.6
    # rank 10 score = (10 - 10 + 1) / 10 = 0.1
    assert pytest.approx(ranker.calculate_normalized_score(1, 10)) == 1.0
    assert pytest.approx(ranker.calculate_normalized_score(5, 10)) == 0.6
    assert pytest.approx(ranker.calculate_normalized_score(10, 10)) == 0.1

