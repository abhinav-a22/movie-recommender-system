"""
Evaluation module providing rigorous metrics for Content-Based and Collaborative Filtering models:
- Content-Based: Catalog Coverage, Intra-List Diversity (ILD), Similarity Distribution.
- Collaborative Filtering: RMSE, MAE, Precision@K, Recall@K, Hit Rate@K, NDCG@K.
"""

from typing import Dict, Any, List, Set, Union
import numpy as np
import pandas as pd


def compute_genre_jaccard_distance(genres_a: Set[str], genres_b: Set[str]) -> float:
    """Computes Jaccard distance 1 - (|A & B| / |A | B|) between two genre sets."""
    if not genres_a or not genres_b:
        return 1.0
    intersection = len(genres_a & genres_b)
    union = len(genres_a | genres_b)
    if union == 0:
        return 1.0
    return 1.0 - (intersection / union)


def evaluate_content_model(
    recommender: Any,
    movies_df: pd.DataFrame,
    sample_size: int = 100,
    top_k: int = 10,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Evaluates Content-Based models:
    - Catalog Coverage: percentage of catalog movies recommended across queries.
    - Intra-List Diversity (ILD): average pairwise genre Jaccard distance in top-K.
    - Similarity Score Distribution: min, mean, max, std.
    """
    rng = np.random.RandomState(random_state)
    available_ids = movies_df["MovieID"].values
    if len(available_ids) > sample_size:
        sample_ids = rng.choice(available_ids, size=sample_size, replace=False)
    else:
        sample_ids = available_ids

    genre_map = {}
    for _, row in movies_df.iterrows():
        g_raw = str(row.get("Genres", ""))
        genre_map[int(row["MovieID"])] = set(g_raw.split("|"))

    all_recommended_ids = set()
    all_scores = []
    diversity_scores = []

    for mid in sample_ids:
        try:
            recs = recommender.recommend(int(mid), top_k=top_k)
        except Exception:
            continue

        if not recs:
            continue

        rec_mids = [r["movie_id"] for r in recs]
        all_recommended_ids.update(rec_mids)
        all_scores.extend([r["similarity_score"] for r in recs])

        if len(rec_mids) > 1:
            pair_distances = []
            for i in range(len(rec_mids)):
                for j in range(i + 1, len(rec_mids)):
                    g_i = genre_map.get(rec_mids[i], set())
                    g_j = genre_map.get(rec_mids[j], set())
                    pair_distances.append(compute_genre_jaccard_distance(g_i, g_j))
            if pair_distances:
                diversity_scores.append(np.mean(pair_distances))

    total_catalog = len(movies_df)
    catalog_coverage = (
        (len(all_recommended_ids) / total_catalog) * 100.0 if total_catalog > 0 else 0.0
    )
    avg_diversity = float(np.mean(diversity_scores)) if diversity_scores else 0.0
    scores_arr = np.array(all_scores) if all_scores else np.array([0.0])

    return {
        "catalog_coverage_pct": round(catalog_coverage, 2),
        "unique_movies_recommended": len(all_recommended_ids),
        "total_catalog_size": total_catalog,
        "intra_list_diversity": round(avg_diversity, 4),
        "similarity_min": round(float(np.min(scores_arr)), 4),
        "similarity_mean": round(float(np.mean(scores_arr)), 4),
        "similarity_max": round(float(np.max(scores_arr)), 4),
        "similarity_std": round(float(np.std(scores_arr)), 4),
    }


def evaluate_cf_rating_prediction(
    user_model: Any,
    test_ratings: pd.DataFrame,
    max_eval_samples: int = 2500,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Evaluates rating prediction accuracy on a holdout test split:
    - Root Mean Squared Error (RMSE)
    - Mean Absolute Error (MAE)
    """
    rng = np.random.RandomState(random_state)
    if len(test_ratings) > max_eval_samples:
        sample_indices = rng.choice(len(test_ratings), size=max_eval_samples, replace=False)
        eval_df = test_ratings.iloc[sample_indices]
    else:
        eval_df = test_ratings

    y_true = []
    y_pred = []

    for _, row in eval_df.iterrows():
        uid = int(row["UserID"])
        mid = int(row["MovieID"])
        true_r = float(row["Rating"])

        pred_r = user_model.predict_rating(uid, mid)
        y_true.append(true_r)
        y_pred.append(pred_r)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "samples_evaluated": len(eval_df),
    }


def evaluate_ranking_metrics(
    user_model: Any,
    test_ratings: pd.DataFrame,
    top_k: int = 10,
    relevance_threshold: float = 4.0,
    max_users: int = 100,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Evaluates Top-K ranking performance on holdout test interactions:
    - Precision@K: fraction of top-K recommendations that are relevant in test set.
    - Recall@K: fraction of test relevant items captured in top-K recommendations.
    - Hit Rate@K: fraction of users for whom at least one recommended item is relevant.
    - NDCG@K: Normalized Discounted Cumulative Gain at position K.
    """
    relevant_test = (
        test_ratings[test_ratings["Rating"] >= relevance_threshold]
        .groupby("UserID")["MovieID"]
        .apply(set)
        .to_dict()
    )

    rng = np.random.RandomState(random_state)
    all_test_users = list(relevant_test.keys())
    if len(all_test_users) > max_users:
        eval_users = rng.choice(all_test_users, size=max_users, replace=False)
    else:
        eval_users = all_test_users

    precisions = []
    recalls = []
    hit_rates = []
    ndcgs = []

    discounts = 1.0 / np.log2(np.arange(2, top_k + 2))

    for uid in eval_users:
        true_relevant = relevant_test.get(uid, set())
        if not true_relevant:
            continue

        try:
            recs = user_model.recommend_for_user(int(uid), top_k=top_k)
        except Exception:
            continue

        if not recs:
            continue

        recommended_ids = [r["movie_id"] for r in recs]
        hits = [1 if mid in true_relevant else 0 for mid in recommended_ids]
        num_hits = sum(hits)

        prec = num_hits / top_k
        precisions.append(prec)

        rec = num_hits / len(true_relevant) if len(true_relevant) > 0 else 0.0
        recalls.append(rec)

        hit_rates.append(1.0 if num_hits > 0 else 0.0)

        dcg = np.sum(np.array(hits) * discounts[: len(hits)])

        ideal_hits = min(len(true_relevant), top_k)
        idcg = np.sum(discounts[:ideal_hits])

        ndcg = (dcg / idcg) if idcg > 0 else 0.0
        ndcgs.append(ndcg)

    return {
        f"precision@{top_k}": round(float(np.mean(precisions)), 4) if precisions else 0.0,
        f"recall@{top_k}": round(float(np.mean(recalls)), 4) if recalls else 0.0,
        f"hit_rate@{top_k}": round(float(np.mean(hit_rates)), 4) if hit_rates else 0.0,
        f"ndcg@{top_k}": round(float(np.mean(ndcgs)), 4) if ndcgs else 0.0,
        "users_evaluated": len(precisions),
    }

