"""
Evaluation script for the Movie Recommendation System.
Performs offline evaluation of Content-Based (V1, V2) and Collaborative Filtering (V3).
Outputs comprehensive metrics including Coverage, Diversity, RMSE, MAE, Precision@K, Recall@K, Hit Rate@K, and NDCG@K.
"""

import sys
import time
from pathlib import Path

# Fix Windows console encoding for Unicode
if sys.platform == "win32":
    try:
        if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
            sys.stdout.reconfigure(encoding="utf-8")
        if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    V1_MODEL_PATH,
    V2_MODEL_PATH,
    V3_ITEM_MODEL_PATH,
    V3_USER_MODEL_PATH,
    TOP_K,
    RANDOM_STATE,
)
from src.data_loader import load_all_data
from src.preprocessing import (
    preprocess_movies,
    preprocess_ratings,
    train_test_split_ratings,
)
from src.version1.basic_content import BasicContentRecommender
from src.version2.advanced_content import AdvancedContentRecommender
from src.version3.user_based import UserBasedCF
from src.evaluation.metrics import (
    evaluate_content_model,
    evaluate_cf_rating_prediction,
    evaluate_ranking_metrics,
)


def run_evaluation():
    print("=" * 65)
    print("      MOVIELENS 1M RECOMMENDATION SYSTEM — MODEL EVALUATION     ")
    print("=" * 65)
    print(f"Random State: {RANDOM_STATE} | Top-K: {TOP_K}")
    print("Loading datasets and trained models...")

    users_df, movies_df, ratings_df = load_all_data()
    movies_clean = preprocess_movies(movies_df)
    ratings_clean = preprocess_ratings(ratings_df)

    # 1. Evaluate Content-Based Models (V1 and V2)
    print("\n" + "-" * 65)
    print("1. EVALUATING CONTENT-BASED RECOMMENDATION MODELS")
    print("-" * 65)

    v1_model = BasicContentRecommender.load(V1_MODEL_PATH)
    v2_model = AdvancedContentRecommender.load(V2_MODEL_PATH)

    print("Evaluating Version 1 (Basic Content-Based - Genres)...")
    v1_metrics = evaluate_content_model(
        v1_model, movies_clean, sample_size=100, top_k=TOP_K, random_state=RANDOM_STATE
    )

    print("Evaluating Version 2 (Advanced Content-Based - TF-IDF Metadata)...")
    v2_metrics = evaluate_content_model(
        v2_model, movies_clean, sample_size=100, top_k=TOP_K, random_state=RANDOM_STATE
    )

    print("\n=== CONTENT-BASED COMPARISON TABLE ===")
    print(f"{'Metric':<30} | {'Version 1 (Basic)':<15} | {'Version 2 (Advanced)':<15}")
    print("-" * 65)
    print(
        f"{'Catalog Coverage (%)':<30} | {v1_metrics['catalog_coverage_pct']:<15.2f} | {v2_metrics['catalog_coverage_pct']:<15.2f}"
    )
    print(
        f"{'Unique Movies Recommended':<30} | {v1_metrics['unique_movies_recommended']:<15} | {v2_metrics['unique_movies_recommended']:<15}"
    )
    print(
        f"{'Intra-List Diversity (ILD)':<30} | {v1_metrics['intra_list_diversity']:<15.4f} | {v2_metrics['intra_list_diversity']:<15.4f}"
    )
    print(
        f"{'Mean Similarity Score':<30} | {v1_metrics['similarity_mean']:<15.4f} | {v2_metrics['similarity_mean']:<15.4f}"
    )
    print(
        f"{'Similarity Std Dev':<30} | {v1_metrics['similarity_std']:<15.4f} | {v2_metrics['similarity_std']:<15.4f}"
    )
    print(
        f"{'Similarity Score Range':<30} | {v1_metrics['similarity_min']:.2f} - {v1_metrics['similarity_max']:.2f}     | {v2_metrics['similarity_min']:.2f} - {v2_metrics['similarity_max']:.2f}"
    )

    # 2. Collaborative Filtering Train/Test Evaluation
    print("\n" + "-" * 65)
    print("2. EVALUATING COLLABORATIVE FILTERING (TRAIN/TEST SPLIT: 80/20)")
    print("-" * 65)
    print("Splitting ratings (avoiding data leakage)...")
    train_ratings, test_ratings = train_test_split_ratings(
        ratings_clean, test_ratio=0.20, random_state=RANDOM_STATE
    )
    print(f"Train ratings: {len(train_ratings):,} | Test ratings: {len(test_ratings):,}")

    print("Fitting evaluation User-Based CF model on train ratings...")
    eval_cf = UserBasedCF(k_neighbors=40)
    eval_cf.fit(movies_clean, train_ratings)

    print("Computing rating prediction metrics (RMSE, MAE) on test set...")
    pred_metrics = evaluate_cf_rating_prediction(
        eval_cf, test_ratings, max_eval_samples=2500, random_state=RANDOM_STATE
    )

    print("Computing Top-K ranking metrics (Precision@K, Recall@K, Hit Rate@K, NDCG@K)...")
    rank_metrics = evaluate_ranking_metrics(
        eval_cf,
        test_ratings,
        top_k=TOP_K,
        relevance_threshold=4.0,
        max_users=150,
        random_state=RANDOM_STATE,
    )

    print("\n=== COLLABORATIVE FILTERING METRICS TABLE ===")
    print(f"{'Evaluation Category':<30} | {'Metric':<18} | {'Score':<10}")
    print("-" * 65)
    print(f"{'Rating Prediction Accuracy':<30} | {'RMSE':<18} | {pred_metrics['rmse']:<10.4f}")
    print(f"{'Rating Prediction Accuracy':<30} | {'MAE':<18} | {pred_metrics['mae']:<10.4f}")
    print(f"{'Top-' + str(TOP_K) + ' Ranking Quality':<30} | {'Precision@' + str(TOP_K):<18} | {rank_metrics['precision@' + str(TOP_K)]:<10.4f}")
    print(f"{'Top-' + str(TOP_K) + ' Ranking Quality':<30} | {'Recall@' + str(TOP_K):<18} | {rank_metrics['recall@' + str(TOP_K)]:<10.4f}")
    print(f"{'Top-' + str(TOP_K) + ' Ranking Quality':<30} | {'Hit Rate@' + str(TOP_K):<18} | {rank_metrics['hit_rate@' + str(TOP_K)]:<10.4f}")
    print(f"{'Top-' + str(TOP_K) + ' Ranking Quality':<30} | {'NDCG@' + str(TOP_K):<18} | {rank_metrics['ndcg@' + str(TOP_K)]:<10.4f}")
    print("=" * 65)
    print("[OK] EVALUATION COMPLETE!")
    print("=" * 65)


if __name__ == "__main__":
    run_evaluation()
