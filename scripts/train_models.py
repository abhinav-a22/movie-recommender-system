"""
Training script for the Movie Recommendation System.
Preprocesses MovieLens 1M and builds/persists:
- Version 1: Basic Content-Based Model
- Version 2: Advanced Content-Based Model
- Version 3: Collaborative Filtering Models (Item-Based & User-Based)
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
)
from scripts.download_dataset import check_dataset_exists, download_and_extract
from src.data_loader import load_all_data
from src.preprocessing import preprocess_movies, preprocess_ratings
from src.version1.basic_content import BasicContentRecommender
from src.version2.advanced_content import AdvancedContentRecommender
from src.version3.item_based import ItemBasedCF
from src.version3.user_based import UserBasedCF


def train_all_models(force_download: bool = False):
    """
    Orchestrates end-to-end data loading, preprocessing, model training, and persistence.
    """
    total_start = time.time()
    print("=" * 60)
    print("      TRAINING MOVIELENS 1M RECOMMENDATION MODELS           ")
    print("=" * 60)

    # 1. Ensure dataset exists
    if not check_dataset_exists():
        print("Dataset not found locally. Initiating download...")
        success = download_and_extract(force=force_download)
        if not success:
            print("[ERROR] Could not download dataset. Aborting training.")
            return False
    else:
        print("[OK] MovieLens 1M dataset verified.")

    # 2. Load dataset
    print("\n[1/5] Loading dataset files (users.dat, movies.dat, ratings.dat)...")
    t0 = time.time()
    users_df, movies_df, ratings_df = load_all_data()
    print(
        f"      Loaded {len(users_df):,} users, {len(movies_df):,} movies, "
        f"{len(ratings_df):,} ratings in {time.time() - t0:.2f}s."
    )

    # 3. Preprocessing
    print("\n[2/5] Cleaning and feature engineering metadata & ratings...")
    t0 = time.time()
    movies_preprocessed = preprocess_movies(movies_df)
    ratings_preprocessed = preprocess_ratings(ratings_df)
    print(f"      Feature engineering complete in {time.time() - t0:.2f}s.")

    # 4. Train Version 1 (Basic Content-Based)
    print("\n[3/5] Building Version 1: Basic Content-Based (Genre Vectorization)...")
    t0 = time.time()
    v1_model = BasicContentRecommender()
    v1_model.fit(movies_preprocessed)
    v1_model.save(V1_MODEL_PATH)
    print(
        f"      Version 1 saved to {V1_MODEL_PATH.name} in {time.time() - t0:.2f}s."
    )

    # 5. Train Version 2 (Advanced Content-Based)
    print("\n[4/5] Building Version 2: Advanced Content-Based (TF-IDF Metadata)...")
    t0 = time.time()
    v2_model = AdvancedContentRecommender()
    v2_model.fit(movies_preprocessed)
    v2_model.save(V2_MODEL_PATH)
    print(
        f"      Version 2 saved to {V2_MODEL_PATH.name} in {time.time() - t0:.2f}s."
    )

    # 6. Train Version 3 (Collaborative Filtering: Item-based & User-based)
    print("\n[5/5] Building Version 3: Collaborative Filtering (Sparse Interaction Matrices)...")
    t0 = time.time()
    v3_item = ItemBasedCF()
    v3_item.fit(movies_preprocessed, ratings_preprocessed)
    v3_item.save(V3_ITEM_MODEL_PATH)

    v3_user = UserBasedCF()
    v3_user.fit(movies_preprocessed, ratings_preprocessed)
    v3_user.save(V3_USER_MODEL_PATH)
    print(
        f"      Version 3 (Item & User models) saved to models/version3/ in {time.time() - t0:.2f}s."
    )

    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"[OK] ALL MODELS SUCCESSFULLY TRAINED AND PERSISTED in {total_time:.2f}s!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = train_all_models()
    sys.exit(0 if success else 1)
