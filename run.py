"""
Main terminal entry point for the Movie Recommendation System.
Provides both an interactive terminal interface and direct CLI execution.
Features automatic dataset and model setup, movie searching, and display of all 4 recommendation versions.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

# Fix Windows console encoding for Unicode characters
if sys.platform == "win32":
    try:
        if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
            sys.stdout.reconfigure(encoding="utf-8")
        if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    V1_MODEL_PATH,
    V2_MODEL_PATH,
    V3_ITEM_MODEL_PATH,
    V3_USER_MODEL_PATH,
    TOP_K,
)
from scripts.download_dataset import check_dataset_exists, download_and_extract
from scripts.train_models import train_all_models
from scripts.evaluate_models import run_evaluation
from src.data_loader import load_all_data
from src.recommender import MovieRecommenderSystem


def check_models_exist() -> bool:
    """Verifies that all precomputed model artifacts exist."""
    return (
        V1_MODEL_PATH.exists()
        and V2_MODEL_PATH.exists()
        and V3_ITEM_MODEL_PATH.exists()
    )


def initialize_system() -> MovieRecommenderSystem:
    """
    Intelligent initialization:
    - Checks dataset -> downloads if missing
    - Checks models -> trains if missing
    - Loads models into memory
    """
    print("=" * 60)
    print("             MOVIE RECOMMENDATION SYSTEM")
    print("=" * 60)

    # 1. Dataset verification
    print("Dataset status:")
    if not check_dataset_exists():
        print("  Dataset missing locally. Downloading MovieLens 1M...")
        if not download_and_extract():
            raise RuntimeError("Failed to prepare MovieLens 1M dataset.")
    print("  ✓ MovieLens 1M dataset found")

    # 2. Models verification
    print("\nModel status:")
    if not check_models_exist():
        print("  Trained models missing. Training models now...")
        if not train_all_models():
            raise RuntimeError("Failed to train recommendation models.")

    # 3. Load models
    recommender_system = MovieRecommenderSystem()
    recommender_system.load_models()
    print("  ✓ Version 1 loaded")
    print("  ✓ Version 2 loaded")
    print("  ✓ Version 3 loaded")
    print("  ✓ Version 4 Union Ranker ready")
    print("-" * 60)

    return recommender_system


def handle_movie_query(
    recommender_system: MovieRecommenderSystem,
    query_str: str,
    top_k: int = TOP_K,
    interactive: bool = True,
) -> bool:
    """
    Searches for movie, resolves multiple matches, generates recommendations across
    all 4 versions, and displays results.
    """
    print(f"\nSearching for '{query_str}'...")
    matches = recommender_system.search_engine.search(query_str)

    if not matches:
        print("\nERROR: Movie not found.")
        print("Please try searching using a shorter or more general title.")
        return False

    selected_movie = None
    if len(matches) == 1:
        selected_movie = matches[0]
        print(f"✓ Movie found: {selected_movie['DisplayTitle']}")
    else:
        print(f"\nFound {len(matches)} matching movies:")
        for idx, m in enumerate(matches[:15], start=1):
            print(f"{idx}. {m['DisplayTitle']} [{m['DisplayGenres']}]")

        if interactive:
            while True:
                try:
                    choice_str = input(
                        f"\nSelect a movie (1-{min(len(matches), 15)}, or 0 to cancel): "
                    ).strip()
                    if choice_str == "0":
                        print("Selection cancelled.")
                        return False
                    choice_idx = int(choice_str)
                    if 1 <= choice_idx <= min(len(matches), 15):
                        selected_movie = matches[choice_idx - 1]
                        break
                    else:
                        print(f"Invalid selection. Enter a number between 1 and {min(len(matches), 15)}.")
                except (ValueError, EOFError, KeyboardInterrupt):
                    print("\nSelection cancelled.")
                    return False
        else:
            # Non-interactive CLI mode: pick best top match
            selected_movie = matches[0]
            print(f"Selecting top match: {selected_movie['DisplayTitle']}")

    print("\n" + "-" * 60)
    print("Generating recommendations across all versions...")
    print("-" * 60)

    rec_result = recommender_system.recommend(
        selected_movie["MovieID"], top_k=top_k
    )

    print("✓ Version 1 complete")
    print("✓ Version 2 complete")
    print("✓ Version 3 complete")
    print("✓ Version 4 complete\n")

    output_str = recommender_system.format_terminal_output(rec_result)
    print(output_str)
    return True


def display_dataset_statistics():
    """Calculates and displays summary statistics of the MovieLens 1M dataset."""
    print("\n" + "=" * 60)
    print("             MOVIELENS 1M DATASET STATISTICS")
    print("=" * 60)
    users_df, movies_df, ratings_df = load_all_data()

    n_users = len(users_df)
    n_movies = len(movies_df)
    n_ratings = len(ratings_df)
    possible_interactions = n_users * n_movies
    sparsity = (1.0 - (n_ratings / possible_interactions)) * 100.0

    print(f"Total Users:              {n_users:,}")
    print(f"Total Movies:             {n_movies:,}")
    print(f"Total Ratings:            {n_ratings:,}")
    print(f"Matrix Sparsity:          {sparsity:.2f}%")
    print(f"Mean Rating:              {ratings_df['Rating'].mean():.2f}")
    print(f"Rating Standard Dev:      {ratings_df['Rating'].std():.2f}")
    print(f"Min Rating:               {ratings_df['Rating'].min():.1f}")
    print(f"Max Rating:               {ratings_df['Rating'].max():.1f}")

    print("\nRating Value Distribution:")
    val_counts = ratings_df["Rating"].value_counts().sort_index(ascending=False)
    for r_val, count in val_counts.items():
        pct = (count / n_ratings) * 100.0
        bar = "█" * int(pct / 2)
        print(f"  {r_val:.1f} stars: {count:7,} ({pct:5.1f}%) | {bar}")

    # Top genres
    all_genres = []
    for g_str in movies_df["Genres"]:
        all_genres.extend(g_str.split("|"))
    genre_series = pd.Series(all_genres).value_counts()
    print(f"\nUnique Genres Count:      {len(genre_series)}")
    print("Top 5 Most Common Genres:")
    for g_name, g_count in genre_series.head(5).items():
        print(f"  - {g_name:<15}: {g_count} movies")
    print("=" * 60 + "\n")


def interactive_menu(recommender_system: MovieRecommenderSystem):
    """Interactive CLI menu loop."""
    while True:
        print("\n" + "=" * 60)
        print("                        MAIN MENU")
        print("=" * 60)
        print("1. Recommend movies")
        print("2. Evaluate models")
        print("3. Show dataset statistics")
        print("4. Retrain models")
        print("5. Exit")
        print("-" * 60)

        try:
            choice = input("Enter choice (1-5): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting application. Goodbye!")
            break

        if choice == "1":
            try:
                query = input("\nEnter movie name (or 'back' to return): ").strip()
                if query.lower() in ["back", "exit", "q", ""]:
                    continue
                handle_movie_query(
                    recommender_system, query, top_k=TOP_K, interactive=True
                )
            except (EOFError, KeyboardInterrupt):
                continue
        elif choice == "2":
            run_evaluation()
        elif choice == "3":
            display_dataset_statistics()
        elif choice == "4":
            print("\nRetraining all models...")
            train_all_models()
            recommender_system.load_models()
            print("Models reloaded successfully.")
        elif choice == "5":
            print("\nThank you for using Movie Recommendation System. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")


def main():
    parser = argparse.ArgumentParser(
        description="Terminal-Based Movie Recommendation System (MovieLens 1M)"
    )
    parser.add_argument(
        "--movie",
        "-m",
        type=str,
        help="Movie title to search and recommend directly.",
    )
    parser.add_argument(
        "--top_k",
        "-k",
        type=int,
        default=TOP_K,
        help=f"Number of recommendations per version (default: {TOP_K}).",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run model evaluation suite directly.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Display dataset statistics directly.",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain all model artifacts.",
    )

    args = parser.parse_args()

    # Direct task flags
    if args.evaluate:
        run_evaluation()
        return

    if args.stats:
        display_dataset_statistics()
        return

    if args.retrain:
        train_all_models()
        return

    # Initialize system
    system = initialize_system()

    if args.movie:
        # Direct CLI movie query
        handle_movie_query(
            system, args.movie, top_k=args.top_k, interactive=False
        )
    else:
        # Interactive loop
        interactive_menu(system)


if __name__ == "__main__":
    main()
