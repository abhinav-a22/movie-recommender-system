#!/usr/bin/env python3
"""Script to generate professional Jupyter Notebooks for portfolio presentation.

Generates:
1. `notebooks/01_data_exploration.ipynb`
2. `notebooks/02_version1_content_based.ipynb`
3. `notebooks/03_version2_advanced_content.ipynb`
4. `notebooks/04_version3_collaborative.ipynb`
"""

from __future__ import annotations

import json
import os
import sys

# Ensure root directory is on Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOKS_DIR = os.path.join(ROOT_DIR, "notebooks")


def make_notebook(cells: list[dict]) -> dict:
    """Create a minimal valid Jupyter Notebook dict structure."""
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def make_markdown_cell(source: str) -> dict:
    """Create a markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")],
    }


def make_code_cell(source: str) -> dict:
    """Create a code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")],
    }


def generate_all_notebooks() -> None:
    """Generate all 4 portfolio exploration and walkthrough notebooks."""
    os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

    # 1. Data Exploration Notebook
    nb1_cells = [
        make_markdown_cell("# MovieLens 1M: Comprehensive Exploratory Data Analysis\n\nThis notebook explores the users, movies, and ratings distributions in MovieLens 1M."),
        make_code_cell("import sys; sys.path.insert(0, '..')\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom src.data_loader import DataLoader\n\nloader = DataLoader()\nusers, movies, ratings = loader.load_all()\nprint(f'Users: {len(users)}, Movies: {len(movies)}, Ratings: {len(ratings)}')"),
        make_code_cell("ratings['Rating'].value_counts().sort_index().plot(kind='bar', color='skyblue', edgecolor='black')\nplt.title('MovieLens 1M Rating Distribution')\nplt.xlabel('Rating (1-5)')\nplt.ylabel('Count')\nplt.show()"),
    ]
    with open(os.path.join(NOTEBOOKS_DIR, "01_data_exploration.ipynb"), "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb1_cells), f, indent=2)

    # 2. Version 1 Notebook
    nb2_cells = [
        make_markdown_cell("# Version 1: Basic Content-Based Recommender\n\nUses multi-hot genre representations and Cosine Similarity to find similar movies."),
        make_code_cell("import sys; sys.path.insert(0, '..')\nfrom src.recommender import MovieRecommenderEngine\n\nengine = MovieRecommenderEngine()\nengine.load_or_train()\nrecs = engine.v1_model.recommend(movie_id=1, top_n=10) # Toy Story\nfor r in recs:\n    print(f'{r.rank}. {r.title} ({r.year}) - Score: {r.similarity_score:.4f} | Genres: {r.genres}')"),
    ]
    with open(os.path.join(NOTEBOOKS_DIR, "02_version1_content_based.ipynb"), "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb2_cells), f, indent=2)

    # 3. Version 2 Notebook
    nb3_cells = [
        make_markdown_cell("# Version 2: Advanced Content-Based Recommender\n\nUses engineered textual representations (cleaned titles, genres, release year, decade) with sublinear TF-IDF and Cosine Similarity."),
        make_code_cell("import sys; sys.path.insert(0, '..')\nfrom src.recommender import MovieRecommenderEngine\n\nengine = MovieRecommenderEngine()\nengine.load_or_train()\nrecs = engine.v2_model.recommend(movie_id=1, top_n=10)\nfor r in recs:\n    print(f'{r.rank}. {r.title} ({r.year}) - Score: {r.similarity_score:.4f} | Genres: {r.genres}')"),
    ]
    with open(os.path.join(NOTEBOOKS_DIR, "03_version2_advanced_content.ipynb"), "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb3_cells), f, indent=2)

    # 4. Version 3 Notebook
    nb4_cells = [
        make_markdown_cell("# Version 3: Collaborative Filtering Recommender\n\nLearns from user interaction patterns using Item-Based CF (sparse cosine similarity) and User-Based CF."),
        make_code_cell("import sys; sys.path.insert(0, '..')\nfrom src.recommender import MovieRecommenderEngine\n\nengine = MovieRecommenderEngine()\nengine.load_or_train()\nrecs = engine.v3_item_cf.recommend(movie_id=1, top_n=10)\nfor r in recs:\n    print(f'{r.rank}. {r.title} ({r.year}) - Score: {r.similarity_score:.4f} | Co-ratings: {r.co_rating_count}')"),
    ]
    with open(os.path.join(NOTEBOOKS_DIR, "04_version3_collaborative.ipynb"), "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb4_cells), f, indent=2)

    print("Successfully generated all 4 Jupyter notebooks.")


if __name__ == "__main__":
    generate_all_notebooks()

