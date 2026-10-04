"""
Central configuration module for the Movie Recommendation System.
Contains all paths, hyperparameters, priority rules, and global constants.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "ml-1m"
MODEL_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Version Specific Model Directories
V1_MODEL_DIR = MODEL_DIR / "version1"
V2_MODEL_DIR = MODEL_DIR / "version2"
V3_MODEL_DIR = MODEL_DIR / "version3"

# Dataset Files & URLs
DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
USERS_FILE = DATA_DIR / "users.dat"
MOVIES_FILE = DATA_DIR / "movies.dat"
RATINGS_FILE = DATA_DIR / "ratings.dat"
REQUIRED_DATA_FILES = [USERS_FILE, MOVIES_FILE, RATINGS_FILE]

# Model Artifact Paths
V1_MODEL_PATH = V1_MODEL_DIR / "basic_content_model.joblib"
V2_MODEL_PATH = V2_MODEL_DIR / "advanced_content_model.joblib"
V3_ITEM_MODEL_PATH = V3_MODEL_DIR / "item_based_cf_model.joblib"
V3_USER_MODEL_PATH = V3_MODEL_DIR / "user_based_cf_model.joblib"

# Aliases for compatibility
VERSION1_MODEL_PATH = V1_MODEL_PATH
VERSION2_MODEL_PATH = V2_MODEL_PATH
VERSION3_ITEM_MODEL_PATH = V3_ITEM_MODEL_PATH
VERSION3_USER_MODEL_PATH = V3_USER_MODEL_PATH

# Hyperparameters & Settings
TOP_K = 10
MIN_MOVIE_RATINGS = 20  # Minimum rating threshold for candidate pool in Item-based CF
RANDOM_STATE = 42
TEST_SPLIT_RATIO = 0.20

# Version 4 Exact Priority Hierarchy (Requirement 18 & 23)
# 1. V1 + V2 + V3
# 2. V2 + V3
# 3. V1 + V3
# 4. V1 + V2
# 5. V3 only
# 6. V2 only
# 7. V1 only
PRIORITY_ORDER = [
    "V1_V2_V3",
    "V2_V3",
    "V1_V3",
    "V1_V2",
    "V3",
    "V2",
    "V1",
]

# Priority Display Names
PRIORITY_DISPLAY_NAMES = {
    "V1_V2_V3": "V1 + V2 + V3",
    "V2_V3": "V2 + V3",
    "V1_V3": "V1 + V3",
    "V1_V2": "V1 + V2",
    "V3": "V3",
    "V2": "V2",
    "V1": "V1",
}

# Encodings
DATASET_ENCODING = "latin-1"  # MovieLens 1M uses ISO-8859-1/Latin-1

