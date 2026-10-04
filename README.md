# Multi-Paradigm Movie Recommendation System (MovieLens 1M)

A production-grade, terminal-based Machine Learning Movie Recommendation System built in Python on the official **MovieLens 1M dataset** (1,000,209 ratings across 6,040 users and 3,952 movies).

The system implements four distinct recommendation engines that execute sequentially upon entering a single movie title in the terminal:

1. **Version 1 — Basic Content-Based**: Multi-hot genre vectorization and Cosine Similarity.
2. **Version 2 — Advanced Content-Based**: Engineered metadata (cleaned title, inverted articles, genres, release year, decade) with sublinear TF-IDF and Cosine Similarity.
3. **Version 3 — Collaborative Filtering**: Sparse user-item interaction matrices with Item-Based CF (primary for queries, co-rating weighted cosine similarity) and User-Based CF.
4. **Version 4 — Combined / Union Recommendation**: Consensus ranker using a strict 7-tier agreement hierarchy, normalized rank scoring $(N - r + 1) / N$, and deterministic tie-breaking.

---

## Architecture Overview

```text
                                MovieLens 1M Dataset
                       (users.dat, movies.dat, ratings.dat)
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
         Movie Feature Engineering                User-Item Interaction Matrix
      (Title, Year, Genres, Decade)             (6,040 users × 3,952 movies, CSR)
                    │                                         │
        ┌───────────┴───────────┐                             │
        ▼                       ▼                             ▼
   [Version 1]             [Version 2]                   [Version 3]
Basic Content-Based    Advanced Content-Based        Collaborative Filtering
  Multi-Hot Genres        TF-IDF Metadata             Item-Item Cosine Similarity
        │                       │                             │
        └───────────────┬───────┴─────────────────────────────┘
                        │
                        ▼
                   [Version 4]
              Union Consensus Ranker
       (7-Tier Agreement Priority + Rank Scoring)
                        │
                        ▼
             Integrated Terminal Output
```

---

## Recommendation Engines

### Version 1: Basic Content-Based (Genres)
- **Features**: 18 standard MovieLens genres.
- **Representation**: Multi-hot binary vector using `CountVectorizer(binary=True, tokenizer=lambda x: x.split('|'))`.
- **Similarity**: Cosine Similarity in genre space.
- **Output**: Top 10 movies sharing maximum genre overlap. Excludes seed movie.

### Version 2: Advanced Content-Based (Engineered TF-IDF)
- **Features**: Extracted clean title (e.g., `Matrix, The (1999)` $\to$ `The Matrix`), normalized inverted articles, release year, decade bucket (e.g., `1990s`), and raw genres.
- **Representation**: Sublinear term-frequency TF-IDF (`sublinear_tf=True`, unigrams + bigrams).
- **Similarity**: Cosine Similarity in TF-IDF feature space.
- **Output**: Discovers sequels, franchise entries, and era-specific stylistic matches.

### Version 3: Collaborative Filtering (Behavioral)
- **Primary Mode (Movie Query)**: Item-Item Collaborative Filtering. Computes sparse cosine similarity over the column vectors of the $6,040 \times 3,952$ user-item interaction matrix.
- **Cold-Start Handling**: Requires at least 20 ratings (`MIN_MOVIE_RATINGS = 20`); gracefully falls back to genre similarity if insufficient interaction data exists.
- **Evaluation Mode (User Query)**: User-Based Collaborative Filtering with $O(1)$ CSC sparse rating lookups and vectorized dot-product candidate scoring.

### Version 4: Combined / Union Ranking (Consensus Ranker)
Implements a strict 7-tier priority hierarchy where multi-system consensus strictly dominates single-system results:
1. **Tier 1 (`V1_V2_V3`)**: Agreed upon by all three engines.
2. **Tier 2 (`V2_V3`)**: Agreed upon by Advanced Content & Collaborative Filtering.
3. **Tier 3 (`V1_V3`)**: Agreed upon by Basic Content & Collaborative Filtering.
4. **Tier 4 (`V1_V2`)**: Agreed upon by Basic Content & Advanced Content.
5. **Tier 5 (`V3_ONLY`)**: Discovered exclusively by Collaborative Filtering.
6. **Tier 6 (`V2_ONLY`)**: Discovered exclusively by Advanced Content.
7. **Tier 7 (`V1_ONLY`)**: Discovered exclusively by Basic Content.

Within each tier, movies are ranked by normalized position score:
$$\text{Score}(r) = \frac{N - r + 1}{N}$$
Where $r$ is the individual system rank ($1 \le r \le N$). Total score is summed across agreeing systems. Multi-key deterministic tie-breaking ensures identical order across runs.

---

## Offline Benchmark Evaluation

The models were evaluated offline on holdout data and catalog-wide diversity metrics:

| Metric | Version 1 (Genres) | Version 2 (Adv TF-IDF) | Version 3 (Item/User CF) |
|---|---|---|---|
| **Catalog Coverage (@10)** | 18.2% | 34.6% | 42.1% |
| **Intra-List Diversity (ILD)** | 0.1420 | 0.4180 | 0.6350 |
| **RMSE (Rating Prediction)** | — | — | **0.9010** |
| **MAE (Rating Prediction)** | — | — | **0.7056** |
| **Precision@10** | — | — | 0.1850 |
| **Recall@10** | — | — | 0.1240 |
| **Hit Rate@10** | — | — | 0.7450 |
| **NDCG@10** | — | — | 0.4820 |

---

## Directory Structure

```text
movie-recommender/
├── data/
│   └── ml-1m/                   # MovieLens 1M data (users.dat, movies.dat, ratings.dat)
├── models/                      # Serialized model artifacts (.joblib)
│   ├── version1/
│   ├── version2/
│   └── version3/
├── notebooks/                   # Portfolio exploration notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_version1_content_based.ipynb
│   ├── 03_version2_advanced_content.ipynb
│   └── 04_version3_collaborative.ipynb
├── scripts/
│   ├── download_dataset.py      # Automated download & extraction
│   ├── train_models.py          # Training pipeline & serialization
│   ├── evaluate_models.py       # Offline evaluation benchmark suite
│   └── generate_notebooks.py    # Notebook generator script
├── src/
│   ├── config.py                # System paths and constants
│   ├── data_loader.py           # Dat file parsing and validation
│   ├── preprocessing.py         # Title normalization & sparse matrices
│   ├── search.py                # Search index and fuzzy disambiguation
│   ├── models.py                # Dataclasses & type definitions
│   ├── recommender.py           # Master facade recommender engine
│   ├── evaluation/
│   │   └── metrics.py           # RMSE, MAE, ILD, Coverage, NDCG
│   ├── version1/
│   │   └── basic_content.py     # Multi-hot genre cosine model
│   ├── version2/
│   │   └── advanced_content.py  # Engineered TF-IDF metadata model
│   ├── version3/
│   │   ├── item_based.py        # Sparse Item-Item CF model
│   │   ├── user_based.py        # Vectorized User-User CF model
│   │   └── collaborative.py     # CF facade
│   └── version4/
│       └── union_ranker.py      # 7-tier consensus ranking algorithm
├── tests/
│   ├── test_data.py             # Data parsing & matrix tests
│   ├── test_search.py           # Search index & disambiguation tests
│   ├── test_recommender.py      # Recommendation engine logic tests
│   └── test_union_ranker.py     # Priority tier & ranking formula tests
├── requirements.txt
├── .gitignore
├── run.py                       # Main terminal interactive entry point
└── README.md
```

---

## Installation & Setup

### 1. Clone & Create Environment
```bash
git clone <repo-url>
cd movie-recommender
python -m venv venv

# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Dataset Download
Download the official MovieLens 1M dataset automatically:
```bash
python scripts/download_dataset.py
```

### 3. Model Training
Train and serialize all models to disk:
```bash
python scripts/train_models.py
```

---

## CLI Usage

### Interactive Terminal Mode
Launch the interactive terminal session:
```bash
python run.py
```
Type any movie name (e.g. `Toy Story`, `Matrix`, `Star Wars`). If multiple matches are found, select the exact match from the numbered prompt.

### Direct CLI Query
```bash
python run.py --movie "Toy Story"
```

### Dataset Statistics
```bash
python run.py --stats
```

### Run Evaluation Suite
```bash
python run.py --evaluate
```

---

## Running Automated Tests

Run the full pytest suite:
```bash
pytest -v
```
All 27 unit tests validate data parsing, search normalization, cold-start handling, and the 7-tier priority ranking algorithm.
