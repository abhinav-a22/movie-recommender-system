"""
Terminal-based movie search engine.
Supports exact matching, case-insensitive matching, partial matching,
token overlap, and title normalization (e.g., handling inverted articles).
"""

import re
from typing import List, Dict, Any, Optional
import pandas as pd


def normalize_string(text: str) -> str:
    """Lowercases, removes special characters, and trims extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


class MovieSearchEngine:
    """
    Search engine for MovieLens movies.
    Provides multi-tiered matching:
    1. Exact raw or normalized title match
    2. Exact clean title match
    3. Title prefix match
    4. Substring match
    5. Token overlap match
    """

    def __init__(self, movies_df: pd.DataFrame):
        self.movies_df = movies_df.copy()
        self._build_index()

    def _build_index(self):
        """Build normalized strings and tokens for fast querying."""
        self.records = []
        for _, row in self.movies_df.iterrows():
            movie_id = int(row["MovieID"])
            raw_title = str(row["Title"])
            clean_title = str(row.get("CleanTitle", raw_title))
            norm_title = str(row.get("NormalizedTitle", clean_title))
            disp_title = str(row.get("DisplayTitle", raw_title))
            disp_genres = str(row.get("DisplayGenres", row.get("Genres", "")))
            year = row.get("Year", None)

            # Normalized tokens
            norm_tokens = set(normalize_string(norm_title).split())
            clean_tokens = set(normalize_string(clean_title).split())
            raw_tokens = set(normalize_string(raw_title).split())
            all_tokens = norm_tokens | clean_tokens | raw_tokens

            self.records.append(
                {
                    "MovieID": movie_id,
                    "Title": raw_title,
                    "CleanTitle": clean_title,
                    "NormalizedTitle": norm_title,
                    "DisplayTitle": disp_title,
                    "DisplayGenres": disp_genres,
                    "Year": year,
                    "norm_title_str": normalize_string(norm_title),
                    "clean_title_str": normalize_string(clean_title),
                    "raw_title_str": normalize_string(raw_title),
                    "all_tokens": all_tokens,
                }
            )

    def search(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Searches movies matching the query.
        Returns a sorted list of matched movie dictionaries.
        """
        if not query or not query.strip():
            return []

        q_clean = query.strip()
        q_norm = normalize_string(q_clean)
        q_tokens = set(q_norm.split())

        tier1 = []  # Exact matches
        tier2 = []  # Prefix matches
        tier3 = []  # Substring matches
        tier4 = []  # All query tokens contained
        tier5 = []  # Any query tokens contained (if no other matches)

        seen_ids = set()

        for rec in self.records:
            mid = rec["MovieID"]
            r_norm = rec["norm_title_str"]
            r_clean = rec["clean_title_str"]
            r_raw = rec["raw_title_str"]
            r_tokens = rec["all_tokens"]

            # Tier 1: Exact match
            if q_norm == r_norm or q_norm == r_clean or q_norm == r_raw:
                tier1.append(rec)
                seen_ids.add(mid)
                continue

            # Tier 2: Prefix match
            if r_norm.startswith(q_norm) or r_clean.startswith(q_norm):
                tier2.append(rec)
                seen_ids.add(mid)
                continue

            # Tier 3: Substring match
            if q_norm in r_norm or q_norm in r_clean or q_norm in r_raw:
                tier3.append(rec)
                seen_ids.add(mid)
                continue

            # Tier 4: All query tokens present in movie tokens
            if q_tokens and q_tokens.issubset(r_tokens):
                tier4.append(rec)
                seen_ids.add(mid)
                continue

        # Sort tiers: shorter title length first (closer match)
        tier1.sort(key=lambda x: len(x["norm_title_str"]))
        tier2.sort(key=lambda x: len(x["norm_title_str"]))
        tier3.sort(key=lambda x: len(x["norm_title_str"]))
        tier4.sort(key=lambda x: len(x["norm_title_str"]))

        matches = tier1 + tier2 + tier3 + tier4

        # If still empty, try partial token match (at least half of tokens match)
        if not matches and len(q_tokens) > 1:
            for rec in self.records:
                mid = rec["MovieID"]
                if mid in seen_ids:
                    continue
                overlap = len(q_tokens & rec["all_tokens"])
                if overlap >= (len(q_tokens) / 2):
                    tier5.append((overlap, rec))
            tier5.sort(key=lambda x: (-x[0], len(x[1]["norm_title_str"])))
            matches = [item[1] for item in tier5]

        return matches[:limit]

    def get_movie_by_id(self, movie_id: int) -> Optional[Dict[str, Any]]:
        """Looks up a movie record directly by MovieID."""
        for rec in self.records:
            if rec["MovieID"] == movie_id:
                return rec
        return None

