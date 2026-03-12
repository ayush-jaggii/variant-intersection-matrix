"""
Manual validation of variant relationships and research fertility analysis.
==========================================================================
Supports multiple authors, inter-rater reliability, and research opportunities.
"""

import logging
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

from config.settings import CONCEPTUAL_VALIDATION_FILE

logger = logging.getLogger(__name__)

class ConceptualValidator:
    """Handles multi-author conceptual validation and reliability metrics."""

    def __init__(self, variant_names: List[str]):
        self.variant_names = sorted(variant_names)
        self.authors: List[str] = ["Author 1", "Author 2", "Author 3"]
        self.ratings: Dict[str, Dict[str, str]] = {}
        self._load_data()

    def _get_key(self, v1: str, v2: str) -> str:
        """Standardize key as 'alpha|beta' where alpha < beta."""
        v_a, v_b = sorted([v1, v2])
        return f"{v_a}|{v_b}"

    def update_config(self, author_names: List[str]):
        """Update author list and persist."""
        self.authors = author_names
        self._save_data()

    def set_rating(self, v1: str, v2: str, author: str, rating: str):
        """Set a single rating."""
        if rating not in ["R", "N", "?"]:
            raise ValueError("Rating must be 'R', 'N', or '?'")
        key = self._get_key(v1, v2)
        if key not in self.ratings:
            self.ratings[key] = {}
        self.ratings[key][author] = rating
        self._save_data()

    def get_ratings(self, v1: str, v2: str) -> Dict[str, str]:
        """Return all author ratings for a pair."""
        return self.ratings.get(self._get_key(v1, v2), {})

    def get_majority_rating(self, v1: str, v2: str) -> str:
        """Combine ratings using majority vote rule."""
        r_dict = self.get_ratings(v1, v2)
        vals = [r_dict.get(a, "?") for a in self.authors]
        r_count = vals.count("R")
        n_count = vals.count("N")
        total = len(self.authors)
        if r_count > total / 2: return "R"
        if n_count > total / 2: return "N"
        return "?"

    def compute_reliability(self) -> float:
        """Compute nominal Krippendorff's Alpha."""
        pairs = self.get_all_pairs()
        if not pairs or len(self.authors) < 2: return 0.0
        mapping = {"R": 0, "N": 1, "?": 2}
        coincidence = np.zeros((3, 3))
        for v1, v2 in pairs:
            r_dict = self.get_ratings(v1, v2)
            vals = [r_dict.get(a, "?") for a in self.authors]
            counts = np.zeros(3)
            for v in vals: counts[mapping[v]] += 1
            ni = np.sum(counts)
            if ni < 2: continue
            for j in range(3):
                for l in range(3):
                    if j == l: coincidence[j, l] += counts[j] * (counts[j] - 1) / (ni - 1)
                    else: coincidence[j, l] += counts[j] * counts[l] / (ni - 1)
        total_n = np.sum(coincidence)
        if total_n <= 1: return 0.0
        obs_dis = np.sum(coincidence) - np.trace(coincidence)
        nj = np.sum(coincidence, axis=1)
        exp_dis = np.sum(np.outer(nj, nj)) - np.sum(nj*nj)
        if exp_dis == 0: return 1.0
        alpha = 1 - (total_n - 1) * (obs_dis / exp_dis)
        return round(float(alpha), 3)

    def get_all_pairs(self) -> List[Tuple[str, str]]:
        """Return all lower-triangular pairs."""
        n = len(self.variant_names)
        return [(self.variant_names[i], self.variant_names[j]) for i in range(n) for j in range(i)]

    def bulk_update(self, df: pd.DataFrame):
        """Update from DF with 'Variant A', 'Variant B' and author columns."""
        for _, row in df.iterrows():
            v1, v2 = row.get("Variant A"), row.get("Variant B")
            if not v1 or not v2: v1, v2 = row.get("Variant_A"), row.get("Variant_B")
            if not v1 or not v2: continue
            key = self._get_key(str(v1), str(v2))
            if key not in self.ratings: self.ratings[key] = {}
            for auth in self.authors:
                if auth in row:
                    val = str(row[auth]).strip().upper()
                    if val in ["R", "N", "?"]: self.ratings[key][auth] = val
        self._save_data()

    def _load_data(self):
        if CONCEPTUAL_VALIDATION_FILE.exists():
            try:
                with open(CONCEPTUAL_VALIDATION_FILE, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, dict) and "authors" in raw:
                    self.authors = raw.get("authors", self.authors)
                    self.ratings = raw.get("ratings", {})
                else:
                    self.authors = ["Author 1"]
                    self.ratings = {k: {"Author 1": v} for k, v in raw.items() if isinstance(v, str)}
            except: self.ratings = {}

    def _save_data(self):
        try:
            with open(CONCEPTUAL_VALIDATION_FILE, "w", encoding="utf-8") as f:
                json.dump({"authors": self.authors, "ratings": self.ratings}, f, indent=4)
        except: pass
