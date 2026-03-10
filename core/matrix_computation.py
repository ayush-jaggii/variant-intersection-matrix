"""
Matrix Computation Module
=========================

Builds and operates on the two core matrices:

    1. Paper × Variant binary matrix  (rows = papers, cols = variants)
       → Shows which variants appear in which papers.
       → Papers are identified by sequential IDs: P1, P2, P3, ...

    2. Variant × Variant intersection matrix  (symmetric)
       → Cell (i, j) = number of papers that mention BOTH variant i AND variant j.
       → CRITICAL: variants from the SAME dimension are NOT paired.
         E.g., "Energy Consuming" × "Non Energy Consuming" (both under
         "Product Energy Consumption") is skipped — such a cell is set to -1
         (or NaN in the DataFrame) to indicate an invalid/excluded pair.
       → Computed efficiently as M.T @ M with dimension masking applied.

    3. Pair details:  for every valid cross-dimension pair, lists the
       intersection count and the supporting papers.

How Dimension Exclusion Works:
    Each variant belongs to a dimension (e.g., "Product Energy Consumption").
    The dimension_map dict maps variant_name → dimension_name.

    When building the intersection matrix, any pair (v1, v2) where
    dimension_map[v1] == dimension_map[v2] is marked as excluded.
    This is because variants within the same dimension are mutually
    exclusive categories — pairing them is logically invalid.

Manual Validation (Single Variant):
    Researchers can override auto-detection for individual variant-paper
    cells.  Overrides are stored as {paper_id: {variant: bool}} in
    manual_overrides.json and applied to the paper-variant matrix before
    the intersection matrix is computed.

Manual Validation (Variant Pairs):
    Researchers can also confirm that a specific paper discusses a
    combination of two variants.  These pair overrides are stored as
    {paper_id: [["variant_a", "variant_b"], ...]} in pair_overrides.json.

    When pair overrides exist, they affect the matrices as follows:
      1. Both individual variants are forced to "present" (1) in the
         paper-variant matrix for that paper.
      2. The intersection matrix is recomputed to reflect the updated
         binary matrix, so the pair count increases accordingly.

    This ensures the exported CSVs (paper_variant_matrix.csv,
    variant_intersection_matrix.csv, pair_details.csv) always include
    manual validations — no separate re-run is needed.

Live Updates:
    When a user saves a new override (single or pair), the method
    `apply_overrides_and_recompute()` is called to:
      1. Apply all single-variant overrides to the paper-variant matrix
      2. Apply all pair overrides (force both variants to present)
      3. Recompute the intersection matrix
      4. Re-export CSVs
    This keeps all views and downloads consistent without requiring
    a full re-run of the analysis.
"""

import logging
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from config.settings import (
    OUTPUT_DIR,
    PAPER_VARIANT_MATRIX_CSV,
    VARIANT_INTERSECTION_MATRIX_CSV,
    PAIR_DETAILS_CSV,
    MANUAL_OVERRIDES_FILE,
    PAIR_OVERRIDES_FILE,
)
from utils.helpers import load_json, save_json
import streamlit as st

logger = logging.getLogger(__name__)

# Sentinel value used in the intersection matrix for same-dimension pairs.
# These cells are excluded from counting, gaps analysis, etc.
EXCLUDED_PAIR_VALUE = -1
SAME_VARIANT_VALUE = -2
LOWER_TRIANGLE_VALUE = -3

@st.cache_data(show_spinner=False)
def _cached_build_base_matrix(
    detection_results: Dict[str, Dict[str, bool]], variant_names: List[str]
) -> pd.DataFrame:
    """Build the raw paper x variant matrix, cached to avoid recomputation."""
    rows = {}
    for paper_id, variant_presence in detection_results.items():
        rows[paper_id] = {v: int(variant_presence.get(v, False)) for v in variant_names}

    df = pd.DataFrame.from_dict(rows, orient="index", columns=variant_names)
    df.index.name = "paper_id"
    return df.sort_index()


@st.cache_data(show_spinner=False)
def _cached_compute_intersection(
    paper_variant_df: pd.DataFrame, dimension_map: Dict[str, str]
) -> pd.DataFrame:
    """Compute the intersection matrix and apply dimension masking, cached for speed."""
    M = paper_variant_df.values.astype(np.int32)
    raw_intersection = M.T @ M

    variant_names = list(paper_variant_df.columns)
    result = pd.DataFrame(
        raw_intersection.astype(float),
        index=variant_names,
        columns=variant_names,
    )

    n = len(variant_names)
    for i in range(n):
        for j in range(n):
            if i == j:
                result.iloc[i, j] = SAME_VARIANT_VALUE
            elif j < i:
                result.iloc[i, j] = LOWER_TRIANGLE_VALUE
            else:
                vi = variant_names[i]
                vj = variant_names[j]
                if dimension_map:
                    dim_i = dimension_map.get(vi, "")
                    dim_j = dimension_map.get(vj, "")
                    if dim_i and dim_j and dim_i == dim_j:
                        result.iloc[i, j] = EXCLUDED_PAIR_VALUE
    return result


class MatrixComputer:
    """
    Builds the paper-variant binary matrix and computes the intersection matrix.

    Supports two kinds of manual overrides:
        1. Single-variant overrides:  {paper_id: {variant: bool}}
           → Force a variant to present/absent for a specific paper.

        2. Pair overrides:  {paper_id: [["variant_a", "variant_b"], ...]}
           → Confirm that a specific paper discusses both variants together.
           → Implicitly forces both variants to "present" for that paper.

    Both override types are persisted to disk and automatically included
    in all matrix computations and CSV exports.

    Attributes:
        paper_variant_df:        DataFrame with papers as rows, variants as columns.
        intersection_df:         Symmetric DataFrame of variant-pair intersection counts.
                                 Same-dimension cells contain EXCLUDED_PAIR_VALUE (-1).
        dimension_map:           Dict mapping variant_name → dimension_name.
        _detection_results_raw:  Original detection results (before overrides).
    """

    def __init__(self):
        self.paper_variant_df: Optional[pd.DataFrame] = None
        self.intersection_df: Optional[pd.DataFrame] = None
        self.dimension_map: Dict[str, str] = {}
        self._detection_results_raw: Optional[Dict[str, Dict[str, bool]]] = None
        self._variant_names: List[str] = []
        self._detection_details: Dict[str, Dict[str, str]] = {}
        self._manual_overrides: Dict[str, Dict[str, bool]] = {}
        self._pair_overrides: Dict[str, List[List[str]]] = {}
        self._load_overrides()
        self._load_pair_overrides()

    # ── Public API ───────────────────────────────────────────────────────

    def build_paper_variant_matrix(
        self,
        detection_results: Dict[str, Dict[str, bool]],
        variant_names: List[str],
        detection_details: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> pd.DataFrame:
        """
        Build the Paper × Variant binary matrix from detection results.

        Args:
            detection_results: {paper_id: {variant_name: bool}}.
            variant_names: Ordered list of variant names (columns).

        Returns:
            DataFrame with paper_ids (P1,P2,...) as index and variant_names
            as columns. Values are 0 or 1.
        """
        logger.info(
            "Building paper–variant matrix: %d papers × %d variants",
            len(detection_results), len(variant_names),
        )

        # Store raw results so we can re-apply overrides later without
        # cumulative drift
        self._detection_results_raw = detection_results
        self._variant_names = variant_names
        if detection_details is not None:
            self._detection_details = detection_details

        # Use cached function for the heavy base computation
        df = _cached_build_base_matrix(detection_results, variant_names)

        # Apply manual overrides (researcher corrections)
        # We apply these ON TOP of the cached result, so overrides are never cached
        # but re-runs without changes are extremely fast.
        df = self._apply_overrides(df.copy())
        # Apply pair overrides (force both variants to present)
        df = self._apply_pair_overrides(df)

        self.paper_variant_df = df
        logger.info("Paper-variant matrix shape: %s", df.shape)
        return df

    def compute_intersection_matrix(
        self,
        paper_variant_df: Optional[pd.DataFrame] = None,
        dimension_map: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Compute the Variant × Variant intersection matrix.

        Algorithm:
            1. Compute the raw intersection via matrix multiplication: M.T @ M
            2. Apply dimension masking: set cells where both variants share
               the same dimension to EXCLUDED_PAIR_VALUE (-1).

        The diagonal values represent the total papers for each variant.

        Args:
            paper_variant_df: Optional; uses stored matrix if not provided.
            dimension_map:    Dict mapping variant_name → dimension_name.
                              If None, uses the stored map (no exclusion if empty).

        Returns:
            Symmetric DataFrame of intersection counts.
            Same-dimension cells are set to -1 (excluded).
        """
        if paper_variant_df is not None:
            self.paper_variant_df = paper_variant_df
        if dimension_map is not None:
            self.dimension_map = dimension_map

        if self.paper_variant_df is None:
            raise ValueError("Paper-variant matrix has not been built yet.")

        # Use cached function for the heavy intersection computation
        result = _cached_compute_intersection(
            self.paper_variant_df, 
            self.dimension_map or {}
        )

        self.intersection_df = result
        logger.info("Intersection matrix computed: %s", result.shape)
        return result

    def apply_overrides_and_recompute(self):
        """
        Re-apply all overrides and recompute both matrices from the
        original detection results.

        This is the method called after saving a new manual override
        (single-variant or pair) to get a live update without re-running
        the full analysis.  It also re-exports CSVs.

        Flow:
            1. Rebuild paper-variant matrix from raw detection results
            2. Apply single-variant overrides
            3. Apply pair overrides (force both variants to present)
            4. Recompute intersection matrix with dimension masking
            5. Re-export all CSV files

        This ensures all views, drill-downs, and downloads reflect
        the latest manual validations immediately.
        """
        if self._detection_results_raw is None or not self._variant_names:
            logger.warning("Cannot recompute — no raw detection results stored")
            return

        # Step 1-3: Rebuild paper-variant with overrides
        self.build_paper_variant_matrix(
            self._detection_results_raw, self._variant_names, self._detection_details
        )
        # Step 4: Recompute intersection
        self.compute_intersection_matrix()
        # Step 5: Re-export
        self.export_all()
        logger.info("Matrices recomputed and CSVs re-exported after override change")

    def get_papers_for_pair(
        self,
        variant_a: str,
        variant_b: str,
    ) -> List[str]:
        """
        Get the list of paper IDs that mention both variant_a and variant_b.

        Args:
            variant_a: First variant name.
            variant_b: Second variant name.

        Returns:
            List of paper_id strings (P1, P2, ...).
        """
        if self.paper_variant_df is None:
            return []

        df = self.paper_variant_df
        if variant_a not in df.columns or variant_b not in df.columns:
            return []

        mask = (df[variant_a] == 1) & (df[variant_b] == 1)
        return list(df[mask].index)

    def get_papers_for_variant(self, variant_name: str) -> List[str]:
        """Get all paper IDs that mention a specific variant."""
        if self.paper_variant_df is None:
            return []
        if variant_name not in self.paper_variant_df.columns:
            return []
        mask = self.paper_variant_df[variant_name] == 1
        return list(self.paper_variant_df[mask].index)

    def is_same_dimension_pair(self, variant_a: str, variant_b: str) -> bool:
        """
        Check if two variants belong to the same dimension.

        Same-dimension pairs are excluded from the intersection matrix.

        Args:
            variant_a: First variant name.
            variant_b: Second variant name.

        Returns:
            True if both variants share the same (non-empty) dimension.
        """
        if not self.dimension_map:
            return False
        dim_a = self.dimension_map.get(variant_a, "")
        dim_b = self.dimension_map.get(variant_b, "")
        return bool(dim_a and dim_b and dim_a == dim_b)

    def get_research_gaps(self) -> List[Tuple[str, str]]:
        """
        Identify research gaps: CROSS-DIMENSION variant pairs
        with zero intersection.

        Same-dimension pairs (marked as -1) are excluded — they are
        not gaps, they are structurally invalid comparisons.

        Returns:
            List of (variant_a, variant_b) tuples where no paper covers both.
        """
        if self.intersection_df is None:
            return []

        gaps = []
        variants = list(self.intersection_df.columns)
        for i, va in enumerate(variants):
            for j in range(i + 1, len(variants)):
                vb = variants[j]
                value = self.intersection_df.iloc[i, j]
                # Skip excluded pairs (same dimension) and only
                # report genuine zeros
                if value == 0:
                    gaps.append((va, vb))

        logger.info("Found %d research gaps (zero cross-dimension pairs)", len(gaps))
        return gaps

    def generate_pair_details(self) -> pd.DataFrame:
        """
        Generate a flat DataFrame with one row per valid variant pair,
        showing the intersection count and the supporting papers.

        Same-dimension pairs are excluded from this listing.

        Columns: dimension_a, variant_a, dimension_b, variant_b,
                 intersection_count, supporting_papers

        Returns:
            DataFrame with pair details.
        """
        if self.paper_variant_df is None or self.intersection_df is None:
            raise ValueError("Matrices not computed yet.")

        variant_names = list(self.intersection_df.columns)
        rows = []

        for i, va in enumerate(variant_names):
            for j in range(i + 1, len(variant_names)):
                vb = variant_names[j]
                count = int(self.intersection_df.iloc[i, j])

                # Skip same-dimension pairs (marked as -1)
                if count == EXCLUDED_PAIR_VALUE:
                    continue

                papers = self.get_papers_for_pair(va, vb) if count > 0 else []
                dim_a = self.dimension_map.get(va, "Uncategorized")
                dim_b = self.dimension_map.get(vb, "Uncategorized")
                rows.append({
                    "dimension_a": dim_a,
                    "variant_a": va,
                    "dimension_b": dim_b,
                    "variant_b": vb,
                    "intersection_count": count,
                    "supporting_papers": "; ".join(papers),
                })

        return pd.DataFrame(rows)

    def generate_detection_details(self) -> pd.DataFrame:
        """
        Generate a DataFrame showing the specific alternate name terms detected 
        for each paper and variant.
        """
        rows = []
        if self._detection_details:
            for paper_id, details in self._detection_details.items():
                for variant_key, term in details.items():
                    rows.append({
                        "paper_id": paper_id,
                        "variant": variant_key,
                        "detected_term": term
                    })
        return pd.DataFrame(rows)

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Return summary statistics about the matrices.

        Stats include:
            • total_papers, total_variants, total_detections
            • avg_variants_per_paper, avg_papers_per_variant
            • variants_never_detected
            • total_valid_pairs (cross-dimension only)
            • research_gaps, covered_pairs, max_intersection
            • excluded_pairs (same-dimension count)
        """
        stats: Dict[str, Any] = {}

        if self.paper_variant_df is not None:
            df = self.paper_variant_df
            stats["total_papers"] = len(df)
            stats["total_variants"] = len(df.columns)
            stats["total_detections"] = int(df.values.sum())
            stats["avg_variants_per_paper"] = float(df.sum(axis=1).mean())
            stats["avg_papers_per_variant"] = float(df.sum(axis=0).mean())
            stats["variants_never_detected"] = int((df.sum(axis=0) == 0).sum())

        if self.intersection_df is not None:
            n = len(self.intersection_df)
            total_all_pairs = n * (n - 1) // 2

            # Count same-dimension excluded pairs and genuine zeros
            excluded_count = 0
            zero_count = 0
            for i in range(n):
                for j in range(i + 1, n):
                    val = self.intersection_df.iloc[i, j]
                    if val == EXCLUDED_PAIR_VALUE:
                        excluded_count += 1
                    elif val == 0:
                        zero_count += 1

            valid_pairs = total_all_pairs - excluded_count
            stats["total_all_pairs"] = total_all_pairs
            stats["excluded_pairs"] = excluded_count
            stats["total_valid_pairs"] = valid_pairs
            stats["research_gaps"] = zero_count
            stats["covered_pairs"] = valid_pairs - zero_count

            # Max intersection (ignoring excluded cells and diagonal)
            values = self.intersection_df.values.copy()
            np.fill_diagonal(values, 0)
            values[values == EXCLUDED_PAIR_VALUE] = 0
            stats["max_intersection"] = int(np.triu(values, k=1).max()) if n > 1 else 0

        return stats

    # ── Manual Overrides (Single Variant) ─────────────────────────────────

    def set_override(self, paper_id: str, variant_name: str, value: bool):
        """
        Manually override a paper-variant detection result.

        After saving, `apply_overrides_and_recompute()` should be called
        to get a live update of both matrices and CSV exports.

        Args:
            paper_id: Paper identifier (P1, P2, ...).
            variant_name: Variant name.
            value: True = variant present, False = absent.
        """
        if paper_id not in self._manual_overrides:
            self._manual_overrides[paper_id] = {}
        self._manual_overrides[paper_id][variant_name] = value
        self._save_overrides()
        logger.info("Override set: %s × %s = %s", paper_id, variant_name, value)

    def clear_override(self, paper_id: str, variant_name: str):
        """Remove a manual override."""
        if paper_id in self._manual_overrides:
            self._manual_overrides[paper_id].pop(variant_name, None)
            if not self._manual_overrides[paper_id]:
                del self._manual_overrides[paper_id]
            self._save_overrides()

    def get_overrides(self) -> Dict[str, Dict[str, bool]]:
        """Return all manual overrides."""
        return self._manual_overrides.copy()

    # ── Manual Overrides (Variant Pairs) ──────────────────────────────────
    #
    # How Pair Validation Works:
    #   When a researcher confirms that paper P12 discusses *both*
    #   "Product Leasing" and "Recycling" together, we store:
    #       pair_overrides["P12"] = [["Product Leasing", "Recycling"]]
    #
    #   During matrix computation:
    #     1. Both "Product Leasing" and "Recycling" are forced to 1 for P12
    #        in the paper-variant matrix.
    #     2. When the intersection matrix is recomputed (M.T @ M), cell
    #        (Product Leasing, Recycling) increases by 1.
    #     3. Because the binary matrix is the source of truth, ALL derived
    #        outputs (CSVs, pair details, research gaps) automatically
    #        reflect the pair validation.
    #
    #   This approach avoids maintaining a separate "pair override counter"
    #   and keeps the single paper-variant matrix as the sole data source.

    def set_pair_override(
        self,
        paper_id: str,
        variant_a: str,
        variant_b: str,
    ):
        """
        Manually validate that a paper discusses a pair of variants.

        This forces both variants to "present" for the given paper and
        recomputes the intersection matrix so the pair count increases.

        Pair overrides are persisted to pair_overrides.json and survive
        across sessions and re-runs.

        Args:
            paper_id:  Paper identifier (P1, P2, ...).
            variant_a: First variant name.
            variant_b: Second variant name.
        """
        if paper_id not in self._pair_overrides:
            self._pair_overrides[paper_id] = []

        # Avoid duplicates (normalize order for comparison)
        pair = sorted([variant_a, variant_b])
        for existing in self._pair_overrides[paper_id]:
            if sorted(existing) == pair:
                logger.info("Pair override already exists: %s × %s in %s",
                            variant_a, variant_b, paper_id)
                return

        self._pair_overrides[paper_id].append([variant_a, variant_b])
        self._save_pair_overrides()
        logger.info("Pair override set: %s × %s in %s", variant_a, variant_b, paper_id)

    def remove_pair_override(
        self,
        paper_id: str,
        variant_a: str,
        variant_b: str,
    ):
        """Remove a specific pair override."""
        if paper_id not in self._pair_overrides:
            return

        pair = sorted([variant_a, variant_b])
        self._pair_overrides[paper_id] = [
            p for p in self._pair_overrides[paper_id]
            if sorted(p) != pair
        ]
        if not self._pair_overrides[paper_id]:
            del self._pair_overrides[paper_id]
        self._save_pair_overrides()

    def get_pair_overrides(self) -> Dict[str, List[List[str]]]:
        """Return all pair overrides."""
        return self._pair_overrides.copy()

    def clear_all_pair_overrides(self):
        """Remove all pair overrides."""
        self._pair_overrides = {}
        self._save_pair_overrides()

    # ── Export ────────────────────────────────────────────────────────────
    #
    # CSV exports always reflect the current state of the matrices,
    # which already include manual overrides (both single-variant and
    # pair overrides).  No extra step is needed — the overrides are
    # baked into paper_variant_df before the intersection is computed.

    def export_all(self, output_dir: Path = OUTPUT_DIR):
        """
        Export all CSV files to the output directory.

        Files generated:
            • paper_variant_matrix.csv  (includes manual overrides)
            • variant_intersection_matrix.csv  (includes manual overrides)
            • pair_details.csv  (excludes same-dimension pairs, includes overrides)

        All files reflect the current matrix state, which already
        incorporates both single-variant and pair manual validations.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.paper_variant_df is not None:
            path = output_dir / PAPER_VARIANT_MATRIX_CSV
            self.paper_variant_df.to_csv(path)
            logger.info("Exported: %s", path)

        if self.intersection_df is not None:
            path = output_dir / VARIANT_INTERSECTION_MATRIX_CSV
            # Replace -1 with "EXCLUDED" in the CSV for clarity
            export_df = self.intersection_df.copy()
            export_df = export_df.replace(EXCLUDED_PAIR_VALUE, "EXCLUDED")
            export_df.to_csv(path)
            logger.info("Exported: %s", path)

            pair_df = self.generate_pair_details()
            path = output_dir / PAIR_DETAILS_CSV
            pair_df.to_csv(path, index=False)
            logger.info("Exported: %s", path)
            
            details_df = self.generate_detection_details()
            if not details_df.empty:
                path = output_dir / "detection_details.csv"
                details_df.to_csv(path, index=False)
                logger.info("Exported: %s", path)

    # ── Internal ─────────────────────────────────────────────────────────

    def _apply_overrides(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply single-variant manual overrides to the paper-variant matrix."""
        for paper_id, overrides in self._manual_overrides.items():
            if paper_id in df.index:
                for variant_name, value in overrides.items():
                    if variant_name in df.columns:
                        df.at[paper_id, variant_name] = int(value)
        return df

    def _apply_pair_overrides(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply pair overrides to the paper-variant matrix.

        How this works:
            For each pair override {paper_id: [["A", "B"]]}, we set
            df.at[paper_id, "A"] = 1  and  df.at[paper_id, "B"] = 1.

            By forcing both variants to "present" in the binary matrix,
            the intersection (M.T @ M) naturally picks up the pair.
            No separate intersection adjustment is needed.
        """
        for paper_id, pairs in self._pair_overrides.items():
            if paper_id in df.index:
                for pair in pairs:
                    for variant in pair:
                        if variant in df.columns:
                            df.at[paper_id, variant] = 1
        return df

    def _load_overrides(self):
        """Load single-variant manual overrides from disk."""
        try:
            self._manual_overrides = load_json(MANUAL_OVERRIDES_FILE)
        except (FileNotFoundError, Exception):
            self._manual_overrides = {}

    def _save_overrides(self):
        """Persist single-variant manual overrides to disk."""
        save_json(self._manual_overrides, MANUAL_OVERRIDES_FILE)

    def _load_pair_overrides(self):
        """Load pair validation overrides from disk."""
        try:
            self._pair_overrides = load_json(PAIR_OVERRIDES_FILE)
        except (FileNotFoundError, Exception):
            self._pair_overrides = {}

    def _save_pair_overrides(self):
        """Persist pair validation overrides to disk."""
        save_json(self._pair_overrides, PAIR_OVERRIDES_FILE)
