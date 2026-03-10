"""
Variant Detection Module
========================

Detects the presence of predefined variants (and their alternate_names) in
preprocessed paper texts.

Strategy:
    • Variants are organized by dimension:
        {
          "Product Energy Consumption": {
            "Energy Consuming": ["energy consuming", "energy-intensive"],
            "Non Energy Consuming": ["manual product", "non powered"]
          }
        }

    • Pre-compile a search index: unique_key → list of normalized search terms
      (the variant name itself + all alternate_names).  Normalization uses the
      same TextPreprocessor.preprocess_variant_term() as paper text.

    • Unique keys: if two variants share the same name but belong to
      different dimensions (e.g., "Immediate Fulfillment" in two dims),
      they are disambiguated as "Immediate Fulfillment (Dim A)" and
      "Immediate Fulfillment (Dim B)".  Otherwise the plain name is used.

    • For each paper text, check each term via substring matching.
      A variant is "present" if any of its terms appears at least
      `presence_threshold` times.

    • Return a binary presence dict per paper:
      {paper_id: {unique_key: True/False}}

Dimension metadata is preserved so the matrix computation module
can skip same-dimension pairs when building the VIM.
"""

import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from config.settings import PRESENCE_THRESHOLD, CASE_INSENSITIVE, VARIANTS_FILE
from core.preprocessing import TextPreprocessor
from utils.helpers import (
    load_json,
    save_json,
    dimensions_to_flat_list,
    flat_list_to_dimensions,
)

logger = logging.getLogger(__name__)


class VariantDetector:
    """
    Detects variant presence across a corpus of preprocessed texts.

    Supports the new dimension-aware variant structure while remaining
    backwards-compatible with legacy flat lists.

    Handles duplicate variant names across dimensions by generating
    unique display keys (appending " (DimName)" when collisions exist).

    Attributes:
        variants:     Flat list of variant dicts, each with "dimension", "name", "alternate_names".
        preprocessor: TextPreprocessor for normalizing variant terms.
        threshold:    Minimum occurrences to consider a variant "present".
    """

    def __init__(
        self,
        variants: Optional[List[Dict[str, Any]]] = None,
        preprocessor: Optional[TextPreprocessor] = None,
        threshold: int = PRESENCE_THRESHOLD,
    ):
        self.preprocessor = preprocessor or TextPreprocessor()
        self.threshold = threshold

        # Load variants (flat list with dimension info)
        if variants is not None:
            self.variants = self._ensure_dimension_fields(variants)
        else:
            self.variants = self._load_variants()

        # Generate unique keys for each variant (handles name collisions)
        self._unique_keys: List[str] = []
        self._key_to_variant: Dict[str, Dict[str, Any]] = {}
        self._generate_unique_keys()

        # Build search index: unique_key → list of preprocessed search terms
        self._search_index: Dict[str, List[Tuple[str, re.Pattern]]] = {}
        self._build_search_index()

    # ── Public API ───────────────────────────────────────────────────────

    def detect_all(
        self,
        preprocessed_texts: Dict[str, str],
        progress_callback=None,
    ) -> Tuple[Dict[str, Dict[str, bool]], Dict[str, Dict[str, str]]]:
        """
        Detect variants in all papers.

        Args:
            preprocessed_texts: {paper_id: preprocessed_text}.
            progress_callback:  Optional callable(current, total).

        Returns:
            Nested dict: {paper_id: {unique_key: True/False}}.
        """
        total = len(preprocessed_texts)
        logger.info(
            "Detecting %d variants across %d papers",
            len(self.variants), total,
        )

        results: Dict[str, Dict[str, bool]] = {}
        detected_details: Dict[str, Dict[str, str]] = {}
        for idx, (paper_id, text) in enumerate(preprocessed_texts.items()):
            pres, dets = self.detect_in_text(text)
            results[paper_id] = pres
            detected_details[paper_id] = dets
            if progress_callback:
                progress_callback(idx + 1, total)

        return results, detected_details

    def detect_in_text(self, text: str) -> Tuple[Dict[str, bool], Dict[str, str]]:
        """
        Check which variants are present in a single preprocessed text.

        Uses phrase-based substring matching on normalized text.
        The match is deterministic and consistent because both
        paper text and alternate_name terms pass through the same normalization.

        Args:
            text: Preprocessed text of a paper.

        Returns:
            Dict mapping each unique variant key to a boolean presence flag.
        """
        presence: Dict[str, bool] = {}
        details: Dict[str, str] = {}
        for unique_key, terms in self._search_index.items():
            found, term = self._check_terms(text, terms)
            presence[unique_key] = found
            if found and term:
                details[unique_key] = term
        return presence, details

    def get_variant_names(self) -> List[str]:
        """
        Return ordered list of unique variant keys.

        These are guaranteed unique — if two variants share the same name
        but belong to different dimensions, they are disambiguated:
            "Immediate Fulfillment (Dimension A)"
            "Immediate Fulfillment (Dimension B)"
        """
        return list(self._unique_keys)

    def get_dimension_map(self) -> Dict[str, str]:
        """
        Return a mapping of unique_key → dimension_name.

        This is used by MatrixComputer to know which variants belong
        to the same dimension (and thus should NOT be paired in the VIM).

        Returns:
            Dict mapping each unique variant key to its parent dimension.
        """
        return {
            key: self._key_to_variant[key].get("dimension", "Uncategorized")
            for key in self._unique_keys
        }

    def get_variant_details(self, unique_key: str) -> Optional[Dict[str, Any]]:
        """Return full definition for a named variant by its unique key."""
        return self._key_to_variant.get(unique_key)

    def count_occurrences(self, text: str, unique_key: str) -> int:
        """
        Count total occurrences of a variant (all alternate_name terms) in text.

        Useful for the UI to show how strongly a variant is represented.
        """
        terms = self._search_index.get(unique_key, [])
        count = 0
        for term in terms:
            if term:
                pattern = re.compile(re.escape(term))
                count += len(pattern.findall(text))
        return count

    def reload_variants(self, variants: Optional[List[Dict[str, Any]]] = None):
        """Reload variant definitions and rebuild search index."""
        if variants is not None:
            self.variants = self._ensure_dimension_fields(variants)
        else:
            self.variants = self._load_variants()
        self._generate_unique_keys()
        self._build_search_index()
        logger.info("Reloaded %d variant definitions", len(self.variants))

    # ── Internal Methods ─────────────────────────────────────────────────

    def _ensure_dimension_fields(
        self, variants: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Ensure every variant dict has a 'dimension' field.

        Legacy data may not include dimensions — assign "Uncategorized".
        """
        for v in variants:
            if "dimension" not in v:
                v["dimension"] = "Uncategorized"
        return variants

    def _generate_unique_keys(self):
        """
        Generate unique display keys for each variant.

        If a variant name appears only once across all dimensions,
        the key is simply the name.  If it appears multiple times
        (e.g., "Immediate Fulfillment" in two different dimensions),
        each instance gets " (DimensionName)" appended.

        This ensures DataFrame columns are always unique.
        """
        # Count how many times each name appears
        name_counts = Counter(v["name"] for v in self.variants)

        self._unique_keys = []
        self._key_to_variant = {}

        for v in self.variants:
            name = v["name"]
            var_id = v.get("variant_id", "")
            dim_id = v.get("dimension_id", "")
            dim = v.get("dimension", "Uncategorized")
            
            if dim_id and dim != "Uncategorized":
                v["dimension_label"] = f"{dim_id} {dim}"
            else:
                v["dimension_label"] = dim

            if var_id:
                key = f"{var_id} {name}"
                v["matrix_label"] = f"{name} ({var_id})"
            else:
                key = name
                v["matrix_label"] = name

            base_key = key
            counter = 2
            while key in self._key_to_variant:
                key = f"{base_key} #{counter}"
                v["matrix_label"] = f'{v["matrix_label"]} #{counter}'
                counter += 1

            self._unique_keys.append(key)
            self._key_to_variant[key] = v

        logger.debug(
            "Generated %d unique keys (%d disambiguated)",
            len(self._unique_keys),
            sum(1 for k, v in self._key_to_variant.items() if k != v["name"]),
        )

    def _build_search_index(self):
        """
        Build the search index: for each variant, compile the list of
        preprocessed search terms (variant name + all alternate_names).

        Both the variant name and each alternate_name are normalized with
        preprocess_variant_term() to match the normalization applied
        to the paper text.

        Keys in the search index are the unique keys (not raw names),
        so they align with what detect_in_text() returns.
        """
        self._search_index.clear()
        for unique_key in self._unique_keys:
            variant = self._key_to_variant[unique_key]
            name = variant["name"]
            alternate_names = variant.get("alternate_names", [])

            # Collect all terms: the name itself + alternate_names
            raw_terms = [name] + alternate_names
            processed_patterns = []
            for term in raw_terms:
                processed = self.preprocessor.preprocess_variant_term(term)
                if processed:
                    # Precompile regex for performance
                    escaped_term = re.escape(processed)
                    # Add word boundaries to avoid partial word matches
                    pattern = re.compile(rf"\b{escaped_term}\b")
                    processed_patterns.append((term, pattern))

            self._search_index[unique_key] = processed_patterns

        logger.debug(
            "Search index built: %d variants, %d total terms",
            len(self._search_index),
            sum(len(patterns) for patterns in self._search_index.values()),
        )

    def _check_terms(self, text: str, compiled_patterns: List[Tuple[str, re.Pattern]]) -> Tuple[bool, Optional[str]]:
        """
        Check if any pattern from the list appears in the text at least
        `threshold` times.

        Uses precompiled regex patterns to improve performance significantly
        over repeated substring or raw regex searching.
        """
        total_count = 0
        detected_term = None
        for term, pattern in compiled_patterns:
            if self.threshold == 1:
                if pattern.search(text):
                    return True, term
            else:
                count = sum(1 for _ in pattern.finditer(text))
                if count > 0 and detected_term is None:
                    detected_term = term  # Store first matching term
                total_count += count
                if total_count >= self.threshold:
                    return True, detected_term
                    
        return (total_count >= self.threshold), detected_term

    def _load_variants(self) -> List[Dict[str, Any]]:
        """
        Load variant definitions from the JSON file.

        Handles both the new dimension-aware format:
            {"dimensions": {"DimName": {"VarName": ["syn1", ...]}}}

        And the legacy flat format:
            {"variants": [{"name": "...", "alternate_names": [...]}]}
        """
        try:
            data = load_json(VARIANTS_FILE)

            # New dimension-aware format
            if isinstance(data, dict) and "dimensions" in data:
                return dimensions_to_flat_list(data["dimensions"])

            # Legacy flat format
            if isinstance(data, dict) and "variants" in data:
                variants = data["variants"]
                return dimensions_to_flat_list(flat_list_to_dimensions(self._ensure_dimension_fields(variants)))

            if isinstance(data, list):
                return dimensions_to_flat_list(flat_list_to_dimensions(self._ensure_dimension_fields(data)))

            return []
        except FileNotFoundError:
            logger.info("No variants file found at %s — starting empty", VARIANTS_FILE)
            return []

    @staticmethod
    def save_variants(
        variants: List[Dict[str, Any]],
        filepath=VARIANTS_FILE,
    ):
        """
        Save variant definitions using the new dimension-aware format.

        Converts the flat list to the canonical dimension dict:
            {"dimensions": {"DimName": {"VarName": ["syn1", ...]}}}
        """
        dimensions = flat_list_to_dimensions(variants)
        save_json({"dimensions": dimensions}, filepath)
        logger.info("Saved %d variant definitions to %s", len(variants), filepath)
