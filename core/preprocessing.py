"""
Text Preprocessing Module
=========================

Normalizes and cleans extracted text to improve variant matching accuracy.

Normalization Pipeline (applied to both paper text AND alternate_name terms):
    1. Unicode normalization (NFKD — decomposes ligatures and accents)
    2. Lowercase conversion
    3. Replace hyphens / underscores with spaces
    4. Remove all non-alphanumeric characters (except spaces)
    5. Collapse multiple spaces into one and strip edges
    6. (Optional) Stopword removal

Why this matters:
    Variant detection uses substring matching.  If the paper says
    "Temporary-use." and the alternate_name is "temporary use", both must
    be normalized identically ("temporary use") for the match to work.

    The same `preprocess_variant_term()` method is used on alternate_names
    during search-index construction, guaranteeing identical normalization.
"""

import logging
import re
import unicodedata
from typing import Dict, Optional

from config.settings import MIN_WORD_LENGTH, APPLY_STEMMING

logger = logging.getLogger(__name__)

# Common English stopwords (minimal set — we keep domain terms)
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "this", "that",
    "these", "those", "it", "its", "not", "no", "nor", "as", "if", "then",
    "than", "so", "such", "both", "each", "all", "any", "few", "more",
    "most", "other", "some", "into", "through", "during", "before", "after",
    "above", "below", "between", "up", "down", "out", "off", "over", "under",
    "again", "further", "once", "here", "there", "when", "where", "why",
    "how", "very", "just", "about", "also",
})


class TextPreprocessor:
    """
    Cleans and normalizes text for variant matching.

    The same normalization is applied to:
        1. Full paper texts   (via preprocess / preprocess_all)
        2. Individual alternate_name terms (via preprocess_variant_term)

    This ensures that matching is deterministic and consistent.

    Attributes:
        min_word_length:  Minimum word length to retain (stopword mode).
        apply_stemming:   Whether to apply Porter stemming (currently unused).
        remove_stopwords: Whether to remove common stopwords.
    """

    def __init__(
        self,
        min_word_length: int = MIN_WORD_LENGTH,
        apply_stemming: bool = APPLY_STEMMING,
        remove_stopwords: bool = False,
    ):
        self.min_word_length = min_word_length
        self.apply_stemming = apply_stemming
        self.remove_stopwords = remove_stopwords

    def preprocess_all(self, texts: Dict[str, str]) -> Dict[str, str]:
        """
        Preprocess a batch of extracted texts.

        Args:
            texts: Dictionary mapping paper_id → raw text.

        Returns:
            Dictionary mapping paper_id → preprocessed text.
        """
        logger.info("Preprocessing %d documents", len(texts))
        return {
            paper_id: self.preprocess(text)
            for paper_id, text in texts.items()
        }

    def preprocess(self, text: str) -> str:
        """
        Apply the full preprocessing pipeline to a single text.

        We deliberately keep the text as a continuous string (not tokenized)
        because variant names can be multi-word phrases — we need substring
        matching to work.

        Steps:
            1. NFKD unicode normalization
            2. Lowercase
            3. Hyphens/underscores → spaces
            4. Remove non-alphanum (except spaces)
            5. Collapse whitespace
            6. (Optional) stopword removal

        Examples:
            "Temporary-use."       → "temporary use"
            "Energy_Consuming!"    → "energy consuming"
            "COVID-19 is a virus." → "covid 19 is a virus"

        Args:
            text: Raw extracted text.

        Returns:
            Cleaned, normalized text.
        """
        if not text:
            return ""

        # 1. Unicode normalization (NFKD decomposes ligatures)
        text = unicodedata.normalize("NFKD", text)

        # 2. Lowercase
        text = text.lower()

        # 3. Replace hyphens and underscores with spaces
        text = re.sub(r"[-_]", " ", text)

        # 4. Remove non-alphanumeric characters except spaces
        text = re.sub(r"[^a-z0-9\s]", " ", text)

        # 5. Collapse multiple spaces
        text = re.sub(r"\s+", " ", text).strip()

        # 6. Optional stopword removal
        if self.remove_stopwords:
            words = text.split()
            words = [w for w in words if w not in _STOPWORDS and len(w) >= self.min_word_length]
            text = " ".join(words)

        return text

    def preprocess_variant_term(self, term: str) -> str:
        """
        Apply the same normalization to a variant name or alternate_name
        so that matching uses identical representations.

        This is critical for correct matching:
            alternate_name "energy-intensive" → "energy intensive"
            paper text "...energy intensive..." → match!

        Args:
            term: Variant name or alternate_name string.

        Returns:
            Preprocessed term.
        """
        if not term:
            return ""

        term = unicodedata.normalize("NFKD", term)
        term = term.lower()
        term = re.sub(r"[-_]", " ", term)
        term = re.sub(r"[^a-z0-9\s]", " ", term)
        term = re.sub(r"\s+", " ", term).strip()
        return term
