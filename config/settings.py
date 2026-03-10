"""
Centralized configuration for the Variant Intersection Matrix system.

All paths, thresholds, and tunables are defined here so that no module
contains hard-coded magic values.

Architecture note:
    Every module imports from this file rather than defining its own paths
    or constants.  This makes it trivial to change directory layouts,
    tweak detection thresholds, or switch color schemes — all in one place.
"""

import os
import sys
from pathlib import Path

# ─── Project Root ────────────────────────────────────────────────────────────
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    # We are running in a PyInstaller bundle
    PROJECT_ROOT = Path(sys._MEIPASS)
else:
    # We are running in a normal Python environment
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ─── Data Directories ───────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
PAPERS_DIR = DATA_DIR / "papers"
VARIANTS_DIR = DATA_DIR / "variants"
OUTPUT_DIR = DATA_DIR / "output"
CACHE_DIR = DATA_DIR / "cache"

# Ensure directories exist at import time
for _dir in (PAPERS_DIR, VARIANTS_DIR, OUTPUT_DIR, CACHE_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ─── Variant Definitions File ───────────────────────────────────────────────
# Now uses dimension-aware format:
#   {
#     "dimensions": {
#       "Dimension Name": {
#         "Variant Name": ["alternate_name1", "alternate_name2", ...]
#       }
#     }
#   }
VARIANTS_FILE = VARIANTS_DIR / "variants.json"

# ─── Output Filenames ───────────────────────────────────────────────────────
PAPER_VARIANT_MATRIX_CSV = "paper_variant_matrix.csv"
VARIANT_INTERSECTION_MATRIX_CSV = "variant_intersection_matrix.csv"
PAIR_DETAILS_CSV = "pair_details.csv"
MANUAL_OVERRIDES_FILE = CACHE_DIR / "manual_overrides.json"

# Manual pair validation overrides (variant combos confirmed by researcher)
# Format: {"paper_id": [["variant_a", "variant_b"], ...]}
PAIR_OVERRIDES_FILE = CACHE_DIR / "pair_overrides.json"

# ─── Paper Configuration ────────────────────────────────────────────────────
# Supported file types for research papers
SUPPORTED_PAPER_EXTENSIONS = [".pdf", ".txt"]

# Prefix for auto-generated paper IDs (P1, P2, P3, ...)
PAPER_ID_PREFIX = "P"

# ─── Processing Configuration ───────────────────────────────────────────────
# Batch size for paper processing (number of papers per batch)
PDF_BATCH_SIZE = 25

# Maximum number of pages to extract per PDF paper (None = all pages)
MAX_PAGES_PER_PAPER = None

# ─── Text Preprocessing ─────────────────────────────────────────────────────
# Minimum word length to keep during preprocessing
MIN_WORD_LENGTH = 2

# Whether to apply stemming (can reduce recall for multi-word variants)
APPLY_STEMMING = False

# ─── Variant Detection ──────────────────────────────────────────────────────
# Minimum number of occurrences to consider a variant "present" in a paper
PRESENCE_THRESHOLD = 1

# Whether to use case-insensitive matching (recommended)
CASE_INSENSITIVE = True

# ─── Streamlit Interface ────────────────────────────────────────────────────
# Page configuration
PAGE_TITLE = "Variant Intersection Matrix Analyzer"
PAGE_ICON = "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 36 36'><rect width='36' height='36' rx='8' fill='%231B2A4A'/><text x='18' y='26' text-anchor='middle' font-size='22' fill='white'>V</text></svg>"
PAGE_LAYOUT = "wide"

# Matrix heatmap color scale
HEATMAP_COLORSCALE = [[0, "#EEF1F6"], [0.25, "#93C5E8"], [0.5, "#3B82B0"], [0.75, "#2C4470"], [1, "#1B2A4A"]]
HEATMAP_ZERO_COLOR = "#F7F8FA"

# Maximum file upload size in MB
MAX_UPLOAD_SIZE_MB = 200
