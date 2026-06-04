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

import shutil

# ─── Project Root & Persistence Setup ────────────────────────────────────────
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    # We are running in a PyInstaller bundle
    BUNDLE_ROOT = Path(sys._MEIPASS)
    PERSISTENT_ROOT = Path(sys.executable).parent
else:
    # We are running in a normal Python environment
    BUNDLE_ROOT = Path(__file__).resolve().parent.parent
    PERSISTENT_ROOT = BUNDLE_ROOT

# ─── Data Directories ───────────────────────────────────────────────────────
DATA_DIR = PERSISTENT_ROOT / "data"
PAPERS_DIR = DATA_DIR / "papers"
VARIANTS_DIR = DATA_DIR / "variants"
OUTPUT_DIR = DATA_DIR / "output"
CACHE_DIR = DATA_DIR / "cache"

# Helper to copy directory contents or files from bundle to persistent space
def _initialize_persistent_data():
    # Only copy if we are frozen and persistent space is different from bundle
    if BUNDLE_ROOT != PERSISTENT_ROOT:
        bundle_data = BUNDLE_ROOT / "data"
        if bundle_data.exists():
            # 1. Copy default papers if papers folder is missing or empty
            bundle_papers = bundle_data / "papers"
            if not PAPERS_DIR.exists() or not any(PAPERS_DIR.iterdir()):
                if bundle_papers.exists():
                    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
                    for f in bundle_papers.iterdir():
                        if f.is_file():
                            shutil.copy2(f, PAPERS_DIR / f.name)
            
            # Also handle if bundle had "Papers" with uppercase P
            bundle_papers_caps = bundle_data / "Papers"
            if bundle_papers_caps.exists() and (not PAPERS_DIR.exists() or not any(PAPERS_DIR.iterdir())):
                PAPERS_DIR.mkdir(parents=True, exist_ok=True)
                for f in bundle_papers_caps.iterdir():
                    if f.is_file():
                        shutil.copy2(f, PAPERS_DIR / f.name)

            # 2. Copy variants.json if missing
            bundle_variants = bundle_data / "variants" / "variants.json"
            dest_variants = VARIANTS_DIR / "variants.json"
            if bundle_variants.exists() and not dest_variants.exists():
                shutil.copy2(bundle_variants, dest_variants)

            # 3. Copy other default files in data/ if missing
            for item in bundle_data.iterdir():
                if item.is_file():
                    dest_file = DATA_DIR / item.name
                    if not dest_file.exists():
                        shutil.copy2(item, dest_file)

# Ensure directories exist at import time
for _dir in (PAPERS_DIR, VARIANTS_DIR, OUTPUT_DIR, CACHE_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# Copy default template data if running from bundle and files don't exist
_initialize_persistent_data()

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

# Conceptual validation (whether a combination of two variants is meaningful)
# Format: {"variant_a|variant_b": "R" | "N" | "?"}
CONCEPTUAL_VALIDATION_FILE = CACHE_DIR / "conceptual_validation.json"

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
HEATMAP_COLORSCALE = [[0, "#FFFFFF"], [1, "#1B2A4A"]]
HEATMAP_ZERO_COLOR = "#FFFFFF"

# Maximum file upload size in MB
MAX_UPLOAD_SIZE_MB = 200
