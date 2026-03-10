"""
Shared utility functions used across the system.

Provides:
    • JSON I/O
    • Filename sanitization
    • File hashing for change detection / caching
    • Paper ID generation (P1, P2, P3, ...)
    • Multi-format variant import (CSV, JSON, Excel)

All functions are stateless and side-effect-free (except I/O).
"""

import csv
import io
import json
import re
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config.settings import PAPER_ID_PREFIX, SUPPORTED_PAPER_EXTENSIONS


# ═══════════════════════════════════════════════════════════════════════════
# JSON I/O
# ═══════════════════════════════════════════════════════════════════════════

def load_json(filepath: Path) -> Any:
    """
    Load and parse a JSON file.

    Args:
        filepath: Path to the JSON file.

    Returns:
        Parsed JSON content (dict, list, etc.).

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, filepath: Path, indent: int = 2) -> None:
    """
    Save data as a formatted JSON file.

    Args:
        data: Data to serialize (must be JSON-serializable).
        filepath: Destination path.
        indent: Number of spaces for indentation.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════
# Paper ID Generation
# ═══════════════════════════════════════════════════════════════════════════

def list_paper_files(papers_dir: Path) -> List[Path]:
    """
    Return a sorted list of all supported paper files in the directory.

    Supports: .pdf, .txt (configurable via SUPPORTED_PAPER_EXTENSIONS).

    Files are sorted alphabetically so that P-IDs are deterministic
    and stable across runs (as long as no files are added/removed).

    Args:
        papers_dir: Path to the papers directory.

    Returns:
        Sorted list of Path objects.
    """
    files = []
    for ext in SUPPORTED_PAPER_EXTENSIONS:
        files.extend(papers_dir.glob(f"*{ext}"))
    return sorted(files, key=lambda p: p.name.lower())


def generate_paper_id_map(papers_dir: Path) -> Dict[str, Path]:
    """
    Assign sequential paper IDs (P1, P2, ...) to each file.

    The mapping is deterministic: files are sorted alphabetically,
    then assigned IDs in order.  This means the same set of files
    always produces the same mapping.

    Args:
        papers_dir: Path to the papers directory.

    Returns:
        OrderedDict mapping paper_id (e.g. "P1") → file Path.
    """
    files = list_paper_files(papers_dir)
    return {
        f"{PAPER_ID_PREFIX}{i + 1}": fpath
        for i, fpath in enumerate(files)
    }


def get_paper_id(filename: str) -> str:
    """
    Generate a stable, unique identifier for a paper based on its filename.

    Uses the filename stem (without extension) — simple and deterministic.
    This is a LEGACY helper kept for cache key compatibility.  New code
    should use generate_paper_id_map() for P1/P2/P3 IDs.

    Args:
        filename: Original filename of the paper.

    Returns:
        Sanitized identifier string.
    """
    stem = Path(filename).stem
    return safe_filename(stem)


# ═══════════════════════════════════════════════════════════════════════════
# Filename / String Utilities
# ═══════════════════════════════════════════════════════════════════════════

def safe_filename(name: str) -> str:
    """
    Sanitize a string for use as a filename.

    Replaces all non-alphanumeric characters (except hyphens and underscores)
    with underscores, collapses runs of underscores, and strips leading/trailing
    underscores.

    Args:
        name: Input string.

    Returns:
        Sanitized filename string.
    """
    sanitized = re.sub(r"[^\w\-]", "_", name)
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized.strip("_")


def format_file_size(size_bytes: int) -> str:
    """
    Format a file size in bytes into a human-readable string.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Formatted string (e.g., "1.5 MB").
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.1f} MB"
    else:
        return f"{size_bytes / (1024 ** 3):.2f} GB"


def compute_file_hash(filepath: Path, algorithm: str = "md5") -> str:
    """
    Compute a hash digest of a file for change detection / caching.

    Args:
        filepath: Path to the file.
        algorithm: Hash algorithm name (default 'md5').

    Returns:
        Hex digest string.
    """
    h = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ═══════════════════════════════════════════════════════════════════════════
# Multi-Format Variant Import
# ═══════════════════════════════════════════════════════════════════════════
#
# All loaders convert external formats into the canonical internal structure:
#
#   {
#     "Dimension Name": {
#       "Variant Name": ["alternate_name1", "alternate_name2", ...]
#     }
#   }
#
# This dict-of-dicts is what the engine uses everywhere.
# ═══════════════════════════════════════════════════════════════════════════


def _sanitize_text(text: str) -> str:
    """
    Sanitize a dimension/variant/alternate_name name by removing non-ASCII characters.

    This guards against byte-level corruption (e.g., OneDrive sync issues)
    that can inject CJK or other unexpected Unicode characters into what
    should be plain English text.

    Keeps: ASCII letters, digits, standard punctuation, spaces.
    Removes: CJK characters, control characters, other non-ASCII bytes.

    Args:
        text: Raw string from CSV/Excel/JSON.

    Returns:
        Cleaned ASCII-safe string with extra whitespace collapsed.
    """
    # Keep only printable ASCII (0x20-0x7E) characters
    cleaned = "".join(ch for ch in text if 32 <= ord(ch) <= 126)
    # Collapse multiple spaces into one and strip
    return " ".join(cleaned.split())

def parse_variants_from_csv(file_content: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Parse dimension/variant/alternate_name definitions from CSV text.

    Expected CSV columns: dimension, variant, alternate_name

    Each row adds one alternate_name to a (dimension, variant) pair.
    Multiple rows with the same dimension+variant accumulate alternate_names.

    Example CSV:
        dimension,variant,alternate_name
        Product Energy Consumption,Energy Consuming,energy consuming
        Product Energy Consumption,Energy Consuming,electric powered
        Product Requirement,Temporary,temporary use

    Args:
        file_content: Raw CSV text (decoded string).

    Returns:
        Canonical dimension dict.

    Raises:
        ValueError: If required columns are missing.
    """
    reader = csv.DictReader(io.StringIO(file_content))

    # Normalize column names (strip whitespace, lowercase)
    if reader.fieldnames is None:
        raise ValueError("CSV file appears to be empty.")

    # Create a mapping of lowercase fieldname → original fieldname
    field_map = {f.strip().lower(): f for f in reader.fieldnames}

    required = {"dimension", "variant", "alternate_name"}
    if not required.issubset(field_map.keys()):
        missing = required - set(field_map.keys())
        raise ValueError(
            f"CSV is missing required columns: {missing}. "
            f"Expected: dimension, variant, alternate_name"
        )

    dimensions: Dict[str, Dict[str, List[str]]] = {}
    for row in reader:
        # Access using original field names mapped from lowercase
        dim = _sanitize_text(row[field_map["dimension"]].strip())
        var = _sanitize_text(row[field_map["variant"]].strip())
        syn = _sanitize_text(row[field_map["alternate_name"]].strip())

        if not dim or not var:
            continue  # Skip rows with empty dimension or variant

        if dim not in dimensions:
            dimensions[dim] = {}
        if var not in dimensions[dim]:
            dimensions[dim][var] = []
        if syn and syn not in dimensions[dim][var]:
            dimensions[dim][var].append(syn)

    return dimensions


def parse_variants_from_excel(file_bytes: bytes) -> Dict[str, Dict[str, List[str]]]:
    """
    Parse dimension/variant/alternate_name definitions from an Excel file (.xlsx).

    Same column expectations as CSV: dimension, variant, alternate_name.

    Args:
        file_bytes: Raw bytes of the uploaded Excel file.

    Returns:
        Canonical dimension dict.

    Raises:
        ValueError: If required columns are missing.
        ImportError: If openpyxl is not installed.
    """
    import pandas as pd

    df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"dimension", "variant", "alternate_name"}
    if not required.issubset(set(df.columns)):
        missing = required - set(df.columns)
        raise ValueError(
            f"Excel file is missing required columns: {missing}. "
            f"Expected: dimension, variant, alternate_name"
        )

    dimensions: Dict[str, Dict[str, List[str]]] = {}
    for _, row in df.iterrows():
        dim = _sanitize_text(str(row["dimension"]).strip())
        var = _sanitize_text(str(row["variant"]).strip())
        syn = _sanitize_text(str(row["alternate_name"]).strip())

        if not dim or not var or dim == "nan" or var == "nan":
            continue

        if dim not in dimensions:
            dimensions[dim] = {}
        if var not in dimensions[dim]:
            dimensions[dim][var] = []
        if syn and syn != "nan" and syn not in dimensions[dim][var]:
            dimensions[dim][var].append(syn)

    return dimensions


def parse_variants_from_json(file_content: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Parse dimension/variant/alternate_name definitions from a JSON string.

    Accepts two formats:

    Format A (new, dimension-aware):
        {
          "dimensions": {
            "Dimension Name": {
              "Variant Name": ["alternate_name1", "alternate_name2"]
            }
          }
        }

    Format B (legacy, flat list — auto-assigns "Uncategorized" dimension):
        {
          "variants": [
            {"name": "Variant", "alternate_names": ["syn1", "syn2"]}
          ]
        }

    Args:
        file_content: Raw JSON text (decoded string).

    Returns:
        Canonical dimension dict.

    Raises:
        ValueError: If the JSON structure is unrecognized.
    """
    data = json.loads(file_content)

    # Format A: dimension-aware
    if isinstance(data, dict) and "dimensions" in data:
        raw_dims = data["dimensions"]
        sanitized: Dict[str, Dict[str, List[str]]] = {}
        for dim_name, variants in raw_dims.items():
            clean_dim = _sanitize_text(dim_name)
            if not clean_dim:
                continue
            sanitized[clean_dim] = {}
            for var_name, syns in variants.items():
                clean_var = _sanitize_text(var_name)
                if not clean_var:
                    continue
                sanitized[clean_dim][clean_var] = [
                    _sanitize_text(s) for s in syns if _sanitize_text(s)
                ]
        return sanitized

    # Format B: legacy flat list
    if isinstance(data, dict) and "variants" in data:
        legacy_list = data["variants"]
        dimensions: Dict[str, Dict[str, List[str]]] = {}
        for v in legacy_list:
            dim = _sanitize_text(v.get("dimension", "Uncategorized"))
            name = _sanitize_text(v["name"])
            syns = [_sanitize_text(s) for s in v.get("alternate_names", []) if _sanitize_text(s)]
            if not dim or not name:
                continue
            if dim not in dimensions:
                dimensions[dim] = {}
            dimensions[dim][name] = syns
        return dimensions

    # Format C: bare list
    if isinstance(data, list):
        dimensions = {}
        for v in data:
            dim = _sanitize_text(v.get("dimension", "Uncategorized"))
            name = _sanitize_text(v["name"])
            syns = [_sanitize_text(s) for s in v.get("alternate_names", []) if _sanitize_text(s)]
            if not dim or not name:
                continue
            if dim not in dimensions:
                dimensions[dim] = {}
            dimensions[dim][name] = syns
        return dimensions

    raise ValueError(
        "Unrecognized JSON format. Expected {'dimensions': {...}} or {'variants': [...]}."
    )


def dimensions_to_flat_list(
    dimensions: Dict[str, Dict[str, List[str]]],
) -> List[Dict[str, Any]]:
    """
    Convert the canonical dimension dict into a flat list for UI iteration.

    Each item: {"dimension_id": str, "dimension": str, "variant_id": str, "name": str, "alternate_names": list[str]}

    Useful for the VariantDetector which iterates over individual variants.

    Args:
        dimensions: Canonical dimension dict.

    Returns:
        Flat list of variant definitions, each tagged with its dimension, dimension_id, and variant_id.
    """
    flat = []
    dim_counter = 1
    var_counter = 1
    for dim_name, variants in dimensions.items():
        clean_dim = _sanitize_text(dim_name)
        if not clean_dim:
            continue
            
        dim_id = f"D{dim_counter}"
        dim_counter += 1
        
        for var_name, alternate_names in variants.items():
            clean_var = _sanitize_text(var_name)
            if not clean_var:
                continue
                
            var_id = f"V{var_counter}"
            var_counter += 1
            
            flat.append({
                "dimension_id": dim_id,
                "dimension": clean_dim,
                "variant_id": var_id,
                "name": clean_var,
                "alternate_names": [_sanitize_text(s) for s in alternate_names if _sanitize_text(s)],
            })
    return flat


def flat_list_to_dimensions(
    flat_list: List[Dict[str, Any]],
) -> Dict[str, Dict[str, List[str]]]:
    """
    Convert a flat variant list back into the canonical dimension dict.

    Inverse of dimensions_to_flat_list().

    Args:
        flat_list: List of {"dimension": ..., "name": ..., "alternate_names": [...]}.

    Returns:
        Canonical dimension dict.
    """
    dimensions: Dict[str, Dict[str, List[str]]] = {}
    for v in flat_list:
        dim = v.get("dimension", "Uncategorized")
        name = v["name"]
        syns = v.get("alternate_names", [])
        if dim not in dimensions:
            dimensions[dim] = {}
        dimensions[dim][name] = list(syns)
    return dimensions
