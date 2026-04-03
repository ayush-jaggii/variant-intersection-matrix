"""
Text Extraction Module
======================

Extracts raw text content from research paper files.

Supported formats:
    • PDF  — extracted page-by-page using pdfplumber
    • TXT  — read directly via Python open()

Key design decisions:
    • Paper IDs are sequential (P1, P2, P3, ...) based on alphabetical
      file ordering.  This makes references short, stable, and human-readable.
    • Batch processing to limit memory usage with large paper sets.
    • Per-page extraction with optional page limit (PDFs only).
    • Caching of extracted text, keyed by file content hash, so re-runs
      skip unchanged papers.
    • Structured return: {paper_id: full_text}.
"""

import logging
import os
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import fitz  # PyMuPDF

from config.settings import (
    PAPERS_DIR,
    CACHE_DIR,
    PDF_BATCH_SIZE,
    MAX_PAGES_PER_PAPER,
)
from utils.helpers import (
    generate_paper_id_map,
    save_json,
    load_json,
    get_paper_id,
)

logger = logging.getLogger(__name__)

_TITLE_CACHE_FILE = CACHE_DIR / "paper_title_cache.json"
_TITLE_HEURISTIC_VERSION = "v4-metadata-plus-multiformat"
_TITLE_HEADER_BLACKLIST = {
    "abstract",
    "keywords",
    "keyword",
    "introduction",
    "references",
    "acknowledgements",
    "acknowledgments",
    "conclusion",
}
_JUNK_PATTERNS = [
    "journal",
    "elsevier",
    "science direct",
    "contents lists available",
    "homepage",
    "issn",
    "doi",
    "volume",
    "issue",
    "www.",
    "available online",
    "article info",
    "abstract",
]
_KNOWN_JOURNALS = [
    "industrial marketing management",
    "journal of",
    "elsevier",
    "springer",
    "ieee",
]


def _load_title_cache() -> Dict[str, Dict[str, str]]:
    """Load on-disk title cache safely."""
    try:
        data = load_json(_TITLE_CACHE_FILE)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _save_title_cache(cache: Dict[str, Dict[str, str]]) -> None:
    """Persist title cache safely without breaking execution."""
    try:
        save_json(cache, _TITLE_CACHE_FILE)
    except Exception:
        pass


def _is_obvious_header(text: str) -> bool:
    """Return True for obvious section headings that are not paper titles."""
    normalized = re.sub(r"[^a-z]+", " ", text.lower()).strip()
    if not normalized:
        return True
    words = normalized.split()
    if not words:
        return True
    if normalized in _TITLE_HEADER_BLACKLIST:
        return True
    if len(words) <= 2 and normalized in _TITLE_HEADER_BLACKLIST:
        return True
    return False


def _is_valid_title_candidate(text: str) -> bool:
    """Check whether a raw extracted text segment is title-like."""
    candidate = (text or "").strip()
    if len(candidate) < 10:
        return False
    if _is_obvious_header(candidate):
        return False

    alpha_chars = sum(1 for c in candidate if c.isalpha())
    if alpha_chars < 6:
        return False

    if candidate.isupper() and len(candidate.split()) <= 3:
        return False

    # Reject numeric/garbage-like strings.
    if re.fullmatch(r"[\W\d_\-\.:;,/\\]+", candidate):
        return False

    return True


def _is_junk_title_text(text: str) -> bool:
    """Reject journal headers and metadata-like fragments."""
    normalized = re.sub(r"\s+", " ", (text or "").strip())
    if not normalized:
        return True

    lower = normalized.lower()
    alpha_only = re.sub(r"[^a-z]", "", lower)
    if any(pattern in lower for pattern in _JUNK_PATTERNS):
        return True
    if "abstract" in alpha_only or "articleinfo" in alpha_only:
        return True

    words = normalized.split()
    if len(words) < 5:
        return True

    if normalized.isupper() and len(words) <= 6:
        return True

    return False


def _score_title_candidate(candidate: Dict[str, float]) -> float:
    """Score title candidates using size, length, and linguistic cues."""
    text = str(candidate.get("text", ""))
    size = float(candidate.get("size", 0.0))

    score = 0.0
    score += size * 2.0
    score += len(text) * 0.4
    score += len(text.split()) * 2.0

    if text.isupper():
        score -= 30.0
    if len(text.split()) < 5:
        score -= 20.0

    return score


def _is_body_like_sentence(text: str) -> bool:
    """Reject long sentence-like fragments commonly pulled from abstract/body."""
    t = re.sub(r"\s+", " ", (text or "").strip())
    if not t:
        return True

    words = t.split()
    if len(words) > 22:
        return True

    # Body sentences often end with period/semicolon and have dense punctuation.
    punctuation_count = len(re.findall(r"[,:;.]", t))
    if t.endswith((".", ";", ":")) and len(words) > 10:
        return True
    if punctuation_count >= 4 and len(words) > 12:
        return True
    if re.search(r"\b(article\s*info|abstract|keywords|introduction)\b", t.lower()):
        return True

    return False


def _looks_like_title_line(text: str) -> bool:
    """Shape filter for title lines to avoid selecting paragraph text."""
    t = re.sub(r"\s+", " ", (text or "").strip())
    if not t:
        return False

    words = t.split()
    wc = len(words)
    if wc < 4 or wc > 26:
        return False
    if t.endswith((".", ";")):
        return False
    if _is_body_like_sentence(t):
        return False

    alpha = [c for c in t if c.isalpha()]
    if not alpha:
        return False
    upper_ratio = sum(1 for c in alpha if c.isupper()) / len(alpha)
    if upper_ratio > 0.75:
        return False

    return True


def _are_title_neighbors(a: Dict[str, float], b: Dict[str, float]) -> bool:
    """Check if two lines are likely consecutive parts of the same title."""
    ay = float(a.get("y", 0.0))
    by = float(b.get("y", 0.0))
    a_size = float(a.get("size", 0.0))
    b_size = float(b.get("size", 0.0))

    if abs(by - ay) > 28.0:
        return False
    if abs(a_size - b_size) > 2.0:
        return False

    a_text = str(a.get("text", "")).strip()
    b_text = str(b.get("text", "")).strip()
    if not (_looks_like_title_line(a_text) and _looks_like_title_line(b_text)):
        return False
    if _is_junk_title_text(a_text) or _is_junk_title_text(b_text):
        return False

    return True


def _score_title_candidate_v2(candidate: Dict[str, float], page_height: float) -> float:
    """Title-specific scoring with stronger position and readability priors."""
    text = str(candidate.get("text", ""))
    size = float(candidate.get("size", 0.0))
    y = float(candidate.get("y", 0.0))

    words = text.split()
    word_count = len(words)
    upper_ratio = (
        sum(1 for ch in text if ch.isupper()) / max(1, sum(1 for ch in text if ch.isalpha()))
    )

    score = 0.0
    score += size * 3.2
    score += min(len(text), 180) * 0.25
    score += min(word_count, 24) * 2.4

    # Prefer upper-middle of page (typical title band) while avoiding header strip.
    if page_height > 0:
        normalized_y = y / page_height
        score -= abs(normalized_y - 0.30) * 45.0

    if text.isupper():
        score -= 28.0
    if word_count < 5:
        score -= 30.0
    if upper_ratio > 0.55:
        score -= 14.0
    if _is_body_like_sentence(text):
        score -= 50.0

    stripped = text.strip()
    if stripped and (stripped[0].islower() or stripped[0] in ":,;.)]"):
        score -= 16.0

    return score


def clean_title(title: str) -> str:
    """
    Normalize and validate an extracted paper title.

    Returns an empty string when the input does not look like a real title.
    """
    if not title:
        return ""

    cleaned = title.replace("\n", " ").replace("\r", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"[\s\.,:;\-_|]{2,}$", "", cleaned).strip()

    if len(cleaned) < 10:
        return ""
    if len(cleaned.split()) < 2:
        return ""
    if len(cleaned.split()) > 30:
        return ""
    if _is_obvious_header(cleaned):
        return ""
    if not _is_valid_title_candidate(cleaned):
        return ""

    return cleaned


def _extract_title_from_pdf_metadata(pdf_path: Path) -> str:
    """Use embedded PDF metadata title when it looks trustworthy."""
    try:
        with fitz.open(str(pdf_path)) as doc:
            metadata = doc.metadata or {}
            raw = str(metadata.get("title") or "").strip()
            if not raw:
                return ""
            cleaned = clean_title(raw)
            if not cleaned:
                return ""
            if _is_junk_title_text(cleaned):
                return ""
            return cleaned
    except Exception:
        return ""


def _extract_title_from_first_page_text(pdf_path: Path) -> str:
    """
    Text-based fallback for papers where span geometry is noisy.

    Scans the first page lines and picks the first title-like candidate near the top.
    """
    try:
        with fitz.open(str(pdf_path)) as doc:
            if len(doc) == 0:
                return ""

            page = doc[0]

            lines = page.get_text("text").splitlines()
            candidates: List[str] = []
            for line in lines:
                t = re.sub(r"\s+", " ", (line or "").strip())
                if not t:
                    continue
                if _is_junk_title_text(t):
                    continue
                if not _looks_like_title_line(t):
                    continue
                cleaned = clean_title(t)
                if cleaned:
                    candidates.append(cleaned)

            if not candidates:
                return ""

            # Prefer medium-length candidates; very short/long lines are usually noise.
            return sorted(candidates, key=lambda c: abs(len(c) - 95))[0]
    except Exception:
        return ""


def extract_title_from_txt(txt_path: Path) -> str:
    """Extract a title-like name from TXT papers using first non-junk lines."""
    try:
        try:
            content = txt_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = txt_path.read_text(encoding="latin-1")

        lines = [re.sub(r"\s+", " ", ln.strip()) for ln in content.splitlines()]
        lines = [ln for ln in lines if ln]
        if not lines:
            return ""

        for line in lines[:25]:
            if _is_junk_title_text(line):
                continue
            if not _looks_like_title_line(line):
                continue
            cleaned = clean_title(line)
            if cleaned:
                return cleaned

        # Last fallback from content: first line if reasonably clean.
        fallback_line = clean_title(lines[0])
        return fallback_line
    except Exception:
        return ""


def get_document_name(file_path) -> str:
    """
    Extract a robust user-facing paper name from PDF or TXT.

    Falls back to filename stem and never raises.
    """
    path = Path(file_path)
    fallback = path.stem

    if path.suffix.lower() == ".pdf":
        return get_paper_title(path)
    if path.suffix.lower() == ".txt":
        txt_title = extract_title_from_txt(path)
        return txt_title or clean_title(fallback) or fallback
    return clean_title(fallback) or fallback


def extract_title_smart(pdf_path) -> str:
    """
    Primary title extraction using first-page structured spans in PyMuPDF.

    Strategy:
        1) Read first page only.
        2) Find the span/line text with the largest font size.
        3) Return the cleaned candidate.
    """
    path = Path(pdf_path)
    fallback = path.stem

    try:
        with fitz.open(str(path)) as doc:
            if len(doc) == 0:
                return fallback

            page = doc[0]
            page_height = float(page.rect.height or 0.0)
            text_dict = page.get_text("dict")

            top_ignore_threshold = 0.12 * page_height
            lower_window_default = 0.62 * page_height
            abstract_y = None

            raw_lines: List[Dict[str, float]] = []

            # Build line-level candidates (more stable than per-span selection).
            for block in text_dict.get("blocks", []):
                if block.get("type", 0) != 0:
                    continue

                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue

                    parts: List[str] = []
                    max_size = 0.0
                    bbox = None
                    for span in spans:
                        t = (span.get("text") or "").strip()
                        if not t:
                            continue
                        parts.append(t)
                        max_size = max(max_size, float(span.get("size", 0.0)))
                        sb = span.get("bbox") or [0.0, 0.0, 0.0, 0.0]
                        if bbox is None:
                            bbox = [float(v) for v in sb]
                        else:
                            bbox[0] = min(bbox[0], float(sb[0]))
                            bbox[1] = min(bbox[1], float(sb[1]))
                            bbox[2] = max(bbox[2], float(sb[2]))
                            bbox[3] = max(bbox[3], float(sb[3]))

                    if not parts or bbox is None:
                        continue

                    text = re.sub(r"\s+", " ", " ".join(parts)).strip()
                    y = float(bbox[1])

                    normalized_alpha = re.sub(r"[^a-z]", "", text.lower())
                    if abstract_y is None and (
                        re.search(r"\babstract\b", text.lower()) or "abstract" in normalized_alpha
                    ):
                        abstract_y = y

                    raw_lines.append({"text": text, "size": max_size, "y": y})

            if not raw_lines:
                return fallback

            lower_window = lower_window_default
            if abstract_y is not None:
                lower_window = min(lower_window_default, max(top_ignore_threshold + 20.0, abstract_y - 8.0))

            candidates: List[Dict[str, float]] = []
            for c in raw_lines:
                text = str(c["text"])
                y = float(c["y"])

                if len(text) <= 10:
                    continue
                if y <= top_ignore_threshold or y >= lower_window:
                    continue
                if re.fullmatch(r"\d+(?:[\d\s\-\.,:]*)", text):
                    continue
                if re.fullmatch(r"[\W_]+", text):
                    continue
                if _is_junk_title_text(text):
                    continue
                if _is_body_like_sentence(text):
                    continue
                if not _looks_like_title_line(text):
                    continue

                lower = text.lower()
                if any(journal in lower for journal in _KNOWN_JOURNALS):
                    continue

                candidates.append(c)

            if not candidates:
                return fallback

            ranked = sorted(
                candidates,
                key=lambda c: _score_title_candidate_v2(c, page_height),
                reverse=True,
            )

            best = ranked[0]

            # Merge lines around the anchor in BOTH directions so wrapped
            # titles are reconstructed even when the anchor is the second line.
            candidates_by_y = sorted(candidates, key=lambda c: float(c["y"]))
            best_idx = 0
            for i, c in enumerate(candidates_by_y):
                if (
                    str(c.get("text", "")) == str(best.get("text", ""))
                    and abs(float(c.get("y", 0.0)) - float(best.get("y", 0.0))) < 0.1
                ):
                    best_idx = i
                    break

            selected = {best_idx}

            cur = best_idx
            for _ in range(2):
                prev_idx = cur - 1
                if prev_idx < 0:
                    break
                if not _are_title_neighbors(candidates_by_y[prev_idx], candidates_by_y[cur]):
                    break
                selected.add(prev_idx)
                cur = prev_idx

            cur = best_idx
            for _ in range(2):
                next_idx = cur + 1
                if next_idx >= len(candidates_by_y):
                    break
                if not _are_title_neighbors(candidates_by_y[cur], candidates_by_y[next_idx]):
                    break
                selected.add(next_idx)
                cur = next_idx

            title_lines = [candidates_by_y[i] for i in sorted(selected)]
            merged_title = " ".join(str(c["text"]) for c in title_lines)
            merged_title = re.sub(r"\s+", " ", merged_title).strip()
            if len(merged_title.split()) > 28:
                merged_title = str(best["text"])

            cleaned = clean_title(merged_title)
            if cleaned:
                return cleaned

            for c in ranked:
                cleaned = clean_title(str(c["text"]))
                if cleaned:
                    return cleaned

            return fallback
    except Exception as e:
        logger.debug("extract_title_smart failed for %s: %s", pdf_path, e)
        return fallback


def extract_title_from_blocks(pdf_path) -> str:
    """
    Positional fallback using top text blocks on the first page.

    Strategy:
        1) Read first page blocks.
        2) Sort blocks by vertical position.
        3) Inspect top 5 meaningful blocks.
        4) Pick the longest valid candidate.
    """
    try:
        with fitz.open(str(pdf_path)) as doc:
            if len(doc) == 0:
                return ""

            page = doc[0]
            blocks = page.get_text("blocks")
            if not blocks:
                return ""

            sorted_blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
            candidates: List[str] = []

            for block in sorted_blocks:
                text = (block[4] or "").strip()
                if not text:
                    continue
                text = re.sub(r"\s+", " ", text).strip()
                lower = text.lower()

                if len(text) <= 10:
                    continue
                if any(term in lower for term in ("abstract", "keywords", "introduction")):
                    continue
                if not _is_valid_title_candidate(text):
                    continue

                candidates.append(text)
                if len(candidates) >= 5:
                    break

            if not candidates:
                return ""

            best = max(candidates, key=len)
            return clean_title(best)
    except Exception as e:
        logger.debug("extract_title_from_blocks failed for %s: %s", pdf_path, e)
        return ""


def get_paper_title(pdf_path) -> str:
    """
    Extract a robust paper title from a PDF via deterministic fallbacks.

    Fallback pipeline:
        1) embedded PDF metadata title
        2) extract_title_smart
        3) extract_title_from_blocks
        4) first-page plain-text scan
        5) filename stem

    Never raises; always returns a string.
    """
    path = Path(pdf_path)
    fallback = path.stem

    try:
        if path.suffix.lower() != ".pdf":
            return clean_title(fallback) or fallback

        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0

        cache = _load_title_cache()
        cache_key = str(path.resolve())
        cached = cache.get(cache_key, {})

        if cached.get("mtime") == mtime and cached.get("title"):
            if cached.get("heuristic_version") == _TITLE_HEURISTIC_VERSION:
                return cached["title"]

        title = _extract_title_from_pdf_metadata(path)
        if not title:
            title = extract_title_smart(path)
        if not title:
            title = extract_title_from_blocks(path)
        if not title:
            title = _extract_title_from_first_page_text(path)

        title = clean_title(title)
        final_title = title or fallback

        cache[cache_key] = {
            "mtime": mtime,
            "title": final_title,
            "heuristic_version": _TITLE_HEURISTIC_VERSION,
        }
        _save_title_cache(cache)
        return final_title
    except Exception as e:
        logger.debug("get_paper_title failed for %s: %s", path, e)
        return fallback


class TextExtractor:
    """
    Handles text extraction from PDFs and TXT files with caching.

    Attributes:
        papers_dir: Directory containing paper files (PDF, TXT).
        cache_dir:  Directory for caching extracted text.
        batch_size: Number of papers to process in each batch.
        max_pages:  Maximum pages per PDF (None = all).
        paper_id_map: Mapping of P-IDs to file paths, computed on init.
    """

    def __init__(
        self,
        papers_dir: Path = PAPERS_DIR,
        cache_dir: Path = CACHE_DIR,
        batch_size: int = PDF_BATCH_SIZE,
        max_pages: Optional[int] = MAX_PAGES_PER_PAPER,
    ):
        self.papers_dir = Path(papers_dir)
        self.cache_dir = Path(cache_dir)
        self.batch_size = batch_size
        self.max_pages = max_pages
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Build the P1/P2/P3 mapping from sorted files
        self.paper_id_map: Dict[str, Path] = generate_paper_id_map(self.papers_dir)

    # ── Public API ───────────────────────────────────────────────────────

    def extract_all(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, str]:
        """
        Extract text from every paper file in papers_dir.

        Uses cached results when the file hash has not changed.

        Args:
            progress_callback: Optional callable(current, total) for UI progress.

        Returns:
            Ordered dictionary mapping paper_id (P1, P2, ...) → full text.
        """
        total = len(self.paper_id_map)
        if total == 0:
            logger.warning("No paper files found in %s", self.papers_dir)
            return {}

        logger.info(
            "Extracting text from %d papers (batch_size=%d)", total, self.batch_size
        )

        results: Dict[str, str] = {}
        items = list(self.paper_id_map.items())

        for batch_start in range(0, total, self.batch_size):
            batch = items[batch_start : batch_start + self.batch_size]
            for i, (paper_id, file_path) in enumerate(batch):
                global_idx = batch_start + i
                text = self._extract_with_cache(file_path, paper_id)
                results[paper_id] = text

                if progress_callback:
                    progress_callback(global_idx + 1, total)

        logger.info("Extraction complete: %d papers processed", len(results))
        return results

    def get_id_to_filename_map(self) -> Dict[str, str]:
        """
        Return a mapping of paper IDs to original filenames.

        Useful for the UI to show "P1 → paper_name.pdf" lookups.

        Returns:
            Dict mapping P-ID → filename string.
        """
        return {pid: fpath.name for pid, fpath in self.paper_id_map.items()}

    def get_id_to_paper_meta_map(
        self,
        title_overrides: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Dict[str, str]]:
        """
        Return paper metadata map with filename and extracted title for each ID.

        Args:
            title_overrides: Optional filename -> custom title overrides from UI.

        Returns:
            Dict mapping paper_id -> {"filename": ..., "title": ...}.
        """
        overrides = title_overrides or {}
        meta: Dict[str, Dict[str, str]] = {}

        for pid, fpath in self.paper_id_map.items():
            filename = fpath.name
            if filename in overrides:
                title = clean_title(overrides[filename]) or overrides[filename].strip()
            else:
                title = get_document_name(fpath)

            meta[pid] = {
                "filename": filename,
                "title": title,
            }

        return meta

    # ── Internal Methods ─────────────────────────────────────────────────

    def _extract_with_cache(self, file_path: Path, paper_id: str) -> str:
        """
        Return extracted text, using cached version if the file hasn't changed.

        Cache key is the paper_id. Cache is invalidated based on file modification time.
        """
        cache_key = get_paper_id(file_path.name)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        # Use modification time instead of hashing for performance
        try:
            mtime = os.path.getmtime(file_path)
        except OSError:
            mtime = 0

        # Check cache validity
        if cache_file.exists():
            try:
                cached = load_json(cache_file)
                if cached.get("mtime") == mtime:
                    logger.debug("Cache hit for %s (%s)", paper_id, file_path.name)
                    return cached["text"]
            except Exception:
                pass  # Cache corrupt — re-extract

        # Extract fresh based on file extension
        text = self._extract_file(file_path)

        # Save to cache (may fail on OneDrive due to sync timeouts)
        try:
            save_json({"mtime": mtime, "text": text}, cache_file)
            logger.debug("Cached extraction for %s (%s)", paper_id, file_path.name)
        except (TimeoutError, OSError) as e:
            logger.warning("Cache write failed for %s: %s", paper_id, e)

        return text

    def _extract_file(self, file_path: Path) -> str:
        """
        Extract text from a file, dispatching to the correct method
        based on file extension.

        Args:
            file_path: Path to the paper file.

        Returns:
            Full extracted text.
        """
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            return self._extract_pdf(file_path)
        elif ext == ".txt":
            return self._extract_txt(file_path)
        else:
            logger.warning("Unsupported file type: %s", ext)
            return ""

    def _extract_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from a PDF file using PyMuPDF (fitz).
        Significantly faster than pdfplumber.

        Concatenates text from each page.
        Respects the max_pages limit if configured.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Concatenated text from all (or limited) pages.
        """
        pages_text: List[str] = []
        try:
            with fitz.open(pdf_path) as pdf:
                page_limit = self.max_pages or len(pdf)
                for page_num in range(min(page_limit, len(pdf))):
                    page = pdf[page_num]
                    text = page.get_text()
                    if text:
                        pages_text.append(text)
        except Exception as e:
            logger.error("Failed to extract PDF %s: %s", pdf_path.name, e)
            return ""

        return "\n".join(pages_text)

    def _extract_txt(self, txt_path: Path) -> str:
        """
        Read text from a plain TXT file.

        Attempts UTF-8 encoding first, falls back to latin-1 if that fails.

        Args:
            txt_path: Path to the TXT file.

        Returns:
            Full file content as a string.
        """
        try:
            return txt_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                return txt_path.read_text(encoding="latin-1")
            except Exception as e:
                logger.error("Failed to read TXT %s: %s", txt_path.name, e)
                return ""
