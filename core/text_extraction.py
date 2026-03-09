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
