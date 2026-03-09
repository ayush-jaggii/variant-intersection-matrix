import logging
from pathlib import Path

from core.text_extraction import TextExtractor
from core.preprocessing import TextPreprocessor
from core.variant_detection import VariantDetector

logger = logging.getLogger(__name__)

def process_single_paper(args):
    """
    Worker function for ProcessPoolExecutor.
    Processes a single paper: extracts text, normalizes, and detects variants.

    Performance Architecture Notes:
    - ThreadPoolExecutor: Processing 50-200 papers sequentially is slow. We use
      ThreadPoolExecutor instead of multiprocessing because spawning new
      Python interpreters inside a OneDrive directory often causes Errno 60
      (TimeoutError) when macOS attempts to load library files from the cloud.
    - PyMuPDF: Inside TextExtractor, we use fitz (PyMuPDF) instead of pdfplumber
      because it is implemented in C and parses PDFs orders of magnitude faster.
    - Regex Precompilation: Inside VariantDetector, synonym patterns are precompiled
      once into regex objects, removing the overhead of parsing regex strings on every paper.
    - Caching Layer: Text extraction results are written to disk based on the 
      file's modification time (mtime), ensuring we never reprocess untouched PDFs.

    Must be a top-level function to be picklable, and isolated from Streamlit imports
    to avoid spawn issues on macOS.
    """
    paper_id, file_path, variants = args
    
    # 1. Initialize components for this worker
    extractor = TextExtractor()
    preprocessor = TextPreprocessor()
    detector = VariantDetector(variants=variants, preprocessor=preprocessor)
    
    # 2. Extract
    text = extractor._extract_with_cache(file_path, paper_id)
    
    # 3. Normalize
    normalized = preprocessor.preprocess(text)
    
    # 4. Detect
    detection = detector.detect_in_text(normalized)
    
    return paper_id, text, normalized, detection
