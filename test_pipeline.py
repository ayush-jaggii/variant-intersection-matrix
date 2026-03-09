import sys, os
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from config.settings import PAPERS_DIR
from core.text_extraction import TextExtractor
from interface.components.variant_manager import _load_variants
from interface.components.analysis_runner import _run_pipeline_in_thread, _RESULT
import multiprocessing

if __name__ == "__main__":
    variants = _load_variants()
    print(f"Loaded {len(variants)} variants.")
    _run_pipeline_in_thread(variants)
    if _RESULT.get("status") == "error":
        print("PIPELINE ERROR:", _RESULT.get("error"))
    else:
        print("PIPELINE SUCCESS")
