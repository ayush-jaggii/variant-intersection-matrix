"""
Analysis Runner Component
=========================

Orchestrates the full analysis pipeline from the Streamlit UI:
    1. Text extraction from PDFs and TXT files
    2. Text preprocessing / normalization
    3. Dimension-aware variant detection
    4. Matrix computation (with same-dimension pair exclusion)
    5. CSV export

Architecture Note — Hybrid Approach:
    The pipeline itself runs SYNCHRONOUSLY using st.status() for real-time
    progress feedback.  This works well because the pipeline completes in
    ~3-5 seconds.

    To survive tab switches, the RESULTS are persisted in a module-level
    dict (not just session_state).  If a user starts analysis, switches
    tabs (interrupting the synchronous run), and comes back, they can
    simply re-run — or if a thread is used, the results will be waiting.

    We use a threading approach:
    - Thread runs the pipeline and stores results in module-level dict
    - A sync polling loop in the UI shows progress via st.status()
    - The loop checks the module-level dict every 0.5 seconds
    - This gives smooth progress updates AND survives tab switches

Paper IDs:
    Papers are assigned sequential IDs (P1, P2, P3, ...) based on
    alphabetical ordering.

CSV Exports & Manual Validation (Req #3):
    CSV downloads include all manual validations because overrides
    are baked into the matrices before export.
"""

import logging
import threading
import time
import streamlit as st
from pathlib import Path

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from config.settings import PAPERS_DIR, CACHE_DIR
from core.pipeline import process_single_paper
from core.text_extraction import TextExtractor
from core.preprocessing import TextPreprocessor
from core.variant_detection import VariantDetector
from core.matrix_computation import MatrixComputer
from utils.helpers import list_paper_files, load_json
from interface.design import section_header, sub_header, icon

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Module-level results store
# ═══════════════════════════════════════════════════════════════════════════
#
# This dict persists across Streamlit reruns because the Python process
# stays alive.  The daemon thread writes results here; on command,
# the main thread reads them.
#
# Using a single key because this is a single-user research tool.

_RESULT = {
    "status": None,       # None | "running" | "complete" | "error"
    "data": None,         # dict of results when complete
    "error": None,        # error message if failed
    "progress_step": 0,   # current step (1-4)
    "progress_msg": "",   # human-readable progress message
    "sub_current": 0,     # sub-step progress: current item (e.g., paper 15)
    "sub_total": 0,       # sub-step progress: total items (e.g., 56 papers)
}

TITLE_OVERRIDES_FILE = CACHE_DIR / "paper_title_overrides.json"


def _get_title_overrides_for_analysis() -> dict:
    """Get title overrides from session state, with disk fallback."""
    overrides = st.session_state.get("paper_title_overrides")
    if isinstance(overrides, dict):
        return overrides

    try:
        data = load_json(TITLE_OVERRIDES_FILE)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    return {}


def _is_analysis_running() -> bool:
    """Check if a background analysis is currently running."""
    return _RESULT["status"] == "running"


def _check_and_load_results():
    """
    Check if background analysis completed and load results into session state.

    Called on EVERY Streamlit rerun (from app.py) so results are picked up
    regardless of which page the user is on.
    """
    if _RESULT["status"] == "complete" and _RESULT["data"] is not None:
        data = _RESULT["data"]
        st.session_state.matrix_computer = data["computer"]
        st.session_state.paper_variant_df = data["paper_variant_df"]
        st.session_state.intersection_df = data["intersection_df"]
        st.session_state.detection_results = data["detection_results"]
        st.session_state.detection_details = data["detection_details"]
        st.session_state.preprocessed_texts = data["preprocessed"]
        st.session_state.variant_detector = data["detector"]
        st.session_state.dimension_map = data["dimension_map"]
        st.session_state.paper_id_map = data["id_to_filename"]
        st.session_state.paper_meta_map = data.get("id_to_paper_meta", {})
        st.session_state.analysis_complete = True
        st.session_state.analysis_running = False

        # Clear so we don't re-load on next rerun
        _RESULT["status"] = None
        _RESULT["data"] = None
        logger.info("Background results loaded into session state")

    elif _RESULT["status"] == "error":
        st.session_state.analysis_running = False

    elif _RESULT["status"] == "running":
        st.session_state.analysis_running = True


def render_analysis_runner():
    """Render the Analysis Runner section of the UI."""
    st.markdown(section_header("settings", "Run Analysis"), unsafe_allow_html=True)
    st.info("Click the button below to extract text, process papers, and build the Variant Intersection Matrix.")

    # Pre-check: are papers and variants ready?
    papers = list_paper_files(PAPERS_DIR)
    variants = st.session_state.get("variants", [])

    col1, col2 = st.columns(2)
    with col1:
        papers_ready = len(papers) > 0
        st.metric("Papers Loaded", len(papers))
        if not papers_ready:
            st.warning("Upload at least one paper before running analysis.")
    with col2:
        variants_ready = len(variants) > 0
        st.metric("Variants Defined", len(variants))
        if not variants_ready:
            st.warning("Define at least one variant before running analysis.")

    can_run = papers_ready and variants_ready
    is_running = _is_analysis_running()

    st.divider()

    # ── Show error if previous run failed ────────────────────────────
    if _RESULT["status"] == "error" and _RESULT["error"]:
        st.error(f"Previous analysis failed: {_RESULT['error']}")
        _RESULT["status"] = None
        _RESULT["error"] = None

    # ── Run Button ───────────────────────────────────────────────────
    run_clicked = st.button(
        "Analysis Running..." if is_running else "Run Full Analysis",
        type="primary",
        disabled=not can_run or is_running,
        use_container_width=True,
    )

    if run_clicked:
        _run_analysis_with_live_progress(
            list(variants),
            _get_title_overrides_for_analysis(),
        )

    # ── In-progress indicator (when user returns to this page mid-run) ──
    if is_running and not run_clicked:
        _show_polling_progress()

    # ── Show cached results summary ─────────────────────────────────
    if (
        not _is_analysis_running()
        and "matrix_computer" in st.session_state
        and st.session_state.matrix_computer is not None
    ):
        st.divider()
        _show_results_summary()


def _run_analysis_with_live_progress(variants: list, title_overrides: dict):
    """
    Start the background thread and show live progress via st.status().

    Uses st.status() as a live-updating container with st.empty()
    placeholders inside.  Polls the module-level _RESULT dict every
    0.5 seconds for progress updates from the thread.

    If the user stays on this page, they see smooth step-by-step progress.
    If they switch tabs, the thread continues and results are picked up
    on the next visit via _check_and_load_results().
    """
    # Start background thread
    _RESULT["status"] = "running"
    _RESULT["progress_step"] = 0
    _RESULT["progress_msg"] = "Starting..."
    st.session_state.analysis_running = True

    thread = threading.Thread(
        target=_run_pipeline_in_thread,
        args=(variants, title_overrides),
        daemon=True,
    )
    thread.start()
    logger.info("Background analysis thread started")

    # Show live progress with st.status()
    step_labels = {
        0: "Starting analysis...",
        1: "Step 1/4 - Loading papers...",
        2: "Step 2/4 - Parallel Processing (Extract, Normalize, Detect)...",
        3: "Step 3/4 - Building matrices...",
        4: "Step 4/4 - Exporting results...",
    }

    with st.status("Running Full Analysis...", expanded=True) as status:
        progress_bar = st.progress(0, text="Starting...")
        step_text = st.empty()
        step_text.markdown("Initializing pipeline...")

        last_step = -1
        last_sub = -1

        # Poll until thread completes (or fails)
        while _RESULT["status"] == "running":
            current_step = _RESULT.get("progress_step", 0)
            current_msg = _RESULT.get("progress_msg", "Processing...")
            sub_current = _RESULT.get("sub_current", 0)
            sub_total = _RESULT.get("sub_total", 0)

            if current_step != last_step or sub_current != last_sub:
                # Compute progress percentage across all 4 steps
                # Each step is worth 25% of the total progress
                if sub_total > 0 and current_step == 2:
                    # Step 2 has sub-progress (per-paper multiprocessing)
                    step_fraction = sub_current / sub_total
                    pct = int(step_fraction * 100)
                    progress_pct = min((1 + step_fraction) / 4, 0.99)
                    label = f"Step 2/4 - Processing papers... {sub_current}/{sub_total} ({pct}%)"
                else:
                    progress_pct = min(current_step / 4, 0.99)
                    label = step_labels.get(current_step, current_msg)

                progress_bar.progress(progress_pct, text=label)
                step_text.markdown(f"**{label}**")
                last_step = current_step
                last_sub = sub_current

            time.sleep(0.3)

        # Thread finished — check result
        if _RESULT["status"] == "complete":
            progress_bar.progress(1.0, text="Analysis complete!")
            step_text.markdown("**All steps completed successfully.**")
            status.update(label="Analysis Complete!", state="complete")

            # Load results into session state
            _check_and_load_results()
            st.rerun()

        elif _RESULT["status"] == "error":
            error_msg = _RESULT.get("error", "Unknown error")
            step_text.markdown(f"**Error:** {error_msg}")
            status.update(label="Analysis Failed", state="error")
            st.session_state.analysis_running = False
            _RESULT["status"] = None
            _RESULT["error"] = None


def _show_polling_progress():
    """
    Show progress for an analysis that was started before a tab switch.

    If the user started analysis, switched tabs, and came back, this
    shows the current progress and polls for updates.
    """
    step = _RESULT.get("progress_step", 0)
    total = 4
    message = _RESULT.get("progress_msg", "Processing...")
    sub_current = _RESULT.get("sub_current", 0)
    sub_total = _RESULT.get("sub_total", 0)

    st.info(
        "**Analysis is running in the background.** "
        "You can switch tabs freely. This page auto-refreshes."
    )

    if sub_total > 0 and step == 2:
        pct = int(sub_current / sub_total * 100)
        progress_pct = min((1 + sub_current / sub_total) / 4, 0.99)
        label = f"Step 2/4: Processing papers... {sub_current}/{sub_total} ({pct}%)"
    else:
        progress_pct = min(step / 4, 0.99)
        label = f"Step {step}/4: {message}"

    st.progress(progress_pct, text=label)

    # Poll for updates
    time.sleep(1)
    st.rerun()


def _run_pipeline_in_thread(variants: list, title_overrides: dict):
    """
    Execute the full analysis pipeline in a background daemon thread.

    This function runs OUTSIDE the Streamlit rerun cycle, so sidebar
    clicks and tab switches do not interrupt it.

    Results are written to the module-level _RESULT dict when complete.
    """
    def update_progress(step, message):
        _RESULT["progress_step"] = step
        _RESULT["progress_msg"] = message

    try:
        # ── Step 1: Loading papers ──────────────────────────────────
        update_progress(1, "Loading papers...")
        extractor = TextExtractor()
        paper_items = list(extractor.paper_id_map.items())
        total_papers = len(paper_items)
        id_to_paper_meta = extractor.get_id_to_paper_meta_map(title_overrides)
        id_to_filename = extractor.get_id_to_filename_map()

        if total_papers == 0:
            _RESULT["status"] = "error"
            _RESULT["error"] = "No paper files found."
            return

        # ── Step 2: Parallel Processing (Extract, Normalize, Detect) 
        _RESULT["sub_current"] = 0
        _RESULT["sub_total"] = total_papers
        update_progress(2, "Parallel Processing (Extract, Normalize, Detect)...")

        raw_texts = {}
        preprocessed = {}
        detection_results = {}
        detection_details = {}

        # Limit to CPU cores safely
        max_workers = max(1, multiprocessing.cpu_count() - 1)
        
        args_list = [(pid, path, variants) for pid, path in paper_items]
        
        # CRITICAL: We use ThreadPoolExecutor instead of ProcessPoolExecutor.
        # Why? Because this project is located inside a OneDrive directory.
        # Spawning new processes forces Python to read the `venv` from disk,
        # which heavily triggers macOS OneDrive "TimeoutError [Errno 60]".
        # Threading avoids this file-system death spiral while still giving us
        # a massive speed boost since PyMuPDF (fitz) releases the GIL during PDF extraction!
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for i, result in enumerate(pool.map(process_single_paper, args_list)):
                paper_id, text, norm_text, detection = result
                presence, details = detection
                raw_texts[paper_id] = text
                preprocessed[paper_id] = norm_text
                detection_results[paper_id] = presence
                detection_details[paper_id] = details
                _RESULT["sub_current"] = i + 1

        if not raw_texts:
            _RESULT["status"] = "error"
            _RESULT["error"] = "No text could be extracted from papers."
            return

        # ── Step 3: Matrix Computation ───────────────────────────────
        update_progress(3, "Building matrices...")
        
        # Initialize detector locally to get dimensions and names for matrices
        preprocessor = TextPreprocessor()
        detector = VariantDetector(variants=variants, preprocessor=preprocessor)
        variant_names = detector.get_variant_names()
        dimension_map = detector.get_dimension_map()

        computer = MatrixComputer()
        computer.set_paper_metadata(id_to_paper_meta)
        paper_variant_df = computer.build_paper_variant_matrix(
            detection_results, variant_names, detection_details
        )
        intersection_df = computer.compute_intersection_matrix(
            dimension_map=dimension_map,
        )
        
        # ── Step 4: Exporting results ────────────────────────────────
        update_progress(4, "Exporting results...")
        computer.export_all()

        # ── Store results ────────────────────────────────────────────
        _RESULT["data"] = {
            "computer": computer,
            "paper_variant_df": paper_variant_df,
            "intersection_df": intersection_df,
            "detection_results": detection_results,
            "detection_details": detection_details,
            "preprocessed": preprocessed,
            "detector": detector,
            "dimension_map": dimension_map,
            "id_to_filename": id_to_filename,
            "id_to_paper_meta": id_to_paper_meta,
        }
        _RESULT["status"] = "complete"
        logger.info("Background analysis completed")

    except Exception as e:
        logger.exception("Background analysis failed")
        _RESULT["status"] = "error"
        _RESULT["error"] = str(e)


def _show_results_summary():
    """Display summary statistics from the last analysis run."""
    st.markdown(sub_header("bar_chart", "Results Summary"), unsafe_allow_html=True)

    computer: MatrixComputer = st.session_state.matrix_computer
    stats = computer.get_summary_stats()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Papers Analyzed", stats.get("total_papers", 0))
    col2.metric("Variants Tracked", stats.get("total_variants", 0))
    col3.metric("Total Detections", stats.get("total_detections", 0))
    col4.metric("Research Gaps", stats.get("research_gaps", 0))

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Avg Variants/Paper", f"{stats.get('avg_variants_per_paper', 0):.1f}")
    col6.metric("Avg Papers/Variant", f"{stats.get('avg_papers_per_variant', 0):.1f}")
    col7.metric("Valid Pairs", stats.get("total_valid_pairs", 0))
    col8.metric("Excluded Pairs", stats.get("excluded_pairs", 0))

    # Paper ID mapping
    if "paper_meta_map" in st.session_state:
        with st.expander("Paper ID Reference (P-ID -> Title)"):
            meta_map = st.session_state.paper_meta_map
            for pid, meta in sorted(meta_map.items(), key=lambda x: int(x[0][1:])):
                title = meta.get("title", "")
                filename = meta.get("filename", "")
                st.markdown(f"**{pid}** -> {title}  ")
                st.caption(f"Filename: {filename}")
    elif "paper_id_map" in st.session_state:
        with st.expander("Paper ID Reference (P-ID -> Filename)"):
            id_map = st.session_state.paper_id_map
            for pid, fname in sorted(id_map.items(), key=lambda x: int(x[0][1:])):
                st.markdown(f"**{pid}** -> {fname}")
