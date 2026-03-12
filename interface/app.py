"""
Variant Intersection Matrix Analyzer — Streamlit Application
=============================================================

Main application entry point. Configures the page, sets up navigation,
and delegates rendering to the component modules.

Navigation is flexible — users can jump to any step at any time via
the sidebar radio buttons.  No rigid ordering is enforced.
"""

import sys
import logging
from pathlib import Path

import streamlit as st

# ── Ensure project root is on sys.path ───────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import PAGE_TITLE, PAGE_ICON, PAGE_LAYOUT
from interface.design import get_theme_css, icon, section_header, status_badge, COLORS
from interface.components.paper_manager import render_paper_manager, get_paper_count
from interface.components.variant_manager import (
    render_variant_manager,
    get_variant_count,
    get_dimension_count,
    _load_variants,
)
from interface.components.analysis_runner import (
    render_analysis_runner,
    _check_and_load_results,
)
from interface.components.matrix_viewer import render_matrix_viewer

# ── Logging Setup ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-25s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)

# ═══════════════════════════════════════════════════════════════════════════
# Page Configuration
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=PAGE_LAYOUT,
    initial_sidebar_state="expanded",
)

# ── Inject Design System ─────────────────────────────────────────────────
st.markdown(get_theme_css(), unsafe_allow_html=True)

# ── Check for background analysis results on every rerun ─────────────────
# This ensures results are picked up even when the user is on another tab.
_check_and_load_results()

# ── Load variants on startup if not already loaded ──────────────────────────
# Prevents the "Run Analysis" tab from requiring a visit to the "Variants" tab first
if "variants" not in st.session_state:
    st.session_state.variants = _load_variants()


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar Navigation
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        f'<h2 style="display:flex;align-items:center;gap:10px;border:none;padding-bottom:0;margin-bottom:0">'
        f'{icon("science", size=26, color="#FFFFFF")}'
        f'<span style="color:#FFFFFF;font-weight:700">VIM Analyzer</span>'
        f'</h2>',
        unsafe_allow_html=True,
    )
    st.caption("Variant Intersection Matrix")
    st.divider()

    # Navigation — flexible ordering, no strict sequence enforced
    page = st.radio(
        "Navigation",
        [
            "Papers",
            "Variants",
            "Run Analysis",
            "View Results",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    # Status indicators
    st.markdown(
        f'<h3 style="display:flex;align-items:center;gap:8px;margin-bottom:12px">'
        f'{icon("dashboard", size=20, color="#E2E8F0")}'
        f'<span style="color:#FFFFFF">System Status</span></h3>',
        unsafe_allow_html=True,
    )

    paper_count = get_paper_count()
    variant_count = get_variant_count()
    dimension_count = get_dimension_count()
    analysis_done = st.session_state.get("analysis_complete", False)
    analysis_running = st.session_state.get("analysis_running", False)

    if analysis_running:
        analysis_icon = "sync"
        analysis_text = "Running"
    elif analysis_done:
        analysis_icon = "check_circle"
        analysis_text = "Complete"
    else:
        analysis_icon = "sync"
        analysis_text = "Pending"

    status_items = [
        ("description", "Papers", str(paper_count)),
        ("category", "Dimensions", str(dimension_count)),
        ("biotech", "Variants", str(variant_count)),
        (analysis_icon, "Analysis", analysis_text),
    ]

    for ico, label, value in status_items:
        st.markdown(
            f'<div class="vim-status-row">'
            f'{icon(ico, size=16, color="#94A3B8")}'
            f'<span class="vim-status-label">{label}:</span>'
            f'<span class="vim-status-value">{value}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.caption("Variant Intersection Matrix Analyzer")
    st.caption("Academic Research Tool")


# ═══════════════════════════════════════════════════════════════════════════
# Main Content Area
# ═══════════════════════════════════════════════════════════════════════════

# Title
st.markdown(
    f'<h1 style="display:flex;align-items:center;gap:12px;margin-bottom:1rem">'
    f'{icon("science", size=34, color=COLORS["secondary"])}'
    f'{PAGE_TITLE}'
    f'</h1>',
    unsafe_allow_html=True,
)

# Route to selected page — flexible navigation, any order
if page == "Papers":
    render_paper_manager()

elif page == "Variants":
    render_variant_manager()

elif page == "Run Analysis":
    render_analysis_runner()

elif page == "View Results":
    render_matrix_viewer()
