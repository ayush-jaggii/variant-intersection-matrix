"""
Paper Manager Component
=======================

Streamlit UI component for uploading and managing research papers.

Features:
    • Upload PDF and TXT files
    • Display papers with auto-generated sequential IDs (P1, P2, ...)
    • Show paper ID → filename mapping
    • Delete individual papers or all at once
"""

import streamlit as st
from pathlib import Path
from typing import List

from config.settings import PAPERS_DIR, MAX_UPLOAD_SIZE_MB, SUPPORTED_PAPER_EXTENSIONS
from utils.helpers import format_file_size, list_paper_files, generate_paper_id_map
from interface.design import section_header, sub_header, icon


def render_paper_manager():
    """Render the Paper Management section of the UI."""
    st.markdown(section_header("description", "Paper Management"), unsafe_allow_html=True)

    # ── Upload Section ───────────────────────────────────────────────
    st.markdown(sub_header("cloud_upload", "Upload Papers"), unsafe_allow_html=True)

    # Build the accepted types list from settings (removes the dot prefix)
    accepted_types = [ext.lstrip(".") for ext in SUPPORTED_PAPER_EXTENSIONS]
    type_label = ", ".join(ext.upper() for ext in accepted_types)

    uploaded_files = st.file_uploader(
        f"Upload research papers ({type_label})",
        type=accepted_types,
        accept_multiple_files=True,
        help=f"Maximum {MAX_UPLOAD_SIZE_MB} MB per file. Select multiple files at once.",
        key="paper_uploader",
    )

    if uploaded_files:
        _handle_uploads(uploaded_files)

    st.divider()

    # ── Paper Library ────────────────────────────────────────────────
    st.markdown(sub_header("layers", "Paper Library"), unsafe_allow_html=True)
    papers = list_paper_files(PAPERS_DIR)

    if not papers:
        st.info("No papers uploaded yet. Upload PDF or TXT files above to get started.")
        return

    # Build paper ID mapping (P1, P2, ...)
    id_map = generate_paper_id_map(PAPERS_DIR)
    # Reverse lookup: filename → paper_id
    filename_to_id = {fpath.name: pid for pid, fpath in id_map.items()}

    # Summary stats
    total_size = sum(p.stat().st_size for p in papers)
    col1, col2 = st.columns(2)
    col1.metric("Total Papers", len(papers))
    col2.metric("Total Size", format_file_size(total_size))

    # Paper list with P-ID, filename, size, and delete
    st.write("")
    for paper_path in papers:
        paper_id = filename_to_id.get(paper_path.name, "?")
        _render_paper_row(paper_path, paper_id)

    # Bulk actions
    st.divider()
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Delete All Papers", type="secondary"):
            st.session_state["confirm_delete_all"] = True

    if st.session_state.get("confirm_delete_all", False):
        st.warning("This will delete ALL uploaded papers. Are you sure?")
        c1, c2, c3 = st.columns([1, 1, 4])
        with c1:
            if st.button("Yes, delete all"):
                _delete_all_papers()
                st.session_state["confirm_delete_all"] = False
                st.rerun()
        with c2:
            if st.button("Cancel"):
                st.session_state["confirm_delete_all"] = False
                st.rerun()


def _handle_uploads(uploaded_files) -> None:
    """Process uploaded files and save to papers directory."""
    saved_count = 0
    skipped_count = 0

    for uploaded_file in uploaded_files:
        dest_path = PAPERS_DIR / uploaded_file.name

        if dest_path.exists():
            skipped_count += 1
            continue

        with open(dest_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_count += 1

    if saved_count > 0:
        st.success(f"Uploaded {saved_count} paper(s) successfully.")
    if skipped_count > 0:
        st.info(f"Skipped {skipped_count} paper(s) - already exist.")


def _render_paper_row(paper_path: Path, paper_id: str) -> None:
    """Render a single paper row with P-ID, filename, size, and delete button."""
    file_size = format_file_size(paper_path.stat().st_size)
    ext_icon = "picture_as_pdf" if paper_path.suffix.lower() == ".pdf" else "article"

    col1, col2, col3, col4 = st.columns([1, 5, 2, 1])
    with col1:
        # Show paper ID badge
        st.markdown(
            f'<span style="background:#1B2A4A;color:#fff;padding:2px 8px;'
            f'border-radius:4px;font-weight:600;font-size:0.85rem">{paper_id}</span>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'{icon(ext_icon, size=16, color="#3B82B0")} {paper_path.name}',
            unsafe_allow_html=True,
        )
    with col3:
        st.text(file_size)
    with col4:
        if st.button("Delete", key=f"del_{paper_path.name}", help=f"Delete {paper_path.name}"):
            paper_path.unlink()
            st.rerun()


def _delete_all_papers() -> None:
    """Delete all paper files from papers directory."""
    for paper_file in list_paper_files(PAPERS_DIR):
        paper_file.unlink()
    st.success("All papers deleted.")


def get_paper_count() -> int:
    """Return the number of uploaded papers."""
    return len(list_paper_files(PAPERS_DIR))
