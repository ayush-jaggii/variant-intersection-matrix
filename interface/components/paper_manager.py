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
import csv
import io
from pathlib import Path
from typing import Dict

from config.settings import PAPERS_DIR, MAX_UPLOAD_SIZE_MB, SUPPORTED_PAPER_EXTENSIONS, CACHE_DIR
from core.text_extraction import clean_title, get_document_name
from utils.helpers import format_file_size, list_paper_files, generate_paper_id_map, load_json, save_json
from interface.design import section_header, sub_header, icon


TITLE_OVERRIDES_FILE = CACHE_DIR / "paper_title_overrides.json"


def _load_title_overrides() -> Dict[str, str]:
    """Load user title overrides from disk safely."""
    try:
        data = load_json(TITLE_OVERRIDES_FILE)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception:
        pass
    return {}


def _save_title_overrides(overrides: Dict[str, str]) -> None:
    """Persist title overrides safely."""
    try:
        save_json(overrides, TITLE_OVERRIDES_FILE)
    except Exception:
        pass


def render_paper_manager():
    """Render the Paper Management section of the UI."""
    st.markdown(section_header("description", "Paper Management"), unsafe_allow_html=True)
    st.info("Upload PDF and TXT files containing the research papers you want to analyze.")

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

    if "paper_title_overrides" not in st.session_state:
        st.session_state.paper_title_overrides = _load_title_overrides()

    title_overrides: Dict[str, str] = st.session_state.paper_title_overrides
    current_names = {p.name for p in papers}
    stale_names = [name for name in title_overrides if name not in current_names]
    for stale_name in stale_names:
        title_overrides.pop(stale_name, None)

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

    # Paper list with P-ID, title, filename, size, and delete
    st.write("")
    papers_metadata = []

    for paper_path in papers:
        paper_id = filename_to_id.get(paper_path.name, "?")
        detected_title = get_document_name(paper_path)
        suggested_title = title_overrides.get(paper_path.name, detected_title)
        edited_title = _render_paper_row(paper_path, paper_id, detected_title, suggested_title)

        cleaned_edited = clean_title(edited_title)
        final_title = cleaned_edited or edited_title.strip() or detected_title
        if final_title != title_overrides.get(paper_path.name):
            title_overrides[paper_path.name] = final_title

        papers_metadata.append({
            "paper_id": paper_id,
            "filename": paper_path.name,
            "detected_name": detected_title,
            "given_name": final_title,
            "title": final_title,
        })

    st.session_state.paper_title_overrides = title_overrides
    st.session_state.papers_metadata = papers_metadata
    _save_title_overrides(title_overrides)

    st.markdown(sub_header("download", "Export Name Mapping"), unsafe_allow_html=True)
    name_map_csv = _build_name_mapping_csv(papers_metadata)
    st.download_button(
        "Download Paper Name Mapping CSV",
        data=name_map_csv,
        file_name="paper_name_mapping.csv",
        mime="text/csv",
        help="Exports paper number (P1/P2/P3), PDF/TXT filename, detected name, and edited name.",
    )

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


def _render_paper_row(
    paper_path: Path,
    paper_id: str,
    detected_name: str,
    suggested_name: str,
) -> str:
    """Render one paper row with detected + editable paper names."""
    file_size = format_file_size(paper_path.stat().st_size)
    ext_icon = "picture_as_pdf" if paper_path.suffix.lower() == ".pdf" else "article"

    col1, col2, col3, col4 = st.columns([1, 6, 2, 1])
    with col1:
        # Show paper ID badge
        st.markdown(
            f'<span style="background:#1B2A4A;color:#fff;padding:2px 8px;'
            f'border-radius:4px;font-weight:600;font-size:0.85rem">{paper_id}</span>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'{icon(ext_icon, size=16, color="#3B82B0")} <strong>{suggested_name}</strong><br>'
            f'<span style="color:#334155; font-size:0.85em">Detected: {detected_name}</span><br>'
            f'<span style="color:#64748B; font-size:0.9em">{paper_path.name}</span>',
            unsafe_allow_html=True,
        )
        edited_title = st.text_input(
            "Edit Suggested Name",
            value=suggested_name,
            key=f"title_edit_{paper_path.name}",
            label_visibility="collapsed",
            help="Optional: edit the suggested paper name before analysis/export.",
        )
    with col3:
        st.text(file_size)
    with col4:
        if st.button("Delete", key=f"del_{paper_path.name}", help=f"Delete {paper_path.name}"):
            paper_path.unlink()
            if "paper_title_overrides" in st.session_state:
                st.session_state.paper_title_overrides.pop(paper_path.name, None)
                _save_title_overrides(st.session_state.paper_title_overrides)
            st.rerun()

    return edited_title


def _build_name_mapping_csv(papers_metadata: list) -> str:
    """Build CSV text for paper-id to filename/name mapping export."""
    out = io.StringIO()
    writer = csv.DictWriter(
        out,
        fieldnames=["paper_number", "pdf_filename", "detected_name", "given_name"],
    )
    writer.writeheader()
    for row in papers_metadata:
        writer.writerow(
            {
                "paper_number": row.get("paper_id", ""),
                "pdf_filename": row.get("filename", ""),
                "detected_name": row.get("detected_name", ""),
                "given_name": row.get("given_name", row.get("title", "")),
            }
        )
    return out.getvalue()


def _delete_all_papers() -> None:
    """Delete all paper files from papers directory."""
    for paper_file in list_paper_files(PAPERS_DIR):
        paper_file.unlink()
    if "paper_title_overrides" in st.session_state:
        st.session_state.paper_title_overrides = {}
    _save_title_overrides({})
    st.success("All papers deleted.")


def get_paper_count() -> int:
    """Return the number of uploaded papers."""
    return len(list_paper_files(PAPERS_DIR))
