"""
Variant Manager Component
=========================

Streamlit UI for defining, editing, and managing dimensions, variants,
and their alternate_names.

Features:
    • Organize variants under dimensions
    • Add/edit/delete individual variants
    • Import variants from CSV, JSON, or Excel
    • Export variant definitions as JSON
    • Search/filter the variant list
    • Add new dimensions

Internal Data Model (session state):
    st.session_state.variants is a flat list:
        [
          {"dimension": "Dim", "name": "Var", "alternate_names": ["syn1", ...]},
          ...
        ]

    Saved to disk in dimension-grouped format:
        {"dimensions": {"Dim": {"Var": ["syn1", ...]}}}
"""

import json
import streamlit as st
from typing import Any, Dict, List, Optional

from config.settings import VARIANTS_FILE
from utils.helpers import (
    load_json,
    save_json,
    dimensions_to_flat_list,
    flat_list_to_dimensions,
    parse_variants_from_csv,
    parse_variants_from_excel,
    parse_variants_from_json,
)
from interface.design import section_header, sub_header, icon, COLORS


def render_variant_manager():
    """Render the Variant & Dimension Management section of the UI."""
    st.markdown(
        section_header("biotech", "Dimension & Variant Manager"),
        unsafe_allow_html=True,
    )
    st.info("Define the dimensions, variants, and their alternate names that you want the analyzer to search for.")

    # Initialize session state for variants
    if "variants" not in st.session_state:
        st.session_state.variants = _load_variants()

    # ── Tabs ─────────────────────────────────────────────────────────
    tab_manage, tab_add, tab_import = st.tabs([
        "Manage Variants",
        "Add Variant",
        "Import / Export",
    ])

    with tab_manage:
        _render_variant_list()

    with tab_add:
        _render_add_variant_form()

    with tab_import:
        _render_import_export()


# ═══════════════════════════════════════════════════════════════════════════
# Manage Tab
# ═══════════════════════════════════════════════════════════════════════════

def _render_variant_list():
    """Render the filterable list of existing variants, grouped by dimension."""
    variants = st.session_state.variants

    if not variants:
        st.info(
            "No variants defined yet. Use the 'Add Variant' tab or "
            "import from CSV/JSON/Excel."
        )
        return

    # Summary metrics
    dimensions = _get_dimensions(variants)
    col1, col2, col3 = st.columns(3)
    col1.metric("Dimensions", len(dimensions))
    col2.metric("Total Variants", len(variants))
    total_alternate_names = sum(len(v.get("alternate_names", [])) for v in variants)
    col3.metric("Total Alternate Names", total_alternate_names)

    # Search filter
    search = st.text_input(
        "Search variants",
        placeholder="Type to filter by variant name, alternate_name, or dimension...",
        key="variant_search",
    )

    # Filter variants
    filtered = variants
    if search:
        search_lower = search.lower()
        filtered = [
            v for v in variants
            if search_lower in v["name"].lower()
            or search_lower in v.get("dimension", "").lower()
            or any(search_lower in s.lower() for s in v.get("alternate_names", []))
        ]

    st.write(f"Showing {len(filtered)} of {len(variants)} variants")
    st.divider()

    # Group by dimension and render
    grouped = _group_by_dimension(filtered)
    for dim_name, dim_variants in sorted(grouped.items()):
        _render_dimension_group(dim_name, dim_variants)


def _render_dimension_group(dim_name: str, variants: List[Dict[str, Any]]):
    """Render a collapsible dimension group with its variants."""
    with st.expander(
        f"{dim_name}  ({len(variants)} variant{'s' if len(variants) != 1 else ''})",
        expanded=True,
    ):
        for idx, variant in enumerate(variants):
            _render_variant_card(variant, idx)


def _render_variant_card(variant: Dict[str, Any], idx: int):
    """Render a single variant card with edit/delete capabilities."""
    name = variant["name"]
    dimension = variant.get("dimension", "Uncategorized")
    alternate_names = variant.get("alternate_names", [])
    edit_key = f"editing_{dimension}_{name}"

    # Check if in edit mode
    if st.session_state.get(edit_key, False):
        _render_edit_form(variant, idx)
        return

    col1, col2, col3 = st.columns([4, 1, 1])
    with col1:
        st.markdown(f"**{name}**")
        if alternate_names:
            alternate_name_tags = ", ".join(f"`{s}`" for s in alternate_names)
            st.caption(f"Alternate Names: {alternate_name_tags}")
        else:
            st.caption("_No alternate_names defined_")
    with col2:
        if st.button("Edit", key=f"edit_{dimension}_{name}_{idx}", help="Edit variant"):
            st.session_state[edit_key] = True
            st.rerun()
    with col3:
        if st.button("Delete", key=f"del_{dimension}_{name}_{idx}", help="Delete variant"):
            st.session_state.variants = [
                v for v in st.session_state.variants
                if not (v["name"] == name and v.get("dimension") == dimension)
            ]
            _save_variants()
            st.rerun()
    st.divider()


def _render_edit_form(variant: Dict[str, Any], idx: int):
    """Render inline edit form for a variant."""
    name = variant["name"]
    dimension = variant.get("dimension", "Uncategorized")
    alternate_names = variant.get("alternate_names", [])
    edit_key = f"editing_{dimension}_{name}"

    existing_dims = _get_dimensions(st.session_state.variants)

    st.markdown(f"**Editing: {name}**")
    new_dim = st.text_input(
        "Dimension",
        value=dimension,
        key=f"edit_dim_{dimension}_{name}_{idx}",
        help="Type an existing or new dimension name.",
    )
    new_name = st.text_input(
        "Variant Name",
        value=name,
        key=f"edit_name_{dimension}_{name}_{idx}",
    )
    new_alternate_names = st.text_area(
        "Alternate Names (one per line)",
        value="\n".join(alternate_names),
        key=f"edit_syns_{dimension}_{name}_{idx}",
        height=100,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save", key=f"save_{dimension}_{name}_{idx}"):
            parsed_alternate_names = [
                s.strip() for s in new_alternate_names.split("\n") if s.strip()
            ]
            for v in st.session_state.variants:
                if v["name"] == name and v.get("dimension") == dimension:
                    v["dimension"] = new_dim.strip() or "Uncategorized"
                    v["name"] = new_name.strip()
                    v["alternate_names"] = parsed_alternate_names
                    break
            _save_variants()
            st.session_state[edit_key] = False
            st.rerun()
    with col2:
        if st.button("Cancel", key=f"cancel_{dimension}_{name}_{idx}"):
            st.session_state[edit_key] = False
            st.rerun()
    st.divider()


# ═══════════════════════════════════════════════════════════════════════════
# Add Tab
# ═══════════════════════════════════════════════════════════════════════════

def _render_add_variant_form():
    """Render the form for adding a new variant with dimension support."""
    existing_dims = _get_dimensions(st.session_state.variants)

    with st.form("add_variant_form", clear_on_submit=True):
        st.markdown(sub_header("add", "Add New Variant"), unsafe_allow_html=True)

        # Dimension selection: choose existing or type new
        dim_options = sorted(existing_dims) + ["+ New Dimension"]
        selected_dim = st.selectbox(
            "Dimension *",
            dim_options if dim_options[:-1] else ["+ New Dimension"],
            help="Select an existing dimension or create a new one.",
        )

        new_dim_name = ""
        if selected_dim == "+ New Dimension":
            new_dim_name = st.text_input(
                "New Dimension Name *",
                placeholder="e.g., Product Energy Consumption",
            )

        dimension = new_dim_name.strip() if selected_dim == "+ New Dimension" else selected_dim

        name = st.text_input(
            "Variant Name *",
            placeholder="e.g., Energy Consuming",
            help="The primary name for this variant.",
        )
        alternate_names_text = st.text_area(
            "Alternate Names (one per line)",
            placeholder="e.g.,\nenergy consuming\nelectric powered\nenergy-intensive",
            help="Enter alternate_names that should also be detected as this variant.",
            height=150,
        )

        submitted = st.form_submit_button("Add Variant", type="primary")

        if submitted:
            if not dimension:
                st.error("Dimension name is required.")
                return
            if not name or not name.strip():
                st.error("Variant name is required.")
                return

            clean_name = name.strip()

            # Check for duplicates within the same dimension
            for v in st.session_state.variants:
                if (v["name"].lower() == clean_name.lower()
                        and v.get("dimension", "").lower() == dimension.lower()):
                    st.error(
                        f"A variant named '{clean_name}' already exists "
                        f"in dimension '{dimension}'."
                    )
                    return

            alternate_names = [s.strip() for s in alternate_names_text.split("\n") if s.strip()]

            new_variant = {
                "dimension": dimension,
                "name": clean_name,
                "alternate_names": alternate_names,
            }
            st.session_state.variants.append(new_variant)
            _save_variants()
            st.success(
                f"Added variant '{clean_name}' to dimension '{dimension}' "
                f"with {len(alternate_names)} alternate_name(s)."
            )


# ═══════════════════════════════════════════════════════════════════════════
# Import / Export Tab
# ═══════════════════════════════════════════════════════════════════════════

def _render_import_export():
    """Render import/export functionality with CSV, JSON, and Excel support."""

    # ── Export ────────────────────────────────────────────────────────
    st.markdown(sub_header("cloud_download", "Export Variants"), unsafe_allow_html=True)
    if st.session_state.variants:
        dimensions = flat_list_to_dimensions(st.session_state.variants)
        export_data = json.dumps(
            {"dimensions": dimensions},
            indent=2,
            ensure_ascii=False,
        )
        st.download_button(
            label="Download variants.json",
            data=export_data,
            file_name="variants.json",
            mime="application/json",
        )

        with st.expander("Preview JSON"):
            st.json({"dimensions": dimensions})
    else:
        st.info("No variants to export.")

    st.divider()

    # ── Import ────────────────────────────────────────────────────────
    st.markdown(sub_header("upload_file", "Import Variants"), unsafe_allow_html=True)
    st.caption(
        "Supported formats: **CSV**, **JSON**, **Excel (.xlsx)**. "
        "CSV and Excel files must have columns: `dimension`, `variant`, `alternate_name`."
    )

    uploaded = st.file_uploader(
        "Upload a variant definitions file",
        type=["json", "csv", "xlsx"],
        key="variant_import",
    )

    if uploaded:
        _handle_variant_import(uploaded)


def _handle_variant_import(uploaded):
    """Process an uploaded variant file (CSV, JSON, or Excel)."""
    filename = uploaded.name.lower()

    try:
        # Parse based on file type → canonical dimension dict
        if filename.endswith(".csv"):
            content = uploaded.read().decode("utf-8")
            dimension_dict = parse_variants_from_csv(content)
        elif filename.endswith(".xlsx"):
            content_bytes = uploaded.read()
            dimension_dict = parse_variants_from_excel(content_bytes)
        elif filename.endswith(".json"):
            content = uploaded.read().decode("utf-8")
            dimension_dict = parse_variants_from_json(content)
        else:
            st.error(f"Unsupported file type: {filename}")
            return

        # Convert to flat list for preview
        imported_flat = dimensions_to_flat_list(dimension_dict)
        dim_count = len(dimension_dict)
        var_count = len(imported_flat)
        syn_count = sum(len(v.get("alternate_names", [])) for v in imported_flat)

        st.success(
            f"Parsed **{var_count}** variants across **{dim_count}** dimensions "
            f"with **{syn_count}** total alternate_names."
        )

        # Preview
        with st.expander("Preview imported data"):
            for dim_name, variants in sorted(dimension_dict.items()):
                st.markdown(f"**{dim_name}**")
                for var_name, syns in variants.items():
                    st.markdown(f"- {var_name}: {', '.join(syns) if syns else '(no alternate_names)'}")

        # Import mode
        import_mode = st.radio(
            "Import mode",
            ["Replace all", "Merge (add new, keep existing)"],
            key="import_mode",
        )

        if st.button("Import", type="primary"):
            if import_mode == "Replace all":
                st.session_state.variants = imported_flat
            else:
                # Merge: add only variants that don't exist
                existing_keys = {
                    (v.get("dimension", "").lower(), v["name"].lower())
                    for v in st.session_state.variants
                }
                new_variants = [
                    v for v in imported_flat
                    if (v.get("dimension", "").lower(), v["name"].lower()) not in existing_keys
                ]
                st.session_state.variants.extend(new_variants)
                st.info(
                    f"Added {len(new_variants)} new variant(s). "
                    f"Skipped {len(imported_flat) - len(new_variants)} duplicate(s)."
                )

            _save_variants()
            st.success(f"Import complete!")
            st.rerun()

    except ValueError as e:
        st.error(f"Import error: {e}")
    except json.JSONDecodeError:
        st.error("Invalid JSON file.")
    except Exception as e:
        st.error(f"Failed to import: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# Data Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _get_dimensions(variants: List[Dict[str, Any]]) -> List[str]:
    """Get sorted list of unique dimension names."""
    dims = set()
    for v in variants:
        dims.add(v.get("dimension", "Uncategorized"))
    return sorted(dims)


def _group_by_dimension(
    variants: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Group a flat variant list by dimension name."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for v in variants:
        dim = v.get("dimension", "Uncategorized")
        if dim not in grouped:
            grouped[dim] = []
        grouped[dim].append(v)
    return grouped


def _load_variants() -> List[Dict[str, Any]]:
    """
    Load variants from disk.

    Handles both the new dimension format and legacy flat format.
    """
    try:
        data = load_json(VARIANTS_FILE)

        # New dimension format
        if isinstance(data, dict) and "dimensions" in data:
            return dimensions_to_flat_list(data["dimensions"])

        # Legacy format
        if isinstance(data, dict) and "variants" in data:
            variants = data["variants"]
            for v in variants:
                if "dimension" not in v:
                    v["dimension"] = "Uncategorized"
            return variants

        if isinstance(data, list):
            for v in data:
                if "dimension" not in v:
                    v["dimension"] = "Uncategorized"
            return data

        return []
    except FileNotFoundError:
        return []


def _save_variants():
    """
    Save current variants to disk in the dimension-grouped format.

    Format: {"dimensions": {"Dim": {"Var": ["syn1", ...]}}}
    """
    dimensions = flat_list_to_dimensions(st.session_state.variants)
    save_json({"dimensions": dimensions}, VARIANTS_FILE)


def get_variant_count() -> int:
    """Return the number of defined variants."""
    try:
        data = load_json(VARIANTS_FILE)
        if isinstance(data, dict) and "dimensions" in data:
            return sum(
                len(variants)
                for variants in data["dimensions"].values()
            )
        if isinstance(data, dict) and "variants" in data:
            return len(data["variants"])
        if isinstance(data, list):
            return len(data)
        return 0
    except FileNotFoundError:
        return 0


def get_dimension_count() -> int:
    """Return the number of defined dimensions."""
    try:
        data = load_json(VARIANTS_FILE)
        if isinstance(data, dict) and "dimensions" in data:
            return len(data["dimensions"])
        return 0
    except FileNotFoundError:
        return 0
