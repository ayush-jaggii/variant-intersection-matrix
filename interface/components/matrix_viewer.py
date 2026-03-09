"""
Matrix Viewer Component
=======================

Interactive Streamlit component for viewing:
    • Paper × Variant binary matrix (heatmap) with P-IDs
    • Variant × Variant intersection matrix (heatmap) with dimension exclusion
    • Cell drill-down: click a pair to see supporting papers
    • Filterable matrix: select specific variants to show a sub-matrix
    • Research gap highlighting (cross-dimension only)
    • Manual validation / override of detections (single + pair)

Same-dimension cells:
    The intersection heatmap shows excluded (same-dimension) cells in a
    distinct gray color.  These are not research gaps — they are
    structurally invalid comparisons.

Filterable Matrix:
    Users can select a subset of variants from a multi-select widget.
    The intersection heatmap and paper-variant matrix update in real time
    to show only the selected variants.  This is done by slicing the
    DataFrames — no recomputation required.

Heatmap Text Visibility:
    Cell text color is dynamically adjusted based on the background
    intensity. Dark cells get white text; light cells get dark text.
    This ensures readability across the full color gradient.

Manual Pair Validation:
    In addition to overriding single variant detection, users can confirm
    that a specific paper discusses a combination of two variants.
    This forces both variants to "present" for that paper and recomputes
    the intersection matrix live.  Changes flow to CSV exports immediately.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import List, Optional, Tuple

from config.settings import HEATMAP_COLORSCALE, HEATMAP_ZERO_COLOR, OUTPUT_DIR
from core.matrix_computation import MatrixComputer, EXCLUDED_PAIR_VALUE
from interface.design import section_header, sub_header, icon, COLORS


def render_matrix_viewer():
    """Render the interactive matrix viewer."""
    st.markdown(section_header("grid_view", "Matrix Viewer"), unsafe_allow_html=True)

    if not st.session_state.get("analysis_complete", False):
        st.info("Run the analysis first to view matrices.")
        return

    computer: MatrixComputer = st.session_state.matrix_computer
    paper_variant_df: pd.DataFrame = st.session_state.paper_variant_df
    intersection_df: pd.DataFrame = st.session_state.intersection_df
    dimension_map = st.session_state.get("dimension_map", {})

    # ── Tabs ─────────────────────────────────────────────────────────
    tab_intersection, tab_paper_variant, tab_gaps, tab_validation, tab_download = st.tabs([
        "Intersection Matrix",
        "Paper x Variant Matrix",
        "Research Gaps",
        "Manual Validation",
        "Download Results",
    ])

    with tab_intersection:
        _render_intersection_matrix(intersection_df, computer, dimension_map)

    with tab_paper_variant:
        _render_paper_variant_matrix(paper_variant_df)

    with tab_gaps:
        _render_research_gaps(computer, dimension_map)

    with tab_validation:
        _render_manual_validation(paper_variant_df, computer)

    with tab_download:
        _render_download_results()


# ═══════════════════════════════════════════════════════════════════════════
# Intersection Matrix
# ═══════════════════════════════════════════════════════════════════════════

def _render_intersection_matrix(
    intersection_df: pd.DataFrame,
    computer: MatrixComputer,
    dimension_map: dict,
):
    """Render the Variant × Variant intersection matrix as an interactive heatmap."""
    st.markdown(
        sub_header("link", "Variant × Variant Intersection Matrix"),
        unsafe_allow_html=True,
    )
    st.caption(
        "Each cell shows how many papers discuss **both** variants. "
        "Diagonal = total papers for that variant. "
        "Gray cells = same-dimension pairs (excluded). "
        "Select a pair below to see supporting papers."
    )

    # ── Variant Filter (Filterable Matrix — Req #2) ──────────────────
    # Users select a subset of variants; the heatmap updates in real time.
    # Filtering is a simple DataFrame slice — no recomputation needed.
    all_variants = list(intersection_df.columns)

    selected_variants = st.multiselect(
        "Filter variants (leave empty to show all)",
        options=all_variants,
        default=[],
        key="filter_intersection_variants",
        help="Select specific variants to display a filtered sub-matrix.",
    )

    # Apply filter: slice both rows and columns
    if selected_variants:
        display_df = intersection_df.loc[selected_variants, selected_variants]
    else:
        display_df = intersection_df

    # Create heatmap with dynamic text colors
    fig = _create_intersection_heatmap(display_df, dimension_map)
    st.plotly_chart(fig, use_container_width=True, key="intersection_heatmap")

    # ── Cell Drill-Down ──────────────────────────────────────────────
    st.divider()
    st.markdown(sub_header("search", "Drill Down into a Pair"), unsafe_allow_html=True)

    variant_names = list(display_df.columns)
    col1, col2 = st.columns(2)

    with col1:
        va = st.selectbox("Variant A", variant_names, key="drilldown_va")
    with col2:
        vb = st.selectbox("Variant B", variant_names, key="drilldown_vb")

    if va and vb:
        # Show dimension info
        dim_a = dimension_map.get(va, "-")
        dim_b = dimension_map.get(vb, "-")
        st.caption(f"**{va}** [{dim_a}]  ×  **{vb}** [{dim_b}]")

        if va == vb:
            papers = computer.get_papers_for_variant(va)
            st.info(f"**{va}** appears in **{len(papers)}** paper(s).")
        elif computer.is_same_dimension_pair(va, vb):
            st.warning(
                f"**Excluded pair:** {va} and {vb} belong to the same "
                f"dimension (**{dim_a}**) and cannot be meaningfully paired."
            )
            papers = []
        else:
            papers = computer.get_papers_for_pair(va, vb)
            count = int(intersection_df.loc[va, vb])
            if count > 0:
                st.success(
                    f"**{count}** paper(s) discuss both **{va}** and **{vb}**:"
                )
            else:
                st.warning(
                    f"**Research Gap:** No papers discuss both **{va}** and **{vb}**."
                )

        if papers:
            id_map = st.session_state.get("paper_id_map", {})
            for paper_id in papers:
                fname = id_map.get(paper_id, "")
                display = f"- **{paper_id}** → {fname}" if fname else f"- `{paper_id}`"
                st.markdown(display)


def _get_dynamic_text_color(value: float, max_value: float) -> str:
    """
    Determine text color for a heatmap cell based on background intensity.

    How this works:
        The heatmap uses a gradient from light (#EEF1F6) to dark (#1B2A4A).
        Values above ~40% of the max are on dark backgrounds, so we switch
        text to white for readability.

    Args:
        value:     The cell's numeric value.
        max_value: The maximum value in the matrix (for normalization).

    Returns:
        CSS color string — white for dark backgrounds, near-black for light.
    """
    if max_value <= 0:
        return COLORS["text"]
    ratio = value / max_value
    # Threshold at 40%: above this the blue gradient is dark enough
    # that black text becomes hard to read
    if ratio > 0.40:
        return "#FFFFFF"
    return COLORS["text"]


def _create_intersection_heatmap(
    df: pd.DataFrame,
    dimension_map: dict,
) -> go.Figure:
    """
    Create a Plotly heatmap for the intersection matrix with dynamic text colors.

    Dynamic text color:
        Each cell's text color is set independently based on the cell value
        relative to the max.  Dark cells get white text; light cells get
        dark text.  This is done via a per-cell font color array passed
        to Plotly's textfont parameter.
    """
    labels = list(df.columns)
    raw_values = df.values.copy().astype(float)

    # Replace excluded cells (-1) with NaN for Plotly rendering
    display_values = raw_values.copy()
    display_values[display_values == EXCLUDED_PAIR_VALUE] = np.nan

    # Compute max for dynamic text color thresholding
    valid_values = raw_values[raw_values != EXCLUDED_PAIR_VALUE]
    max_val = float(np.nanmax(valid_values)) if len(valid_values) > 0 else 1.0

    # Custom hover text
    hover_text = []
    for i, row_label in enumerate(labels):
        row_texts = []
        for j, col_label in enumerate(labels):
            val = raw_values[i][j]
            if i == j:
                row_texts.append(f"{row_label}: {int(val)} papers total")
            elif val == EXCLUDED_PAIR_VALUE:
                dim = dimension_map.get(row_label, "?")
                row_texts.append(
                    f"{row_label} × {col_label}: EXCLUDED (same dimension: {dim})"
                )
            else:
                row_texts.append(f"{row_label} ∩ {col_label}: {int(val)} papers")
        hover_text.append(row_texts)

    # Text annotations and font colors are handled via layout annotations
    # because Plotly's Heatmap textfont.color does NOT support per-cell
    # 2D color arrays.  Instead, we disable texttemplate on the trace
    # and add one go.layout.Annotation per cell with individual colors.

    # Build annotations list for per-cell text with dynamic colors (Req #4)
    annotations = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = raw_values[i][j]
            if val == EXCLUDED_PAIR_VALUE:
                text = "×"
                font_color = COLORS["text_muted"]
            else:
                text = str(int(val))
                font_color = _get_dynamic_text_color(val, max_val)

            annotations.append(dict(
                x=labels[j],
                y=labels[i],
                text=text,
                font=dict(size=9, color=font_color),
                showarrow=False,
                xref="x",
                yref="y",
            ))

    fig = go.Figure(data=go.Heatmap(
        z=display_values,
        x=labels,
        y=labels,
        hovertext=hover_text,
        hoverinfo="text",
        colorscale=HEATMAP_COLORSCALE,
        showscale=True,
        colorbar=dict(title="Count"),
        # No texttemplate — we use layout annotations for per-cell colors
    ))

    fig.update_layout(
        height=max(600, len(labels) * 18),
        font=dict(family="Inter, sans-serif"),
        annotations=annotations,
        xaxis=dict(
            tickangle=45,
            side="bottom",
            tickfont=dict(size=9),
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=9),
        ),
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor=COLORS["surface"],
        paper_bgcolor=COLORS["white"],
    )

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Paper × Variant Matrix
# ═══════════════════════════════════════════════════════════════════════════

def _render_paper_variant_matrix(df: pd.DataFrame):
    """Render the Paper × Variant binary matrix."""
    st.markdown(
        sub_header("table_chart", "Paper × Variant Binary Matrix"),
        unsafe_allow_html=True,
    )
    st.caption(
        "1 = variant detected in paper, 0 = not detected. "
        "Paper rows use sequential IDs (P1, P2, ...)."
    )

    # ── Variant Filter (Filterable Matrix — Req #2) ──────────────────
    all_variants = list(df.columns)
    selected_variants = st.multiselect(
        "Filter variants (leave empty to show all)",
        options=all_variants,
        default=[],
        key="filter_pv_variants",
        help="Select specific variants to filter the matrix.",
    )

    display_df = df[selected_variants] if selected_variants else df

    # Stats per variant (column sums)
    variant_counts = display_df.sum(axis=0).sort_values(ascending=False)

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = go.Figure(data=go.Heatmap(
            z=display_df.values,
            x=list(display_df.columns),
            y=list(display_df.index),
            colorscale=[[0, COLORS["surface"]], [1, COLORS["secondary"]]],
            showscale=False,
            hovertemplate="Paper: %{y}<br>Variant: %{x}<br>Present: %{z}<extra></extra>",
        ))
        fig.update_layout(
            height=max(400, len(display_df) * 16),
            font=dict(family="Inter, sans-serif"),
            xaxis=dict(tickangle=45, tickfont=dict(size=8)),
            yaxis=dict(tickfont=dict(size=8), autorange="reversed"),
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor=COLORS["white"],
            paper_bgcolor=COLORS["white"],
        )
        st.plotly_chart(fig, use_container_width=True, key="paper_variant_heatmap")

    with col2:
        st.markdown("**Variant Detection Counts**")
        for variant_name, count in variant_counts.items():
            pct = count / len(df) * 100 if len(df) > 0 else 0
            st.progress(pct / 100, text=f"{variant_name}: {int(count)} ({pct:.0f}%)")

    # Paper ID mapping
    id_map = st.session_state.get("paper_id_map", {})
    if id_map:
        with st.expander("Paper ID Reference"):
            for pid, fname in sorted(id_map.items(), key=lambda x: int(x[0][1:])):
                st.markdown(f"**{pid}** → {fname}")

    # Expandable raw data table
    with st.expander("View Raw Data Table"):
        st.dataframe(display_df, use_container_width=True, height=400)


# ═══════════════════════════════════════════════════════════════════════════
# Research Gaps
# ═══════════════════════════════════════════════════════════════════════════

def _render_research_gaps(computer: MatrixComputer, dimension_map: dict):
    """Show variant pairs with zero intersection (research gaps)."""
    st.markdown(sub_header("psychology", "Research Gaps"), unsafe_allow_html=True)
    st.caption(
        "These **cross-dimension** variant pairs have no papers that discuss "
        "both variants. Same-dimension pairs are excluded by design."
    )

    gaps = computer.get_research_gaps()

    if not gaps:
        st.success("No research gaps found - all cross-dimension variant pairs are covered!")
        return

    st.metric("Total Research Gaps", len(gaps))

    # Filter by variant
    all_gap_variants = sorted(set(v for pair in gaps for v in pair))
    filter_variant = st.selectbox(
        "Filter by variant",
        ["(All)"] + all_gap_variants,
        key="gap_filter",
    )

    filtered_gaps = gaps
    if filter_variant != "(All)":
        filtered_gaps = [
            (a, b) for a, b in gaps
            if a == filter_variant or b == filter_variant
        ]

    st.write(f"Showing {len(filtered_gaps)} gap(s)")

    # Display as a dataframe with dimension info
    gap_rows = []
    for a, b in filtered_gaps:
        gap_rows.append({
            "Dimension A": dimension_map.get(a, "-"),
            "Variant A": a,
            "Dimension B": dimension_map.get(b, "-"),
            "Variant B": b,
        })
    gap_df = pd.DataFrame(gap_rows)
    gap_df.index = range(1, len(gap_df) + 1)
    gap_df.index.name = "#"
    st.dataframe(gap_df, use_container_width=True, height=400)

    # Download gaps
    csv = gap_df.to_csv()
    st.download_button(
        "Download Research Gaps CSV",
        data=csv,
        file_name="research_gaps.csv",
        mime="text/csv",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Manual Validation (Single + Pair)
# ═══════════════════════════════════════════════════════════════════════════
#
# Two sub-sections:
#   1. Single Variant Validation:
#      Override whether a variant is "present" or "absent" in a paper.
#
#   2. Pair Validation (NEW):
#      Confirm that a paper discusses a specific combination of two variants.
#      This forces both variants to present for that paper, so the
#      intersection count increases.
#
# Both save immediately, recompute the matrices live, and update the
# session state so all tabs reflect the change without a full re-run.

def _render_manual_validation(df: pd.DataFrame, computer: MatrixComputer):
    """Render manual validation: single variants + variant pairs."""
    st.markdown(sub_header("tune", "Manual Validation"), unsafe_allow_html=True)
    st.caption(
        "Override automatic detection results. Changes apply immediately - "
        "matrices, drill-downs, and CSV exports update in real time."
    )

    subtab_single, subtab_pair = st.tabs([
        "Single Variant Override",
        "Variant Pair Validation",
    ])

    with subtab_single:
        _render_single_override(df, computer)

    with subtab_pair:
        _render_pair_validation(df, computer)


def _render_single_override(df: pd.DataFrame, computer: MatrixComputer):
    """Allow manual override of individual variant detection results."""
    paper_ids = list(df.index)
    variant_names = list(df.columns)

    # Show filename reference for the selected paper
    id_map = st.session_state.get("paper_id_map", {})

    col1, col2 = st.columns(2)
    with col1:
        selected_paper = st.selectbox("Select Paper", paper_ids, key="val_paper")
        if selected_paper and selected_paper in id_map:
            st.caption(f"File: {id_map[selected_paper]}")
    with col2:
        selected_variant = st.selectbox("Select Variant", variant_names, key="val_variant")

    if selected_paper and selected_variant:
        current_value = bool(df.at[selected_paper, selected_variant])
        if current_value:
            st.markdown(
                f'Current detection: {icon("check_circle", size=16, color=COLORS["success"])} **Present**',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'Current detection: {icon("cancel", size=16, color=COLORS["error"])} **Not present**',
                unsafe_allow_html=True,
            )

        new_value = st.toggle(
            "Variant is present in this paper",
            value=current_value,
            key="val_toggle",
        )

        if new_value != current_value:
            if st.button("Save Override", type="primary", key="save_single_override"):
                computer.set_override(selected_paper, selected_variant, new_value)
                # Live recompute: update both matrices and re-export CSVs
                computer.apply_overrides_and_recompute()
                _sync_session_state(computer)
                st.success(
                    f"Override saved: {selected_paper} × {selected_variant} "
                    f"= {'Present' if new_value else 'Absent'}"
                )
                st.rerun()

    # Show existing overrides
    st.divider()
    overrides = computer.get_overrides()
    if overrides:
        st.markdown(sub_header("list", "Current Overrides"), unsafe_allow_html=True)
        override_rows = []
        for pid, vars_dict in overrides.items():
            for var_name, val in vars_dict.items():
                override_rows.append({
                    "Paper": pid,
                    "Variant": var_name,
                    "Override Value": "Present" if val else "Absent",
                })
        override_df = pd.DataFrame(override_rows)
        st.dataframe(override_df, use_container_width=True)

        if st.button("Clear All Single Overrides", key="clear_single_overrides"):
            from config.settings import MANUAL_OVERRIDES_FILE
            from utils.helpers import save_json
            save_json({}, MANUAL_OVERRIDES_FILE)
            computer._manual_overrides = {}
            computer.apply_overrides_and_recompute()
            _sync_session_state(computer)
            st.success("All single overrides cleared.")
            st.rerun()
    else:
        st.info("No single-variant overrides set.")


def _render_pair_validation(df: pd.DataFrame, computer: MatrixComputer):
    """
    Allow manual validation of variant pairs (combinations).

    How Pair Validation Works:
        1. User selects a Paper ID (P1, P2, ...) and two variants.
        2. Clicking "Save Pair Validation" stores the pair override
           and forces both variants to "present" for that paper.
        3. The paper-variant matrix is updated, the intersection matrix
           is recomputed, and CSVs are re-exported — all instantly.
        4. The intersection count for that variant pair increases.

    The pair override persists across sessions in pair_overrides.json.
    """
    paper_ids = list(df.index)
    variant_names = list(df.columns)
    id_map = st.session_state.get("paper_id_map", {})
    dimension_map = st.session_state.get("dimension_map", {})

    st.markdown("#### Validate a Variant Pair")
    st.caption(
        "Confirm that a paper discusses **both** variants. "
        "This will update the binary matrix and the intersection count."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        pair_paper = st.selectbox(
            "Select Paper",
            paper_ids,
            key="pair_val_paper",
        )
        if pair_paper and pair_paper in id_map:
            st.caption(f"File: {id_map[pair_paper]}")
    with col2:
        pair_va = st.selectbox(
            "Variant A",
            variant_names,
            key="pair_val_va",
        )
        if pair_va:
            dim_a = dimension_map.get(pair_va, "-")
            st.caption(f"Dimension: {dim_a}")
    with col3:
        pair_vb = st.selectbox(
            "Variant B",
            variant_names,
            key="pair_val_vb",
        )
        if pair_vb:
            dim_b = dimension_map.get(pair_vb, "-")
            st.caption(f"Dimension: {dim_b}")

    if pair_paper and pair_va and pair_vb:
        if pair_va == pair_vb:
            st.warning("Please select two different variants.")
        elif computer.is_same_dimension_pair(pair_va, pair_vb):
            st.warning(
                f"'{pair_va}' and '{pair_vb}' belong to the same dimension. "
                f"Same-dimension pairs are excluded from the VIM."
            )
        else:
            # Show current status
            va_present = bool(df.at[pair_paper, pair_va])
            vb_present = bool(df.at[pair_paper, pair_vb])
            both_present = va_present and vb_present

            status_a = "Present" if va_present else "Not detected"
            status_b = "Present" if vb_present else "Not detected"

            st.markdown(
                f"- **{pair_va}** in {pair_paper}: {status_a}\n"
                f"- **{pair_vb}** in {pair_paper}: {status_b}"
            )

            if both_present:
                st.info("This pair is already detected in this paper.")
            else:
                st.warning(
                    "One or both variants are not currently detected. "
                    "Saving will force both to 'Present'."
                )

            if st.button(
                f"Save Pair Validation: {pair_va} + {pair_vb} in {pair_paper}",
                type="primary",
                key="save_pair_override",
            ):
                computer.set_pair_override(pair_paper, pair_va, pair_vb)
                # Live recompute: update both matrices and re-export CSVs
                computer.apply_overrides_and_recompute()
                _sync_session_state(computer)
                st.success(
                    f"Pair validated: **{pair_paper}** contains both "
                    f"**{pair_va}** and **{pair_vb}**."
                )
                st.rerun()

    # ── Show existing pair overrides ─────────────────────────────────
    st.divider()
    pair_overrides = computer.get_pair_overrides()
    if pair_overrides:
        st.markdown(sub_header("link", "Current Pair Validations"), unsafe_allow_html=True)
        pair_rows = []
        for pid, pairs in pair_overrides.items():
            fname = id_map.get(pid, "")
            for pair in pairs:
                pair_rows.append({
                    "Paper": pid,
                    "Filename": fname,
                    "Variant A": pair[0],
                    "Variant B": pair[1],
                })
        pair_df = pd.DataFrame(pair_rows)
        st.dataframe(pair_df, use_container_width=True)

        if st.button("Clear All Pair Validations", key="clear_pair_overrides"):
            computer.clear_all_pair_overrides()
            computer.apply_overrides_and_recompute()
            _sync_session_state(computer)
            st.success("All pair validations cleared.")
            st.rerun()
    else:
        st.info("No pair validations set yet.")


def _sync_session_state(computer: MatrixComputer):
    """
    Sync the recomputed matrices back into session state.

    After `apply_overrides_and_recompute()`, the computer object has
    updated DataFrames.  We push them into session state so that
    all tabs (intersection, paper-variant, gaps, drill-down) see
    the latest data without a full re-run.
    """
    st.session_state.matrix_computer = computer
    st.session_state.paper_variant_df = computer.paper_variant_df
    st.session_state.intersection_df = computer.intersection_df


# ═══════════════════════════════════════════════════════════════════════════
# Download Results
# ═══════════════════════════════════════════════════════════════════════════

def _render_download_results():
    """Render download buttons for all CSV exports."""
    st.markdown(
        sub_header("download", "Download Results"),
        unsafe_allow_html=True,
    )
    st.caption(
        "Download analysis results as CSV files. "
        "All exports include any manual validations applied."
    )

    output_files = list(OUTPUT_DIR.glob("*.csv"))
    if not output_files:
        st.info("No output files found. Run the analysis first.")
        return

    cols = st.columns(min(len(output_files), 4))
    for i, file_path in enumerate(sorted(output_files)):
        with cols[i % len(cols)]:
            with open(file_path, "r") as f:
                csv_data = f.read()
            st.download_button(
                label=file_path.name,
                data=csv_data,
                file_name=file_path.name,
                mime="text/csv",
                use_container_width=True,
            )

    # Paper ID reference
    if "paper_id_map" in st.session_state:
        st.divider()
        with st.expander("Paper ID Reference (P-ID -> Filename)"):
            id_map = st.session_state.paper_id_map
            for pid, fname in sorted(id_map.items(), key=lambda x: int(x[0][1:])):
                st.markdown(f"**{pid}** -> {fname}")
