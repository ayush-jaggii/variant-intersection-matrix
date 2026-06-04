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
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Optional, Tuple



from config.settings import HEATMAP_COLORSCALE, HEATMAP_ZERO_COLOR, OUTPUT_DIR
from core.matrix_computation import MatrixComputer, EXCLUDED_PAIR_VALUE, SAME_VARIANT_VALUE, UPPER_TRIANGLE_VALUE
import importlib
import core.conceptual_validation
importlib.reload(core.conceptual_validation)
from core.conceptual_validation import ConceptualValidator
from interface.design import section_header, sub_header, icon, COLORS


def _get_paper_meta(paper_id: str) -> dict:
    """Get paper metadata for a paper ID from session state."""
    meta_map = st.session_state.get("paper_meta_map", {})
    return meta_map.get(paper_id, {}) if isinstance(meta_map, dict) else {}


def _paper_option_label(paper_id: str) -> str:
    """Format paper selection labels as P-ID | title."""
    meta = _get_paper_meta(paper_id)
    title = meta.get("title", "").strip()
    if title:
        return f"{paper_id} | {title}"
    return paper_id


def render_matrix_viewer():
    """Render the interactive matrix viewer."""
    st.markdown(section_header("grid_view", "Matrix Viewer"), unsafe_allow_html=True)
    st.info("Explore the interactive matrices, find research gaps, and manually validate findings.")

    if not st.session_state.get("analysis_complete", False):
        st.info("Run the analysis first to view matrices.")
        return

    computer: MatrixComputer = st.session_state.matrix_computer
    paper_variant_df: pd.DataFrame = st.session_state.paper_variant_df
    intersection_df: pd.DataFrame = st.session_state.intersection_df
    dimension_map = st.session_state.get("dimension_map", {})

    if "conceptual_validator" not in st.session_state:
        st.session_state.conceptual_validator = ConceptualValidator(
            list(intersection_df.columns),
            dimension_map=dimension_map
        )
    validator = st.session_state.conceptual_validator
    validator.dimension_map = dimension_map

    # ── Tabs ─────────────────────────────────────────────────────────
    tab_intersection, tab_paper_variant, tab_gaps, tab_overrides, tab_validation, tab_download = st.tabs([
        "Intersection Matrix",
        "Paper x Variant Matrix",
        "Research Gaps",
        "Detection Overrides",
        "Manual Validation & Research Fertility",
        "Download Results",
    ])

    with tab_intersection:
        _render_intersection_matrix(intersection_df, computer, dimension_map)

    with tab_paper_variant:
        _render_paper_variant_matrix(paper_variant_df)

    with tab_gaps:
        _render_research_gaps(computer, dimension_map)

    with tab_overrides:
        _render_detection_overrides(paper_variant_df, computer)

    with tab_validation:
        _render_manual_validation(intersection_df, validator, computer, dimension_map)

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

    # ── Variant Filter (Filterable Matrix — Req #2 and Req #1) ──────────────────
    # Users select a subset of dimensions or variants; the heatmap updates in real time.
    all_variants = list(intersection_df.columns)
    detector = st.session_state.get("variant_detector")
    
    if detector:
        # Get unique dimension labels
        dimension_labels = []
        dim_to_vars = {}
        for key in all_variants:
            details = detector.get_variant_details(key)
            if details:
                d_label = details.get("dimension_label", details.get("dimension", "Uncategorized"))
                if d_label not in dimension_labels:
                    dimension_labels.append(d_label)
                if d_label not in dim_to_vars:
                    dim_to_vars[d_label] = []
                dim_to_vars[d_label].append(key)
                
        selected_dimensions = st.multiselect(
            "Filter by Dimensions (Optional)",
            options=dimension_labels,
            default=[],
            key="filter_intersection_dims",
            help="Select one or more dimensions to filter the matrix."
        )
        
        # Determine valid variants based on dimensions
        if selected_dimensions:
            filtered_variants = []
            for d in selected_dimensions:
                filtered_variants.extend(dim_to_vars.get(d, []))
        else:
            filtered_variants = all_variants
            
        selected_variants = st.multiselect(
            "Filter specific variants (Optional)",
            options=filtered_variants,
            default=[],
            key="filter_intersection_variants",
            help="Narrow down specific variants within the selected dimensions."
        )
        display_variants = selected_variants if selected_variants else filtered_variants
    else:
        selected_variants = st.multiselect(
            "Filter variants (leave empty to show all)",
            options=all_variants,
            default=[],
            key="filter_intersection_variants",
            help="Select specific variants to display a filtered sub-matrix.",
        )
        display_variants = selected_variants if selected_variants else all_variants

    # Apply filter: slice both rows and columns
    if display_variants and len(display_variants) < len(intersection_df.columns):
        display_df = intersection_df.loc[display_variants, display_variants]
    else:
        display_df = intersection_df

    # ── Heatmap Styling Config ──────────────────────────
    st.markdown("### Visualization Settings")
    colA, colB = st.columns([1, 2])
    with colA:
        color_scheme = st.selectbox(
            "Heatmap Color Scheme",
            options=["Blues", "Viridis", "Plasma", "Greys", "Cividis"],
            index=0,
            key="heatmap_color_scheme",
            help="Select the color palette for both the display and downloaded image."
        )

    # Create heatmap with dynamic text colors
    fig = _create_intersection_heatmap(display_df, dimension_map, color_scheme)
    st.plotly_chart(
        fig,
        use_container_width=False,
        key="intersection_heatmap",
        config={"responsive": True},
    )

    # Image download buttons
    st.markdown("Download Heatmap Visualization", unsafe_allow_html=True)
    
    matrix_hash = str(hash(display_df.values.tobytes())) + "_" + color_scheme
    state_key = f"heatmap_export_{matrix_hash}"

    if st.session_state.get(state_key) is None:
        if st.button("Prepare Output Images for Download", help="Generate high-resolution PNG and SVG files."):
            with st.spinner("Generating imagery..."):
                try:
                    png_bytes = _generate_static_heatmap(display_df, dimension_map, color_scheme, format="png")
                    svg_bytes = _generate_static_heatmap(display_df, dimension_map, color_scheme, format="svg")
                    st.session_state[state_key] = {"png": png_bytes, "svg": svg_bytes}
                    st.rerun()
                except Exception as e:
                    st.error(f"Image generation failed: {e}")
    else:
        export_data = st.session_state[state_key]
        col_png, col_svg, _ = st.columns([1, 1, 4])
        col_png.download_button("Download PNG", data=export_data["png"], file_name="vim_heatmap.png", mime="image/png")
        col_svg.download_button("Download SVG", data=export_data["svg"], file_name="vim_heatmap.svg", mime="image/svg+xml")

    # Variant Legend Table
    if detector:
        with st.expander("View Variant Legend", expanded=False):
            legend_data = []
            for key in all_variants:
                details = detector.get_variant_details(key)
                if details:
                    legend_data.append({
                        "Variant ID": details.get("variant_id", ""),
                        "Variant Name": details.get("name", ""),
                        "Dimension ID": details.get("dimension_id", ""),
                        "Dimension Name": details.get("dimension", "")
                    })
            if legend_data:
                st.dataframe(pd.DataFrame(legend_data), use_container_width=True)

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
            count = computer.get_pair_intersection_count(va, vb)
            if count > 0:
                st.success(
                    f"**{count}** paper(s) discuss both **{va}** and **{vb}**:"
                )
            else:
                st.warning(
                    f"**Research Gap:** No papers discuss both **{va}** and **{vb}**."
                )

        if papers:
            details_map = st.session_state.get("detection_details", {})
            for paper_id in papers:
                meta = _get_paper_meta(paper_id)
                title = meta.get("title", "")
                fname = meta.get("filename", "")
                display = f"- **{paper_id}** → **{title}**" if title else f"- **{paper_id}**"
                if fname:
                    display += f" <br>&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:gray; font-size:0.9em'>File: {fname}</span>"
                
                # Show detected alternate names if available
                p_details = details_map.get(paper_id, {})
                term_a = p_details.get(va)
                
                if va == vb:
                    if term_a:
                        display += f" <br>&nbsp;&nbsp;&nbsp;&nbsp;↳ <span style='color:gray; font-size:0.9em'>Found as: '{term_a}'</span>"
                else:
                    term_b = p_details.get(vb)
                    parts = []
                    if term_a: parts.append(f"**{va}** found as '{term_a}'")
                    if term_b: parts.append(f"**{vb}** found as '{term_b}'")
                    if parts:
                        display += f" <br>&nbsp;&nbsp;&nbsp;&nbsp;↳ <span style='color:gray; font-size:0.9em'>{' | '.join(parts)}</span>"

                st.markdown(display, unsafe_allow_html=True)


def _get_dynamic_text_color(value: float, max_value: float, color_scheme: str = "Blues") -> str:
    """
    Determine text color for a heatmap cell based on background intensity.
    """
    if value == 0 or max_value <= 0:
        return COLORS.get("text", "#212121")
        
    ratio = value / max_value
    
    # White background requires dark text, dark requires white
    if color_scheme == "Blues":
        return "#FFFFFF" if ratio > 0.40 else COLORS.get("text", "#212121")
    elif color_scheme == "Greys":
        return "#FFFFFF" if ratio > 0.50 else COLORS.get("text", "#212121")
    elif color_scheme in ["Viridis", "Plasma", "Cividis"]:
        return COLORS.get("text", "#212121") if ratio > 0.50 else "#FFFFFF"
        
    # Default fallback
    return COLORS.get("text", "#212121")


def _create_intersection_heatmap(
    df: pd.DataFrame,
    dimension_map: dict,
    color_scheme: str = "Blues",
) -> go.Figure:
    """
    Create a Plotly heatmap for the intersection matrix with dynamic text colors and selectable palettes.
    Zeroes are filtered out to render completely white and background-free constraints.
    """
    labels = list(df.columns)
    raw_values = df.values.copy().astype(float)

    display_values = raw_values.copy()
    display_values[display_values == EXCLUDED_PAIR_VALUE] = np.nan
    display_values[display_values == SAME_VARIANT_VALUE] = np.nan
    display_values[display_values == UPPER_TRIANGLE_VALUE] = np.nan

    valid_values = raw_values[~np.isin(raw_values, [EXCLUDED_PAIR_VALUE, SAME_VARIANT_VALUE, UPPER_TRIANGLE_VALUE])]
    max_val = float(np.nanmax(valid_values)) if len(valid_values) > 0 else 1.0

    import plotly.colors as pc
    base_colors = pc.get_colorscale(color_scheme)
    is_opportunity_matrix = len(valid_values) > 0 and np.all(valid_values == 0.0)

    if is_opportunity_matrix:
        dark_color = base_colors[-1][1]
        custom_colorscale = [[0.0, dark_color], [1.0, dark_color]]
    elif max_val > 0:
        custom_colorscale = [[0.0, "#FFFFFF"], [0.000001, base_colors[0][1]]] + [[v[0], v[1]] for v in base_colors[1:]]
    else:
        custom_colorscale = color_scheme

    hover_text = []
    for i, row_label in enumerate(labels):
        row_texts = []
        for j, col_label in enumerate(labels):
            val = raw_values[i][j]
            if val in [SAME_VARIANT_VALUE, UPPER_TRIANGLE_VALUE, EXCLUDED_PAIR_VALUE] or np.isnan(val):
                row_texts.append("")
            else:
                row_texts.append(f"{row_label} ∩ {col_label}: {int(val)} papers")
        hover_text.append(row_texts)

    text_values = []
    for i in range(len(labels)):
        row_text = []
        for j in range(len(labels)):
            val = raw_values[i][j]
            if val in [EXCLUDED_PAIR_VALUE, SAME_VARIANT_VALUE, UPPER_TRIANGLE_VALUE] or np.isnan(val):
                row_text.append("")
            else:
                row_text.append(str(int(val)))
        text_values.append(row_text)

    indices = list(range(len(labels)))



    fig = go.Figure(data=go.Heatmap(
        z=display_values,
        x=indices,
        y=indices,
        xgap=1,
        ygap=1,
        text=text_values,
        texttemplate="%{text}",
        textfont=dict(size=8, family="Inter, sans-serif"),
        hovertext=hover_text,
        hoverinfo="text",
        colorscale=custom_colorscale,
        showscale=True,
        colorbar=dict(
            title="Count",
            thickness=15,
            lenmode="fraction",
            len=0.85,
            yanchor="middle",
            y=0.5
        ),
        hoverongaps=False,
    ))

    height = max(450, min(950, len(labels) * 16 + 150))
    width = height + 120

    fig.update_layout(
        height=height,
        width=width,
        autosize=False,
        font=dict(family="Inter, sans-serif"),
        xaxis=dict(
            type="linear",
            range=[-0.5, len(labels) - 0.5],
            tickmode="array",
            tickvals=indices,
            ticktext=labels,
            tickangle=45,
            side="bottom",
            tickfont=dict(size=9),
            automargin=True,
        ),
        yaxis=dict(
            type="linear",
            range=[len(labels) - 0.5, -0.5],
            tickmode="array",
            tickvals=indices,
            ticktext=labels,
            tickfont=dict(size=9),
            automargin=True,
            scaleanchor="x",
            scaleratio=1,
        ),
        margin=dict(l=140, r=100, t=24, b=96),
        plot_bgcolor=COLORS["surface"],
        paper_bgcolor=COLORS["white"],
        uirevision="intersection-matrix",
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
            height=max(360, min(1200, len(display_df) * 18 + 120)),
            autosize=True,
            font=dict(family="Inter, sans-serif"),
            xaxis=dict(tickangle=45, tickfont=dict(size=8), automargin=True),
            yaxis=dict(tickfont=dict(size=8), autorange="reversed", automargin=True),
            margin=dict(l=50, r=10, t=10, b=96),
            plot_bgcolor=COLORS["white"],
            paper_bgcolor=COLORS["white"],
            uirevision="paper-variant-matrix",
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            key="paper_variant_heatmap",
            config={"responsive": True},
        )

    with col2:
        st.markdown("**Variant Detection Counts**")
        for variant_name, count in variant_counts.items():
            pct = count / len(df) * 100 if len(df) > 0 else 0
            st.progress(pct / 100, text=f"{variant_name}: {int(count)} ({pct:.0f}%)")

    # Paper ID mapping
    meta_map = st.session_state.get("paper_meta_map", {})
    if meta_map:
        with st.expander("Paper ID Reference"):
            for pid, meta in sorted(meta_map.items(), key=lambda x: int(x[0][1:])):
                st.markdown(f"**{pid}** → {meta.get('title', '')}")
                if meta.get("filename"):
                    st.caption(f"Filename: {meta['filename']}")

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

    if not filtered_gaps:
        st.success("No gaps match the current filter.")
        return

    gap_df = pd.DataFrame(filtered_gaps, columns=["Variant A", "Variant B"])
    gap_df["Dimension A"] = gap_df["Variant A"].map(lambda v: dimension_map.get(v, "Uncategorized"))
    gap_df["Dimension B"] = gap_df["Variant B"].map(lambda v: dimension_map.get(v, "Uncategorized"))
    st.dataframe(gap_df, use_container_width=True, height=400)

    st.download_button(
        "Download Research Gaps CSV",
        data=gap_df.to_csv(index=False),
        file_name="research_gaps.csv",
        mime="text/csv",
    )


def _render_manual_validation(
    intersection_df: pd.DataFrame,
    validator: ConceptualValidator,
    computer: MatrixComputer,
    dimension_map: dict,
):
    """Render the manual validation / fertility section."""
    if "ratings_version" not in st.session_state:
        st.session_state.ratings_version = 0

    st.markdown("### SECTION 2 — Rating Table / CSV Upload")



    # Display success message if it exists in session state from previous rerun
    if st.session_state.get("ratings_success_msg"):
        st.success(st.session_state.ratings_success_msg)
        del st.session_state.ratings_success_msg

    sub_up, sub_table = st.tabs(["CSV Upload", "Interactive Table"])

    pairs = validator.get_all_pairs()

    with sub_table:
        st.markdown("#### Rate Variant Combinations")
        st.caption("R = Relevant, N = Not relevant, ? = Uncertain")

        data = []
        for v1, v2 in pairs:
            row = {"Variant A": v1, "Variant B": v2}
            r_dict = validator.get_ratings(v1, v2)
            for a in validator.authors:
                row[a] = r_dict.get(a, "?")
            data.append(row)

        df_editor = pd.DataFrame(data)
        col_config = {a: st.column_config.SelectboxColumn(a, options=["R", "N", "?"], required=True) for a in validator.authors}

        edited_df = st.data_editor(
            df_editor,
            column_config=col_config,
            disabled=["Variant A", "Variant B"],
            hide_index=True,
            use_container_width=True,
            key=f"fertility_editor_multi_{st.session_state.ratings_version}"
        )

        c1, c2 = st.columns(2)
        if c1.button("Save Manual Ratings", type="primary"):
            validator.bulk_update(edited_df)
            st.session_state.ratings_version += 1
            st.session_state.ratings_success_msg = "Ratings saved!"
            st.rerun()

        csv_template = edited_df.to_csv(index=False)
        c2.download_button("Download Rating Template CSV", data=csv_template, file_name="rating_template.csv", mime="text/csv")

    with sub_up:
        st.markdown("#### Bulk Upload Ratings")
        st.info("Upload a CSV containing 'Variant A', 'Variant B' and author columns.")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                up_df = pd.read_csv(uploaded_file)
                if st.button("Process Uploaded CSV"):
                    validator.bulk_update(up_df)
                    st.session_state.ratings_version += 1
                    st.session_state.ratings_success_msg = "Ratings imported successfully!"
                    st.rerun()
            except Exception as e:
                st.error(f"Error parsing CSV: {e}")

    st.divider()

    st.markdown("### SECTION 3 — Inter-Rater Reliability")
    alpha = validator.compute_reliability()

    c1, _ = st.columns([1, 2])
    c1.metric(
        "Krippendorff Alpha",
        alpha,
        help="Measures the agreement between different raters (authors) considering chance. 1.0 = perfect agreement, 0.0 = chance agreement."
    )

    if alpha > 0.8:
        st.success("**Excellent agreement** (Alpha > 0.8)")
    elif alpha >= 0.7:
        st.info("**Acceptable agreement** (Alpha 0.7 - 0.8)")
    else:
        st.warning("**Low agreement** (Alpha < 0.7)")

    st.caption("Alpha measures the agreement between raters considering chance. 1.0 is perfect, 0.0 is chance.")

    st.divider()

    st.markdown("### SECTION 4 — Opportunity Matrix")
    st.caption("Showing pairs marked 'Relevant' (majority) but with 0 existing papers.")

    results = []
    opportunity_pairs = []

    for v1, v2 in pairs:
        final = validator.get_majority_rating(v1, v2)
        count = computer.get_pair_intersection_count(v1, v2)
        is_opp = (final == "R" and count == 0)
        results.append({
            "Variant A": v1,
            "Variant B": v2,
            "Final Rating": final,
            "Paper Count": count,
            "Is Opportunity": is_opp,
        })
        if is_opp:
            opportunity_pairs.append((v1, v2))

    res_df = pd.DataFrame(results)

    opp_matrix = intersection_df.copy()
    for i, r_label in enumerate(intersection_df.index):
        for j, c_label in enumerate(intersection_df.columns):
            if j >= i:
                continue

            final_rating = validator.get_majority_rating(r_label, c_label)
            count = computer.get_pair_intersection_count(r_label, c_label)
            if final_rating == "R" and count == 0:
                opp_matrix.iloc[i, j] = 0.0
            else:
                opp_matrix.iloc[i, j] = np.nan

    color_scheme = st.session_state.get("heatmap_color_scheme", "Blues")
    fig = _create_intersection_heatmap(opp_matrix, dimension_map, color_scheme)
    st.plotly_chart(
        fig,
        use_container_width=False,
        key="opportunity_heatmap",
        config={"responsive": True},
    )

    st.download_button(
        "Download Opportunity Matrix CSV",
        data=opp_matrix.to_csv(),
        file_name="opportunity_matrix.csv",
    )

    st.divider()

    st.markdown("### SECTION 5 — Research Fertility Ratio")
    total_possible = len(pairs)
    opp_count = len(opportunity_pairs)
    ratio = opp_count / total_possible if total_possible > 0 else 0

    f1, f2, f3 = st.columns(3)
    f1.metric(
        "Total Possible Pairs",
        total_possible,
        help="Total number of valid cross-dimension pairs (excludes same-dimension combinations of the same concept)."
    )
    f2.metric(
        "Research Opportunities",
        opp_count,
        help="Number of valid cross-dimension pairs with 0 papers that the authors rated as 'Relevant' (consensus)."
    )
    f3.metric(
        "Fertility Ratio",
        f"{ratio:.3f}",
        help="Ratio of Research Opportunities to Total Possible Pairs (Opportunities / Total Possible)."
    )

    if ratio > 0.5:
        st.success("Large unexplored research space (Ratio > 0.5)")
    elif ratio > 0.2:
        st.info("Moderate research opportunities (Ratio 0.2 - 0.5)")
    else:
        st.warning("Research area getting saturated (Ratio < 0.2)")

    st.markdown("#### Export Fertility Analysis")
    sum_data = {
        "Metric": ["Krippendorff Alpha", "Total Pairs", "Opportunity Count", "Fertility Ratio"],
        "Value": [alpha, total_possible, opp_count, ratio],
    }
    sum_csv = pd.DataFrame(sum_data).to_csv(index=False)

    col_dl1, col_dl2 = st.columns(2)
    col_dl1.download_button("Download Ratings Summary Table", data=res_df.to_csv(index=False), file_name="fertility_ratings_summary.csv")
    col_dl2.download_button("Download Fertility Analysis Summary (Stats)", data=sum_csv, file_name="fertility_analysis_summary.csv")


def _render_detection_overrides(df: pd.DataFrame, computer: MatrixComputer):
    """Render the detection override tab.

    This preserves the original tab entry point used by render_matrix_viewer()
    while delegating to the existing single-override and pair-validation
    sections.
    """
    st.markdown(sub_header("edit", "Detection Overrides"), unsafe_allow_html=True)
    st.caption("Adjust individual detections or confirm variant pairs.")
    _render_single_override(df, computer)
    st.divider()
    _render_pair_validation(df, computer)


def _render_single_override(df: pd.DataFrame, computer: MatrixComputer):
    """Allow manual override of individual variant detection results."""
    paper_ids = list(df.index)
    variant_names = list(df.columns)
    meta_map = st.session_state.get("paper_meta_map", {})

    col1, col2 = st.columns(2)
    with col1:
        selected_paper = st.selectbox(
            "Select Paper",
            paper_ids,
            key="val_paper",
            format_func=_paper_option_label,
            help="Select a paper to override its variant detection status."
        )
        if selected_paper and selected_paper in meta_map:
            st.caption(f"Filename: {meta_map[selected_paper].get('filename', '')}")
    with col2:
        selected_variant = st.selectbox(
            "Select Variant",
            variant_names,
            key="val_variant",
            help="Select a variant to override its presence/absence in the selected paper."
        )

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
            help="Toggle whether this variant is marked as present (1) or absent (0) in the selected paper. This corrects automated NLP detection errors."
        )

        if new_value != current_value:
            if st.button(
                "Save Override",
                type="primary",
                key="save_single_override",
                help="Persist this single-variant override to manual_overrides.json and recompute the matrices."
            ):
                computer.set_override(selected_paper, selected_variant, new_value)
                computer.apply_overrides_and_recompute()
                _sync_session_state(computer)
                st.success(f"Override saved: {selected_paper} × {selected_variant} = {'Present' if new_value else 'Absent'}")
                st.rerun()

    st.divider()
    overrides = computer.get_overrides()
    if overrides:
        st.markdown(sub_header("list", "Current Overrides"), unsafe_allow_html=True)
        override_rows = []
        for pid, vars_dict in overrides.items():
            for var_name, val in vars_dict.items():
                override_rows.append({
                    "Paper": pid,
                    "Title": _get_paper_meta(pid).get("title", ""),
                    "Variant": var_name,
                    "Override Value": "Present" if val else "Absent",
                })
        st.dataframe(pd.DataFrame(override_rows), use_container_width=True)

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
    """Allow manual validation of variant pairs (combinations)."""
    paper_ids = list(df.index)
    variant_names = list(df.columns)
    meta_map = st.session_state.get("paper_meta_map", {})
    dimension_map = st.session_state.get("dimension_map", {})

    st.markdown("#### Validate a Variant Pair")
    col1, col2, col3 = st.columns(3)
    with col1:
        pair_paper = st.selectbox(
            "Select Paper",
            paper_ids,
            key="pair_val_paper",
            format_func=_paper_option_label,
            help="Select a paper to validate a pair of variants within."
        )
        if pair_paper and pair_paper in meta_map:
            st.caption(f"Filename: {meta_map[pair_paper].get('filename', '')}")
    with col2:
        pair_va = st.selectbox(
            "Variant A",
            variant_names,
            key="pair_val_va",
            help="First variant of the pair you want to validate as co-occurring in the paper."
        )
        if pair_va: st.caption(f"Dimension: {dimension_map.get(pair_va, '-')}")
    with col3:
        pair_vb = st.selectbox(
            "Variant B",
            variant_names,
            key="pair_val_vb",
            help="Second variant of the pair you want to validate as co-occurring in the paper."
        )
        if pair_vb: st.caption(f"Dimension: {dimension_map.get(pair_vb, '-')}")

    if pair_paper and pair_va and pair_vb:
        if pair_va == pair_vb:
            st.warning("Please select two different variants.")
        elif computer.is_same_dimension_pair(pair_va, pair_vb):
            st.warning(f"Same-dimension pairs are excluded.")
        else:
            va_present = bool(df.at[pair_paper, pair_va])
            vb_present = bool(df.at[pair_paper, pair_vb])
            if st.button(
                "Save Pair Validation",
                type="primary",
                key="save_pair_override",
                help="Force both Variant A and Variant B to be Present (1) in this paper. This updates the intersection matrix and increases the pair count."
            ):
                computer.set_pair_override(pair_paper, pair_va, pair_vb)
                computer.apply_overrides_and_recompute()
                _sync_session_state(computer)
                st.success(f"Pair validated in {pair_paper}.")
                st.rerun()

    st.divider()
    pair_overrides = computer.get_pair_overrides()
    if pair_overrides:
        st.markdown(sub_header("link", "Current Pair Validations"), unsafe_allow_html=True)
        pair_rows = []
        for pid, pairs in pair_overrides.items():
            for pair in pairs:
                pair_rows.append({
                    "Paper": pid,
                    "Title": _get_paper_meta(pid).get("title", ""),
                    "Variant A": pair[0],
                    "Variant B": pair[1],
                })
        st.dataframe(pd.DataFrame(pair_rows), use_container_width=True)
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
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                csv_data = f.read()
            st.download_button(
                label=file_path.name,
                data=csv_data,
                file_name=file_path.name,
                mime="text/csv",
                use_container_width=True,
            )

    # Paper ID reference
    if "paper_meta_map" in st.session_state:
        st.divider()
        with st.expander("Paper ID Reference (P-ID -> Title)"):
            meta_map = st.session_state.paper_meta_map
            for pid, meta in sorted(meta_map.items(), key=lambda x: int(x[0][1:])):
                st.markdown(f"**{pid}** -> {meta.get('title', '')}")
                if meta.get("filename"):
                    st.caption(f"Filename: {meta['filename']}")
    elif "paper_id_map" in st.session_state:
        st.divider()
        with st.expander("Paper ID Reference (P-ID -> Filename)"):
            id_map = st.session_state.paper_id_map
            for pid, fname in sorted(id_map.items(), key=lambda x: int(x[0][1:])):
                st.markdown(f"**{pid}** -> {fname}")

@st.cache_data(show_spinner=False)
def _generate_static_heatmap(df: pd.DataFrame, dimension_map: dict, color_scheme: str = "Blues", format: str = "png") -> bytes:
    """
    Generate a static matplotlib heatmap matching the visual style of Plotly.
    Zeroes are filtered out to render completely white and background-free constraints.
    Uses standard seaborn palettes dynamically linked.
    """
    labels = list(df.columns)
    raw_values = df.values.copy().astype(float)

    display_values = raw_values.copy()
    display_values[display_values == EXCLUDED_PAIR_VALUE] = np.nan
    display_values[display_values == SAME_VARIANT_VALUE] = np.nan
    display_values[display_values == UPPER_TRIANGLE_VALUE] = np.nan

    valid_values = raw_values[~np.isin(raw_values, [EXCLUDED_PAIR_VALUE, SAME_VARIANT_VALUE, UPPER_TRIANGLE_VALUE])]
    max_val = float(np.nanmax(valid_values)) if len(valid_values) > 0 else 1.0

    import copy
    import matplotlib.pyplot as plt

    # Create the figure
    size = max(10, len(labels) * 0.4)
    fig, ax = plt.subplots(figsize=(size + 2, size), dpi=150)
    
    # Generate labels array for annotations
    annot = np.empty_like(raw_values, dtype='<U10')
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = raw_values[i][j]
            if val == EXCLUDED_PAIR_VALUE: annot[i, j] = ""
            elif val == SAME_VARIANT_VALUE: annot[i, j] = ""
            elif val == UPPER_TRIANGLE_VALUE: annot[i, j] = ""
            else: annot[i, j] = str(int(val))
    
    # We pass the NaNs to matplotlib to show them blank, but we need the background color
    ax.set_facecolor(COLORS["surface"] if "surface" in COLORS else "#FFFFFF")

    # Matplotlib expects specific casing for colormap names, while Plotly accepts Title Case.
    cmap_mapping = {
        "Blues": "Blues",
        "Viridis": "viridis",
        "Plasma": "plasma",
        "Greys": "Greys",
        "Cividis": "cividis"
    }
    mpl_cmap = cmap_mapping.get(color_scheme, color_scheme)
    
    base_cmap = copy.copy(plt.get_cmap(mpl_cmap))
    base_cmap.set_under('#FFFFFF')

    # Plot seaborn heatmap utilizing matplotlib's literal standard string palette maps.
    sns.heatmap(
        display_values, 
        cmap=base_cmap,
        annot=annot,
        fmt="",
        cbar_kws={'label': 'Count'},
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        mask=np.isnan(display_values),
        vmin=0.001, 
        vmax=max_val if max_val > 0 else 1,
        square=True,
        linewidths=1,
        linecolor='#E0E0E0'
    )
    
    # Adjust tick labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9, family='sans-serif')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9, family='sans-serif')
    
    # Adjust axes
    plt.tight_layout()
    
    # Save to buffer
    buf = BytesIO()
    fig.savefig(buf, format=format, bbox_inches='tight', facecolor='white', transparent=False)
    plt.close(fig)
    return buf.getvalue()
