"""
Design System
=============

Centralized design language for the Variant Intersection Matrix Analyzer.

Provides:
    • Material Symbols icon rendering via Google Fonts CDN
    • Consistent color palette tokens
    • CSS theme injection
    • Helper functions for icon-labelled headers, section titles, and badges

Usage:
    from interface.design import icon, section_header, inject_theme

    inject_theme()                          # Call once in app.py
    section_header("description", "Papers") # Renders header with icon
    icon("science")                         # Returns inline icon HTML
"""

from typing import Optional

# ─── Color Palette ───────────────────────────────────────────────────────────
# Design tokens used across the theme CSS and Plotly charts.

COLORS = {
    "primary":      "#1B2A4A",   # deep navy
    "primary_light":"#2C4470",   # lighter navy
    "secondary":    "#3B82B0",   # steel blue
    "accent":       "#E8913A",   # warm amber
    "success":      "#2D8A56",   # green
    "warning":      "#D4A843",   # gold
    "error":        "#C4483E",   # crimson
    "surface":      "#F7F8FA",   # light gray surface
    "surface_alt":  "#EEF1F6",   # slightly darker surface
    "border":       "#DDE2EA",   # border gray
    "text":         "#1A1A2E",   # near-black text
    "text_muted":   "#6B7280",   # muted gray text
    "white":        "#FFFFFF",
}

# ─── Icon helper ─────────────────────────────────────────────────────────────

def icon(name: str, size: int = 20, color: Optional[str] = None) -> str:
    """
    Return an inline HTML span rendering a Material Symbols Rounded icon.

    Args:
        name:  Material Symbols icon name (e.g. "description", "science").
               See https://fonts.google.com/icons for the full catalogue.
        size:  Icon size in pixels.
        color: Optional CSS color override.

    Returns:
        HTML string.
    """
    style_parts = [f"font-size:{size}px", "vertical-align:middle"]
    if color:
        style_parts.append(f"color:{color}")
    style = ";".join(style_parts)
    return f'<span class="material-symbols-rounded" style="{style}">{name}</span>'


def icon_label(icon_name: str, text: str, size: int = 20, gap: int = 8) -> str:
    """
    Return HTML for an icon followed by a label, properly aligned.

    Args:
        icon_name: Material Symbols icon name.
        text:      Label text.
        size:      Icon size in pixels.
        gap:       Gap between icon and text in pixels.

    Returns:
        HTML string.
    """
    return (
        f'<span style="display:inline-flex;align-items:center;gap:{gap}px">'
        f'{icon(icon_name, size)}'
        f'<span>{text}</span>'
        f'</span>'
    )


# ─── Section rendering helpers ───────────────────────────────────────────────

def section_header(icon_name: str, title: str, tag: str = "h2") -> str:
    """
    Build an HTML header string with an icon prefix.

    Use with st.markdown(..., unsafe_allow_html=True).

    Args:
        icon_name: Material Symbols icon name.
        title:     Header text.
        tag:       HTML heading tag (h1–h6).

    Returns:
        HTML string.
    """
    return (
        f'<{tag} class="vim-section-header">'
        f'<span class="vim-header-icon">{icon(icon_name, size=28, color=COLORS["secondary"])}</span>'
        f'{title}'
        f'</{tag}>'
    )


def sub_header(icon_name: str, title: str) -> str:
    """Build a sub-header with icon. Use with st.markdown(...)."""
    return (
        f'<h3 class="vim-sub-header">'
        f'<span class="vim-header-icon">{icon(icon_name, size=22, color=COLORS["secondary"])}</span>'
        f'{title}'
        f'</h3>'
    )


def status_badge(label: str, variant: str = "default") -> str:
    """
    Return HTML for a small status badge.

    Args:
        label:   Badge text.
        variant: One of 'success', 'warning', 'error', 'default'.

    Returns:
        HTML string.
    """
    bg_map = {
        "success": f"background:{COLORS['success']};color:#fff",
        "warning": f"background:{COLORS['warning']};color:#fff",
        "error":   f"background:{COLORS['error']};color:#fff",
        "default": f"background:{COLORS['surface_alt']};color:{COLORS['text']}",
    }
    style = bg_map.get(variant, bg_map["default"])
    return (
        f'<span class="vim-badge" style="{style}">{label}</span>'
    )


# ─── Theme injection ────────────────────────────────────────────────────────

def get_theme_css() -> str:
    """
    Return the full CSS theme string.

    Loads Material Symbols Rounded from Google Fonts and defines all
    custom styles for the application.
    """
    return f"""
    <style>
        /* ── Load Google Fonts via @import (Streamlit strips <link> tags) ── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=block');

        /* ── Base typography ──────────────────────────────────────── */
        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        /* ── Main container ──────────────────────────────────────── */
        .block-container {{
            max-width: 1200px;
            padding-top: 1.2rem;
        }}

        /* ── Section headers ─────────────────────────────────────── */
        .vim-section-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            color: {COLORS['primary']};
            font-weight: 700;
            font-size: 1.6rem;
            border-bottom: 2px solid {COLORS['border']};
            padding-bottom: 10px;
            margin-bottom: 1rem;
        }}

        .vim-sub-header {{
            display: flex;
            align-items: center;
            gap: 8px;
            color: {COLORS['primary_light']};
            font-weight: 600;
            font-size: 1.15rem;
            margin-bottom: 0.6rem;
        }}

        .vim-header-icon {{
            display: inline-flex;
            align-items: center;
            flex-shrink: 0;
        }}

        /* ── Streamlit heading overrides ──────────────────────────── */
        h1 {{
            color: {COLORS['primary']};
            font-weight: 700;
        }}
        h2 {{
            color: {COLORS['primary']};
            font-weight: 600;
            border-bottom: 2px solid {COLORS['border']};
            padding-bottom: 8px;
        }}
        h3 {{
            color: {COLORS['primary_light']};
            font-weight: 600;
        }}

        /* ── Metric cards ────────────────────────────────────────── */
        [data-testid="stMetric"] {{
            background: {COLORS['white']};
            border-radius: 10px;
            padding: 14px 18px;
            border: 1px solid {COLORS['border']};
            box-shadow: 0 1px 3px rgba(27,42,74,0.06);
        }}
        [data-testid="stMetricLabel"] {{
            color: {COLORS['text_muted']};
            font-weight: 500;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        [data-testid="stMetricValue"] {{
            color: {COLORS['primary']};
            font-weight: 700;
        }}

        /* ── Sidebar ─────────────────────────────────────────────── */
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {COLORS['primary']} 0%, #243756 100%);
        }}
        [data-testid="stSidebar"] * {{
            color: #CBD5E1 !important;
        }}
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {{
            color: #FFFFFF !important;
            border-bottom-color: rgba(255,255,255,0.12) !important;
        }}
        [data-testid="stSidebar"] .stRadio label {{
            color: #E2E8F0 !important;
            font-weight: 500;
        }}
        [data-testid="stSidebar"] .stRadio label:hover {{
            color: #FFFFFF !important;
        }}
        [data-testid="stSidebar"] hr {{
            border-color: rgba(255,255,255,0.1) !important;
        }}
        [data-testid="stSidebar"] .stCaption,
        [data-testid="stSidebar"] small {{
            color: #94A3B8 !important;
        }}

        /* ── Buttons ─────────────────────────────────────────────── */
        .stButton > button[kind="primary"] {{
            background: {COLORS['primary']};
            border: none;
            border-radius: 8px;
            font-weight: 600;
            letter-spacing: 0.02em;
            transition: background 0.2s, box-shadow 0.2s;
        }}
        .stButton > button[kind="primary"]:hover {{
            background: {COLORS['primary_light']};
            box-shadow: 0 2px 8px rgba(27,42,74,0.18);
        }}
        .stButton > button[kind="secondary"] {{
            border-radius: 8px;
            border: 1px solid {COLORS['border']};
            font-weight: 500;
            transition: border-color 0.2s, box-shadow 0.2s;
        }}
        .stButton > button[kind="secondary"]:hover {{
            border-color: {COLORS['secondary']};
            box-shadow: 0 1px 4px rgba(59,130,176,0.12);
        }}

        /* ── Tabs ────────────────────────────────────────────────── */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px;
            border-bottom: 2px solid {COLORS['border']};
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 6px 6px 0 0;
            font-weight: 500;
            padding: 8px 16px;
        }}
        .stTabs [aria-selected="true"] {{
            border-bottom: 2px solid {COLORS['secondary']} !important;
            color: {COLORS['secondary']} !important;
            font-weight: 600;
        }}

        /* ── Divider ─────────────────────────────────────────────── */
        hr {{
            border-color: {COLORS['border']};
        }}

        /* ── Badge ───────────────────────────────────────────────── */
        .vim-badge {{
            display: inline-block;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }}

        /* ── Status row ──────────────────────────────────────────── */
        .vim-status-row {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 4px 0;
            font-size: 0.92rem;
        }}
        .vim-status-label {{
            color: #94A3B8;
            font-weight: 500;
        }}
        .vim-status-value {{
            color: #E2E8F0;
            font-weight: 700;
        }}

        /* ── File uploader ───────────────────────────────────────── */
        [data-testid="stFileUploader"] {{
            border-radius: 10px;
        }}

        /* ── Expander ────────────────────────────────────────────── */
        .streamlit-expanderHeader {{
            font-weight: 600;
            color: {COLORS['primary_light']};
        }}
    </style>
    """
