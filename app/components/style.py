"""Small, static stylesheet for the dashboard (no user input is ever inserted)."""

from __future__ import annotations

import streamlit as st

_CSS = """
<style>
/* Page width and rhythm: a readable measure instead of edge-to-edge content. */
.block-container, [data-testid="stMainBlockContainer"] { max-width: 1120px; padding-top: 2.2rem; }

/* Numbers align in columns (tabular figures) wherever results are shown. */
[data-testid="stMarkdownContainer"] table,
[data-testid="stMetricValue"], [data-testid="stCaptionContainer"] { font-variant-numeric: tabular-nums; }

/* Readout tables: thin rules, no zebra striping, compact rows. */
[data-testid="stMarkdownContainer"] table { border-collapse: collapse; width: 100%; font-size: 0.93rem; margin: 0.25rem 0 0.5rem; }
[data-testid="stMarkdownContainer"] thead th {
  font-weight: 600; color: #52514e; border: none; border-bottom: 1px solid #b9b8af;
  padding: 0.3rem 0.6rem; background: transparent; }
[data-testid="stMarkdownContainer"] tbody td { border: none; border-bottom: 1px solid #e6e5df; padding: 0.38rem 0.6rem; background: transparent; }
[data-testid="stMarkdownContainer"] tbody tr { background: transparent !important; }

/* Section labels: small caps-style overline used above groups of results. */
.qsr-overline { font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; color: #6b6a65; margin: 1.4rem 0 0.2rem; }
</style>
"""


def apply_style() -> None:
    """Inject the stylesheet once per rerun (st.html renders style-only content without layout space)."""
    st.html(_CSS)


def overline(text: str) -> None:
    """A small uppercase label that introduces a group of results (static text only)."""
    st.html(f'<div class="qsr-overline">{_escape(text)}</div>')


def _escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))
