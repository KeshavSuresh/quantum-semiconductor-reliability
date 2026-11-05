"""Quantum Semiconductor Reliability Explorer: Streamlit dashboard (Milestone 4).

Run from the repository root:

    streamlit run app/streamlit_app.py

Requires the ``app`` extra (Python >= 3.11); quantum pages also need the
``quantum`` extra. All calculations come from the ``qsr`` package.
"""

from __future__ import annotations

import pathlib
import sys

import streamlit as st

# Make ``app/components`` importable regardless of the working directory or Streamlit
# version (not every version adds the entry script's folder to sys.path).
_APP_DIR = str(pathlib.Path(__file__).resolve().parent)
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from components.model_sidebar import render_sidebar  # noqa: E402
from components.style import apply_style  # noqa: E402

st.set_page_config(page_title="QSR Explorer · TDDB reliability", layout="wide")
apply_style()

render_sidebar()

# Plain text navigation (no decorative icons); titles name the method, not the page type.
pages = [
    st.Page("views/overview.py", title="Overview", default=True),
    st.Page("views/classical.py", title="Classical Monte Carlo"),
    st.Page("views/quantum.py", title="Quantum amplitude estimation"),
    st.Page("views/comparison.py", title="Comparison"),
    st.Page("views/limitations.py", title="Limitations"),
]
st.navigation(pages).run()
