"""Plotly figure builders for the Quantum Semiconductor Reliability Explorer.

Every builder takes a :class:`qsr.models.uncertainty.DiscreteEtaModel` (the
single source of truth) and returns a ``plotly.graph_objects.Figure`` without
displaying or saving it, so the same figures can be written to HTML, shown in
a notebook, or passed to ``streamlit.plotly_chart`` in a later milestone.

Plotly is an optional dependency: install with ``pip install -e ".[viz]"``.
"""

from qsr.visualization.plots import (
    curve_data,
    default_time_grid,
    mc_results_table,
    plot_component_curves,
    plot_failure_and_reliability,
    plot_mc_vs_exact,
    run_mc_convergence,
)
from qsr.visualization.theme import DARK, LIGHT, Theme

__all__ = [
    "DARK",
    "LIGHT",
    "Theme",
    "curve_data",
    "default_time_grid",
    "mc_results_table",
    "plot_component_curves",
    "plot_failure_and_reliability",
    "plot_mc_vs_exact",
    "run_mc_convergence",
]
