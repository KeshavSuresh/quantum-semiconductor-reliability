"""Error-versus-query-budget figure for QAE vs classical Monte Carlo.

Takes plain row dicts (as produced by ``qsr.quantum.error_scaling_study``),
so this module never imports Qiskit and works in a classical-only install.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import plotly.graph_objects as go

from qsr.visualization.theme import LIGHT, LINE_WIDTH, MARKER_SIZE, Theme, apply_layout

REQUIRED_KEYS = ("oracle_calls", "qae_rmse", "mc_rmse", "qae_crb_se", "mc_binomial_se")


def _check_rows(rows: Sequence[Mapping[str, float]]) -> list[Mapping[str, float]]:
    rs = list(rows)
    if not rs:
        raise ValueError("rows must not be empty")
    for r in rs:
        missing = [k for k in REQUIRED_KEYS if k not in r]
        if missing:
            raise ValueError(f"row is missing keys: {missing}")
    calls = [r["oracle_calls"] for r in rs]
    if any(b <= a for a, b in zip(calls, calls[1:])):
        raise ValueError("rows must be sorted by strictly increasing oracle_calls")
    values = np.array([[r[k] for k in REQUIRED_KEYS] for r in rs], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values[:, 0] <= 0) or np.any(values[:, 1:] < 0):
        raise ValueError("oracle_calls must be positive and errors non-negative and finite")
    return rs


def plot_error_vs_queries(rows: Sequence[Mapping[str, float]], *, theme: Theme = LIGHT) -> go.Figure:
    """Log-log plot of RMSE vs oracle calls: measured points plus theoretical scaling lines.

    The x axis counts oracle queries: applications of A or A^dagger summed over
    all shots (QAE), or samples (MC). It says nothing about wall-clock time.
    Dashed lines are theoretical predictions (QAE: asymptotic Cramer-Rao bound;
    MC: binomial standard error), not measurements.
    """
    if not isinstance(theme, Theme):
        raise TypeError("theme must be a qsr.visualization.Theme (use LIGHT or DARK)")
    rs = _check_rows(rows)
    x = np.array([r["oracle_calls"] for r in rs], dtype=float)

    fig = go.Figure()
    series = [
        ("QAE (MLAE)", "qae", "qae_crb_se", "theory: Cramér–Rao bound", theme.series[0]),
        ("Classical Monte Carlo", "mc", "mc_binomial_se", "theory: binomial SE", theme.series[1]),
    ]
    for name, key, theory_key, theory_label, color in series:
        y = np.array([r[f"{key}_rmse"] for r in rs], dtype=float)
        y_plot = np.where(y > 0, y, np.nan)  # log axis cannot show exact zeros
        fig.add_trace(go.Scatter(
            x=x, y=y_plot, mode="lines+markers", name=f"{name}: measured RMSE",
            line=dict(color=color, width=LINE_WIDTH),
            marker=dict(color=color, size=MARKER_SIZE, line=dict(color=theme.surface, width=2)),
            hovertemplate=f"{name}<br>oracle calls = %{{x:,}}<br>RMSE = %{{y:.3g}}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=x, y=[r[theory_key] for r in rs], mode="lines", name=f"{name}: {theory_label}",
            line=dict(color=color, width=LINE_WIDTH, dash="dash"),
            hovertemplate=f"{name} {theory_label}<br>oracle calls = %{{x:,}}<br>SE = %{{y:.3g}}<extra></extra>",
        ))
        fig.add_annotation(
            x=float(np.log10(x[-1])), y=float(np.log10(rs[-1][theory_key])), xref="x", yref="y",
            text="QAE" if key == "qae" else "MC", showarrow=False, xanchor="left", xshift=8,
            font=dict(color=theme.ink_secondary, size=12),
        )

    apply_layout(
        fig, theme,
        title="Estimation error vs oracle calls (query complexity, not runtime)",
        x_title="Oracle queries: A or A† applications (QAE) / samples (MC), log scale",
        y_title="RMSE over fixed seeds (log scale)",
    )
    # Decade ticks only; four legend entries go below the plot so they never collide with the title.
    fig.update_xaxes(type="log", dtick=1)
    fig.update_yaxes(type="log", dtick=1, exponentformat="power")
    fig.update_layout(
        hovermode="closest",
        margin=dict(r=60, t=70, b=140),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0.0),
    )
    return fig
