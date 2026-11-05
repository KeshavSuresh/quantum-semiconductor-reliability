"""Interval plot: estimates with their uncertainty intervals against the exact reference.

Presentation only: it draws values it is given and computes nothing but the
axis range. The x axis is linear and always contains every interval and the
reference value, so no interval is cropped or exaggerated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import plotly.graph_objects as go

from qsr.visualization.theme import LIGHT, MARKER_SIZE, Theme, apply_layout

KINDS = ("classical", "quantum")


def _check(estimates: Sequence[Mapping]) -> list[Mapping]:
    rows = list(estimates)
    if not rows:
        raise ValueError("estimates must not be empty")
    for r in rows:
        for key in ("kind", "label", "estimate", "low", "high"):
            if key not in r:
                raise ValueError(f"estimate is missing {key!r}")
        if r["kind"] not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if not (np.isfinite([r["estimate"], r["low"], r["high"]]).all() and r["low"] <= r["estimate"] <= r["high"]):
            raise ValueError("each estimate needs finite values with low <= estimate <= high")
    return rows


def plot_estimates_vs_reference(
    exact: float,
    estimates: Sequence[Mapping],
    *,
    x_title: str = "Failure probability",
    title: str = "Estimates and intervals against the exact value",
    theme: Theme = LIGHT,
) -> go.Figure:
    """One row per estimate: point + asymmetric interval, with a dashed vertical line at ``exact``.

    ``estimates``: mappings with ``kind`` ("classical" or "quantum"), ``label``,
    ``estimate``, ``low``, ``high`` and optionally ``detail`` (shown on hover).
    """
    if not isinstance(theme, Theme):
        raise TypeError("theme must be a qsr.visualization.Theme (use LIGHT or DARK)")
    if not np.isfinite(exact):
        raise ValueError("exact must be finite")
    rows = _check(estimates)
    colour = {"classical": theme.classical, "quantum": theme.quantum}

    fig = go.Figure()
    for r in rows:
        err = abs(r["estimate"] - exact)
        fig.add_trace(go.Scatter(
            x=[r["estimate"]], y=[r["label"]], mode="markers", name=r["label"],
            marker=dict(color=colour[r["kind"]], size=MARKER_SIZE + 1, line=dict(color=theme.surface, width=2)),
            error_x=dict(type="data", symmetric=False, array=[r["high"] - r["estimate"]],
                         arrayminus=[r["estimate"] - r["low"]], color=colour[r["kind"]], thickness=2.5, width=7),
            hovertemplate=(f"{r['label']}<br>estimate %{{x:.6f}}<br>interval [{r['low']:.6f}, {r['high']:.6f}]"
                           f"<br>|error| {err:.6f}" + (f"<br>{r['detail']}" if r.get("detail") else "") + "<extra></extra>"),
            showlegend=False,
        ))
    lo = min([exact] + [r["low"] for r in rows])
    hi = max([exact] + [r["high"] for r in rows])
    pad = 0.12 * (hi - lo) if hi > lo else max(abs(exact) * 0.05, 1e-6)
    fig.add_vline(x=exact, line=dict(color=theme.reference, width=1.5, dash="dash"),
                  annotation=dict(text=f"exact p = {exact:.6f}", font=dict(color=theme.ink_secondary, size=12)),
                  annotation_position="top")
    apply_layout(fig, theme, title=title, x_title=x_title, y_title="")
    fig.update_xaxes(range=[lo - pad, hi + pad], tickformat=".4f")
    # Row labels are written above each interval (left-aligned, in the method's colour) rather
    # than as y tick labels, which would be clipped or squeeze the plot on a narrow screen.
    fig.update_yaxes(autorange="reversed", showgrid=False, ticks="", showticklabels=False, showline=False)
    for r in rows:
        fig.add_annotation(x=0, xref="paper", xanchor="left", y=r["label"], yref="y", yshift=17,
                           text=r["label"], showarrow=False, font=dict(color=colour[r["kind"]], size=13))
    fig.update_layout(height=130 + 80 * len(rows), margin=dict(l=20, r=30, t=70, b=55), hovermode="closest")
    return fig
