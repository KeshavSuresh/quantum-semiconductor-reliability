"""Reusable Plotly figures built from a :class:`DiscreteEtaModel`.

Design rules:

* **Single source of truth.** Every plotted probability comes from
  ``DiscreteEtaModel`` methods, ``qsr.models.tddb_weibull.weibull_cdf``, or
  ``qsr.simulation.monte_carlo``. This module adds no reliability mathematics.
* **Data and drawing are separate.** ``curve_data``, ``run_mc_convergence`` and
  ``mc_results_table`` return plain data (testable, and reusable as a table
  view). The ``plot_*`` functions only draw.
* **Builders return figures.** They never show or save, so the same figure can
  be written to HTML or passed to ``streamlit.plotly_chart`` later.
* **Units.** All times use one unit, given as ``time_unit`` for axis labels
  only. Values are never converted.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray

from qsr.models.tddb_weibull import weibull_cdf
from qsr.models.uncertainty import DiscreteEtaModel
from qsr.simulation.monte_carlo import MonteCarloResult, estimate_failure_probability
from qsr.validation import check_nonnegative_finite, check_positive_finite, check_positive_int
from qsr.visualization.theme import (
    LIGHT,
    LINE_WIDTH,
    MARKER_SIZE,
    MAX_CATEGORICAL,
    MIXTURE_LINE_WIDTH,
    Theme,
    apply_layout,
)

#: Direct end-of-line labels are drawn only when a figure has at most this many series.
MAX_DIRECT_LABELS = 4

#: Default plotting horizon as a multiple of the largest eta.
DEFAULT_HORIZON_FACTOR = 3.0


# --------------------------------------------------------------------------
# validation helpers
# --------------------------------------------------------------------------

def _check_model(model: DiscreteEtaModel) -> DiscreteEtaModel:
    if not isinstance(model, DiscreteEtaModel):
        raise TypeError("model must be a DiscreteEtaModel")
    return model


def _check_theme(theme: Theme) -> Theme:
    if not isinstance(theme, Theme):
        raise TypeError("theme must be a qsr.visualization.Theme (use LIGHT or DARK)")
    return theme


def _check_time_unit(time_unit: str) -> str:
    if not isinstance(time_unit, str) or not time_unit.strip():
        raise ValueError("time_unit must be a non-empty string, e.g. 'h'")
    return time_unit.strip()


def _check_times(times: ArrayLike) -> NDArray[np.float64]:
    arr = check_nonnegative_finite("times", times)
    if arr.ndim != 1:
        raise ValueError("times must be one-dimensional")
    if arr.size < 2:
        raise ValueError("times must contain at least two points")
    if np.any(np.diff(arr) <= 0):
        raise ValueError("times must be strictly increasing")
    return arr


def _check_optional_mission_time(mission_time: float | None) -> float | None:
    if mission_time is None:
        return None
    arr = check_nonnegative_finite("mission_time", mission_time)
    if arr.ndim != 0:
        raise ValueError("mission_time must be a scalar")
    return float(arr)


def _resolve_times(model: DiscreteEtaModel, times: ArrayLike | None) -> NDArray[np.float64]:
    return default_time_grid(model) if times is None else _check_times(times)


# --------------------------------------------------------------------------
# data functions (no plotting)
# --------------------------------------------------------------------------

def default_time_grid(
    model: DiscreteEtaModel, t_max: float | None = None, n_points: int = 400
) -> NDArray[np.float64]:
    """Evenly spaced times from 0 to ``t_max`` (default: 3 x the largest eta), same unit as the model."""
    model = _check_model(model)
    n = check_positive_int("n_points", n_points)
    if n < 2:
        raise ValueError("n_points must be >= 2")
    if t_max is None:
        upper = DEFAULT_HORIZON_FACTOR * float(np.max(model.etas))
    else:
        arr = check_positive_finite("t_max", t_max)
        if arr.ndim != 0:
            raise ValueError("t_max must be a scalar")
        upper = float(arr)
    return np.linspace(0.0, upper, n)


def curve_data(model: DiscreteEtaModel, times: ArrayLike | None = None) -> dict[str, NDArray[np.float64]]:
    """Curve values on a time grid.

    Returns a dict with:

    * ``time``: shape (N,)
    * ``mixture_failure``: ``model.failure_probability(time)``, shape (N,)
    * ``mixture_reliability``: ``model.reliability(time)``, shape (N,)
    * ``component_failure``: ``F(time | beta, eta_i)`` per component, shape (K, N)
    """
    model = _check_model(model)
    t = _resolve_times(model, times)
    component = np.asarray(weibull_cdf(t[np.newaxis, :], model.beta, model.etas[:, np.newaxis]), dtype=np.float64)
    return {
        "time": t,
        "mixture_failure": np.asarray(model.failure_probability(t), dtype=np.float64),
        "mixture_reliability": np.asarray(model.reliability(t), dtype=np.float64),
        "component_failure": component,
    }


def run_mc_convergence(
    model: DiscreteEtaModel,
    mission_time: float,
    sample_sizes: Sequence[int],
    seed: int,
    confidence: float = 0.95,
) -> list[MonteCarloResult]:
    """Run ``estimate_failure_probability`` once per sample size, each with the same ``seed``.

    Reusing the seed makes every point reproducible on its own. Sample sizes
    must be strictly increasing so the plotted x axis is ordered.
    """
    model = _check_model(model)
    sizes = list(sample_sizes)
    if not sizes:
        raise ValueError("sample_sizes must not be empty")
    sizes = [check_positive_int("sample_sizes element", n) for n in sizes]
    if any(b <= a for a, b in zip(sizes, sizes[1:])):
        raise ValueError("sample_sizes must be strictly increasing")
    return [estimate_failure_probability(model, mission_time, n, seed=seed, confidence=confidence) for n in sizes]


def mc_results_table(model: DiscreteEtaModel, results: Sequence[MonteCarloResult]) -> list[dict[str, float | int | bool]]:
    """Table rows comparing each Monte Carlo result with the exact mixture probability.

    This is the accessible table view of :func:`plot_mc_vs_exact`, and can be
    shown directly as ``st.dataframe`` later.
    """
    model = _check_model(model)
    mission_time, _ = _check_results(results)
    exact = model.failure_probability(mission_time)
    return [
        {
            "n_samples": r.n_samples,
            "n_failures": r.n_failures,
            "estimate": r.estimate,
            "std_error": r.std_error,
            "ci_low": r.ci_low,
            "ci_high": r.ci_high,
            "exact": exact,
            "abs_error": abs(r.estimate - exact),
            "exact_in_ci": bool(r.ci_low <= exact <= r.ci_high),
        }
        for r in results
    ]


def _check_results(results: Sequence[MonteCarloResult]) -> tuple[float, float]:
    rs = list(results)
    if not rs:
        raise ValueError("results must not be empty")
    if not all(isinstance(r, MonteCarloResult) for r in rs):
        raise TypeError("results must contain MonteCarloResult objects")
    mission_times = {r.mission_time for r in rs}
    confidences = {r.confidence for r in rs}
    if len(mission_times) != 1:
        raise ValueError("all results must share the same mission_time")
    if len(confidences) != 1:
        raise ValueError("all results must share the same confidence level")
    return mission_times.pop(), confidences.pop()


# --------------------------------------------------------------------------
# drawing helpers
# --------------------------------------------------------------------------

def _label_index(curves: NDArray[np.float64]) -> int:
    """Grid index where the curves are furthest apart (largest minimum vertical gap).

    Placing every direct label at this x keeps labels on their own line and
    away from each other, whatever the model parameters. The first 10 % of the
    grid is skipped, because all curves start together at F(0) = 0.
    """
    n = curves.shape[1]
    start = max(1, n // 10)
    ys = np.sort(curves[:, start:], axis=0)
    min_gap = np.min(np.diff(ys, axis=0), axis=0) if curves.shape[0] > 1 else np.ones(n - start)
    return start + int(np.argmax(min_gap))


def _add_direct_labels(
    fig: go.Figure, t: NDArray[np.float64], curves: NDArray[np.float64], texts: list[str], theme: Theme
) -> None:
    """Direct labels beside each line. Text uses ink colours, never the series colour."""
    idx = _label_index(curves)
    for text, y in zip(texts, curves[:, idx]):
        fig.add_annotation(
            x=float(t[idx]), y=float(y), xref="x", yref="y", text=text, showarrow=False,
            xanchor="left", yanchor="middle", xshift=6,
            font=dict(color=theme.ink_secondary, size=12), bgcolor=theme.surface, borderpad=1,
        )


def _add_mission_line(fig: go.Figure, mission_time: float | None, time_unit: str, theme: Theme) -> None:
    if mission_time is None:
        return
    fig.add_vline(
        x=mission_time, line=dict(color=theme.ink_muted, width=1, dash="dot"),
        annotation=dict(
            text=f"mission time {mission_time:g} {time_unit}", font=dict(color=theme.ink_muted, size=12),
        ),
        annotation_position="top right",
    )


# --------------------------------------------------------------------------
# figure builders
# --------------------------------------------------------------------------

def plot_failure_and_reliability(
    model: DiscreteEtaModel,
    times: ArrayLike | None = None,
    *,
    time_unit: str = "h",
    mission_time: float | None = None,
    theme: Theme = LIGHT,
) -> go.Figure:
    """Mixture failure probability F(t) and reliability R(t) = 1 - F(t) on one 0-1 axis."""
    model = _check_model(model)
    theme = _check_theme(theme)
    unit = _check_time_unit(time_unit)
    t_mission = _check_optional_mission_time(mission_time)
    data = curve_data(model, times)
    t = data["time"]

    fig = go.Figure()
    series = [
        ("Failure probability F(t)", data["mixture_failure"], theme.series[0]),
        ("Reliability R(t)", data["mixture_reliability"], theme.series[1]),
    ]
    for name, y, color in series:
        fig.add_trace(go.Scatter(
            x=t, y=y, mode="lines", name=name,
            line=dict(color=color, width=LINE_WIDTH, shape="spline"),
            hovertemplate=f"%{{y:.4f}}<extra>{name}</extra>",
        ))
    _add_direct_labels(fig, t, np.vstack([y for _, y, _ in series]), ["F(t)", "R(t)"], theme)

    _add_mission_line(fig, t_mission, unit, theme)
    apply_layout(
        fig, theme,
        title=f"Mixture failure probability and reliability (β = {model.beta:g}, K = {model.n_components})",
        x_title=f"Time ({unit})", y_title="Probability",
    )
    fig.update_yaxes(range=[0, 1.02], tickformat=".1f")
    fig.update_xaxes(range=[float(t[0]), float(t[-1])])
    fig.update_layout(hovermode="x unified")
    return fig


def plot_component_curves(
    model: DiscreteEtaModel,
    times: ArrayLike | None = None,
    *,
    time_unit: str = "h",
    mission_time: float | None = None,
    theme: Theme = LIGHT,
) -> go.Figure:
    """Per-component curves F(t | β, η_i) with the weighted mixture Σ w_i F(t | β, η_i).

    Components get categorical colours in fixed order. If K exceeds the eight
    available colours, all components are drawn in one muted colour as a
    single legend group, because more hues could not be told apart reliably.
    """
    model = _check_model(model)
    theme = _check_theme(theme)
    unit = _check_time_unit(time_unit)
    t_mission = _check_optional_mission_time(mission_time)
    data = curve_data(model, times)
    t = data["time"]
    k = model.n_components
    grouped = k > MAX_CATEGORICAL

    fig = go.Figure()
    label_texts: list[str] = []
    for i, (eta, w) in enumerate(zip(model.etas, model.weights)):
        name = f"η{_subscript(i + 1)} = {eta:g} {unit} (w = {w:.3g})"
        fig.add_trace(go.Scatter(
            x=t, y=data["component_failure"][i], mode="lines",
            name="Components" if grouped else name,
            legendgroup="components" if grouped else None,
            showlegend=(not grouped) or i == 0,
            line=dict(color=theme.ink_muted if grouped else theme.series[i], width=1 if grouped else LINE_WIDTH),
            opacity=0.6 if grouped else 1.0,
            hovertemplate=f"%{{y:.4f}}<extra>{name}</extra>",
        ))
        label_texts.append(f"η{_subscript(i + 1)}")

    mixture_name = "Mixture Σ wᵢ F(t | β, ηᵢ)"
    fig.add_trace(go.Scatter(
        x=t, y=data["mixture_failure"], mode="lines", name=mixture_name,
        line=dict(color=theme.mixture, width=MIXTURE_LINE_WIDTH),
        hovertemplate=f"%{{y:.4f}}<extra>{mixture_name}</extra>",
    ))
    label_texts.append("Mixture")

    if not grouped and len(label_texts) <= MAX_DIRECT_LABELS:
        curves = np.vstack([data["component_failure"], data["mixture_failure"][np.newaxis, :]])
        _add_direct_labels(fig, t, curves, label_texts, theme)

    _add_mission_line(fig, t_mission, unit, theme)
    apply_layout(
        fig, theme,
        title=f"Component and mixture failure probability (β = {model.beta:g})",
        x_title=f"Time ({unit})", y_title="Failure probability F(t)",
    )
    fig.update_yaxes(range=[0, 1.02], tickformat=".1f")
    fig.update_xaxes(range=[float(t[0]), float(t[-1])])
    fig.update_layout(hovermode="x unified")
    return fig


def plot_mc_vs_exact(
    model: DiscreteEtaModel,
    results: Sequence[MonteCarloResult],
    *,
    time_unit: str = "h",
    theme: Theme = LIGHT,
) -> go.Figure:
    """Monte Carlo estimates with Wilson intervals against the exact mixture probability.

    The x axis (sample size) is logarithmic. Error bars are asymmetric, as the
    Wilson interval is. The exact value is a horizontal reference line.
    """
    model = _check_model(model)
    theme = _check_theme(theme)
    unit = _check_time_unit(time_unit)
    mission_time, confidence = _check_results(results)
    rows = mc_results_table(model, results)
    exact = rows[0]["exact"]

    n = np.array([r["n_samples"] for r in rows], dtype=float)
    est = np.array([r["estimate"] for r in rows])
    lo = np.array([r["ci_low"] for r in rows])
    hi = np.array([r["ci_high"] for r in rows])
    custom = np.column_stack([lo, hi, [r["abs_error"] for r in rows], [r["n_failures"] for r in rows]])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[n.min() / 1.6, n.max() * 1.6], y=[exact, exact], mode="lines",
        name=f"Exact p = {exact:.6f}",
        line=dict(color=theme.mixture, width=LINE_WIDTH, dash="dash"),
        hoverinfo="skip",
    ))
    pct = f"{confidence * 100:g}%"
    fig.add_trace(go.Scatter(
        x=n, y=est, mode="markers", name=f"Monte Carlo p̂ ± {pct} Wilson CI",
        marker=dict(color=theme.series[0], size=MARKER_SIZE, line=dict(color=theme.surface, width=2)),
        error_y=dict(type="data", symmetric=False, array=hi - est, arrayminus=est - lo,
                     color=theme.series[0], thickness=2, width=6),
        customdata=custom,
        hovertemplate=(
            "n = %{x:,}<br>p̂ = %{y:.6f}<br>"
            f"{pct} CI = [%{{customdata[0]:.6f}}, %{{customdata[1]:.6f}}]<br>"
            "|p̂ − p| = %{customdata[2]:.6f}<br>failures = %{customdata[3]:,}<extra></extra>"
        ),
    ))
    apply_layout(
        fig, theme,
        title=f"Monte Carlo vs exact failure probability at t = {mission_time:g} {unit}",
        x_title="Number of Monte Carlo samples (log scale)", y_title="Failure probability",
    )
    fig.update_xaxes(type="log")
    fig.update_layout(hovermode="closest")
    return fig


_SUBSCRIPTS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def _subscript(i: int) -> str:
    return str(i).translate(_SUBSCRIPTS)
