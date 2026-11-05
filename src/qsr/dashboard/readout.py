"""Markdown readouts for results (pure functions: no Streamlit, no calculations).

A readout is a compact table, one row per quantity, with a badge that encodes
what the row is. The same colours are used in every figure:

* Reference: the exact model value (gray badge; ink in figures)
* Classical: Monte Carlo (orange)
* Quantum: MLAE on a simulator (blue)

Values are formatted from existing result objects; nothing is re-computed
except interval widths and absolute errors (simple differences of shown values).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from qsr.dashboard import labels
from qsr.dashboard.config import BuiltModel, ModelInputs
from qsr.dashboard.services import format_probability

KINDS = {
    "reference": labels.BADGE_REFERENCE,
    "classical": labels.BADGE_CLASSICAL,
    "quantum": labels.BADGE_QUANTUM,
}


@dataclass(frozen=True)
class ReadoutRow:
    kind: str  # "reference" | "classical" | "quantum"
    label: str
    estimate: float
    low: float | None = None
    high: float | None = None
    error: float | None = None
    budget: str | None = None


def _cell(value: float | None) -> str:
    return "—" if value is None else format_probability(value)


def readout_table(rows: Sequence[ReadoutRow], *, interval_label: str = "95 % interval") -> str:
    """Markdown table: badge + label, estimate, interval, width, |error| vs exact, budget."""
    header = f"| | Estimate | {interval_label} | Width | Error vs exact | Budget |\n|:--|--:|--:|--:|--:|--:|\n"
    lines = []
    for r in rows:
        if r.kind not in KINDS:
            raise ValueError(f"unknown readout kind {r.kind!r}")
        interval = "—" if r.low is None or r.high is None else f"[{format_probability(r.low)}, {format_probability(r.high)}]"
        width = "—" if r.low is None or r.high is None else format_probability(r.high - r.low)
        lines.append(f"| {KINDS[r.kind]} {r.label} | **{format_probability(r.estimate)}** | {interval} | {width} "
                     f"| {_cell(r.error)} | {r.budget or '—'} |")
    return header + "\n".join(lines)


def reference_row(p: float) -> ReadoutRow:
    return ReadoutRow("reference", labels.EXACT, p)


def classical_row(result, exact: float) -> ReadoutRow:
    return ReadoutRow("classical", labels.MC, result.estimate, result.ci_low, result.ci_high,
                      abs(result.estimate - exact), f"{result.n_samples:,} samples")


def quantum_row(result, exact: float) -> ReadoutRow:
    return ReadoutRow("quantum", labels.QAE, result.estimate, result.ci_low, result.ci_high,
                      abs(result.estimate - exact), f"{result.state_preparation_calls:,} oracle queries")


def _fmt_list(values: Sequence[float], fmt: str) -> str:
    return " / ".join(format(v, fmt) for v in values)


def model_line(inputs: ModelInputs, built: BuiltModel) -> str:
    """One compact line describing the model in use (used as page context, never truncated)."""
    unit = inputs.time_unit
    etas = _fmt_list(built.model.etas, ",.0f")
    weights = _fmt_list(built.normalized_weights, ".2f")
    return (f"Weibull β = {inputs.beta:g} · η = {etas} {unit} · w = {weights} · "
            f"mission time t = {inputs.mission_time:,.0f} {unit}")


def seconds(value: float, digits: int = 3) -> str:
    """A time in seconds with a non-breaking space, so the number and unit never wrap apart."""
    return f"{value:.{digits}f}\u00a0s"


def _md(text: str) -> str:
    """Escape the table delimiter so a value can never break the table layout."""
    return str(text).replace("|", "\\|")


def facts_table(rows: Sequence[tuple[str, ...]], header: Sequence[str] = ("Quantity", "Value")) -> str:
    """Markdown table for a few labelled facts (costs, times). First column left, the rest right-aligned."""
    if not rows:
        raise ValueError("facts_table needs at least one row")
    width = len(header)
    if any(len(r) != width for r in rows):
        raise ValueError(f"every row needs {width} cells to match the header")
    lines = ["| " + " | ".join(_md(h) for h in header) + " |",
             "|:--|" + "--:|" * (width - 1)]
    lines += ["| " + " | ".join(_md(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def reliability_table(p: float, r: float, mission_time: float, unit: str) -> str:
    """Two-row reference readout: failure probability and reliability at the mission time."""
    t = f"{mission_time:,.0f} {unit}"
    return ("| | Value |\n|:--|--:|\n"
            f"| {labels.BADGE_REFERENCE} Failure probability p(t = {t}) | **{format_probability(p)}** |\n"
            f"| {labels.BADGE_REFERENCE} Reliability R(t = {t}) = 1 − p | **{format_probability(r)}** |")
