"""Session-state helpers, cached runners and shared result widgets."""

from __future__ import annotations

from typing import Any

import streamlit as st

from qsr.dashboard import labels, services
from qsr.dashboard.config import BuiltModel, ModelInputs
from qsr.visualization import DARK, LIGHT, Theme

INPUTS_KEY = "qsr_inputs"
BUILT_KEY = "qsr_built"
ERROR_KEY = "qsr_input_error"


def current_model() -> tuple[ModelInputs | None, BuiltModel | None]:
    """The sidebar's current inputs and validated model (None if invalid)."""
    return st.session_state.get(INPUTS_KEY), st.session_state.get(BUILT_KEY)


def require_model() -> tuple[ModelInputs, BuiltModel]:
    """Stop the page with a clear message if the sidebar model is invalid."""
    inputs, built = current_model()
    if built is None or inputs is None:
        reason = str(st.session_state.get(ERROR_KEY) or "check the inputs").rstrip(".")
        st.error(f"The model in the sidebar is not valid yet: {reason}. Fix it to see results.")
        st.stop()
    return inputs, built


def plot_theme() -> Theme:
    """Match figures to Streamlit's active light/dark theme (falls back to light)."""
    try:
        kind = st.context.theme.get("type")
    except Exception:  # older Streamlit or bare mode
        kind = None
    return DARK if kind == "dark" else LIGHT


def show_figure(fig: Any) -> None:
    # theme=None keeps the project's validated colours instead of Streamlit's restyling.
    st.plotly_chart(fig, theme=None, width="stretch")


def stale_banner(run: Any, current: services.Fingerprint) -> bool:
    """Show a warning if ``run`` was computed with different parameters. Returns True if stale."""
    if run is None or run.fingerprint == current:
        return False
    changed = ", ".join(services.changed_fields(run.fingerprint, current)) or "settings"
    st.warning(labels.STALE.format(changed=changed), icon="⚠️")
    return True


def interval_scope_notice(result: Any) -> None:
    """Calibration caption for settings the study tested; otherwise an 'unknown coverage' warning.

    Judged from the displayed run's own settings, so stale results are handled correctly.
    The measured coverage range is never shown for untested settings.
    """
    status = services.calibration_status(result.evaluation_schedule, result.shots)
    if status.covered:
        st.caption(labels.QAE_CI_CAPTION)
    else:
        st.warning(labels.UNCALIBRATED_WARNING.format(reasons=status.reason_text), icon="⚠️")


def scope_preview(schedule: tuple[int, ...], shots: int) -> None:
    """Before a run: flag settings outside the calibration study."""
    status = services.calibration_status(schedule, shots)
    if not status.covered:
        st.caption(labels.UNCALIBRATED_PREVIEW.format(reasons=status.reason_text))


def detail_line(*parts: str) -> None:
    """Full-width line for values that must never be truncated (intervals, counts)."""
    st.markdown(" · ".join(parts))


def model_summary(inputs: ModelInputs, built: BuiltModel) -> None:
    exact = services.exact_values(inputs)
    unit = inputs.time_unit
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(labels.EXACT, services.format_probability(exact.failure_probability), help=labels.EXACT_HELP)
        c2.metric("Exact reliability R = 1 − p", services.format_probability(exact.reliability))
        c3.metric("Mission time", f"{inputs.mission_time:g} {unit}")
        c4.metric("Components K / shape β", f"{built.model.n_components} / {inputs.beta:g}")


# --- cached runners: arguments are hashable; results are frozen dataclasses ---

@st.cache_data(show_spinner=False, max_entries=64)
def cached_classical(inputs: ModelInputs, n_samples: int) -> services.ClassicalRun:
    return services.run_classical(inputs, n_samples)


@st.cache_data(show_spinner=False, max_entries=64)
def cached_quantum(inputs: ModelInputs, schedule_length: int, shots: int) -> services.QuantumRun:
    return services.run_quantum(inputs, schedule_length, shots)


@st.cache_data(show_spinner=False, max_entries=64)
def cached_comparison(inputs: ModelInputs, schedule_length: int, shots: int) -> services.ComparisonRun:
    return services.run_comparison(inputs, schedule_length, shots)


# The scaling study is deliberately NOT cached: it draws a progress bar while it runs, and
# st.cache_data would try to replay those drawing calls into another session's page on a cache
# hit (CacheReplayClosureError). It only runs on an explicit button and is kept in session state.
