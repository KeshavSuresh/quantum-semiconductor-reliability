"""Persistent sidebar: the TDDB model configuration shared by every page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from qsr.dashboard.config import (
    DEFAULT_INPUTS,
    LIMITS,
    DashboardInputError,
    ModelInputs,
    build_model,
    parse_component_rows,
)

from components.state import BUILT_KEY, ERROR_KEY, INPUTS_KEY

ETA_COL = "η"
RAW_COL = "Raw weight"
WIDGET_KEYS = ("sb_beta", "sb_unit", "sb_time", "sb_components", "sb_seed")
#: Shown under the editable table. After a page switch the grid is re-mounted and the
#: first click only focuses it (Streamlit data-editor behaviour), so typing may go elsewhere.
EDITOR_HINT = ("Tip: after switching pages, the first click may only focus the table. "
               "Click the cell again until it is highlighted, then edit.")


def _default_table() -> pd.DataFrame:
    return pd.DataFrame({ETA_COL: list(DEFAULT_INPUTS.etas), RAW_COL: list(DEFAULT_INPUTS.raw_weights)})


def _rows(df: pd.DataFrame) -> list[tuple[object, object]]:
    return [(row.get(ETA_COL), row.get(RAW_COL)) for _, row in df.iterrows()]


def render_sidebar() -> None:
    """Draw the sidebar and store the inputs and validated model in session state."""
    with st.sidebar:
        st.header("Model configuration")
        st.caption("Shared by every page. Results computed earlier are marked as out of date when you change anything.")

        beta = st.number_input(
            "Weibull shape β", min_value=LIMITS.min_beta, max_value=LIMITS.max_beta,
            value=DEFAULT_INPUTS.beta, step=0.1, key="sb_beta",
            help="How sharply failures cluster in time. β > 1: wear-out (failure rate rises with age).",
        )
        unit = st.text_input("Time unit (label only)", value=DEFAULT_INPUTS.time_unit, key="sb_unit",
                             help="Used for labels only. Values are never converted.")
        mission_time = st.number_input(
            f"Mission time t ({unit or '?'})", min_value=0.0, value=DEFAULT_INPUTS.mission_time, step=500.0,
            key="sb_time", help="The time by which we ask: what fraction of devices has failed?",
        )

        st.markdown("**Uncertain characteristic life η**")
        st.caption(f"Each row is one possible η with a weight (how likely it is). Up to {LIMITS.max_components} rows. "
                   "Raw weights can be any non-negative numbers; they are normalised to sum to 1.")
        table = st.data_editor(
            _default_table(), num_rows="dynamic", key="sb_components", hide_index=True, width="stretch",
            column_config={
                # No min_value: the grid would silently drop a typed minus sign (-1 becomes 1).
                # Out-of-range values are reported by build_model with a readable message instead.
                ETA_COL: st.column_config.NumberColumn(
                    ETA_COL, format="%g", width="small",
                    help=f"Characteristic life η, in {unit or 'the time unit'} (63 % have failed by t = η)"),
                RAW_COL: st.column_config.NumberColumn(
                    RAW_COL, format="%g", width="small",
                    help="Relative likelihood of this η. Any non-negative numbers; normalised to sum to 1."),
            },
        )
        st.caption(EDITOR_HINT)
        seed = int(st.number_input("Random seed", min_value=0, value=DEFAULT_INPUTS.seed, step=1, key="sb_seed",
                                   help="Makes every random run reproducible."))

        try:
            etas, raw = parse_component_rows(_rows(table))
            inputs = ModelInputs(beta=float(beta), etas=etas, raw_weights=raw, mission_time=float(mission_time),
                                 time_unit=unit.strip() or "", seed=seed)
            st.session_state[INPUTS_KEY] = inputs
            built = build_model(inputs)
        except DashboardInputError as exc:
            st.session_state[INPUTS_KEY] = None
            st.session_state[BUILT_KEY] = None
            st.session_state[ERROR_KEY] = str(exc)
            st.error(str(exc))
        else:
            st.session_state[BUILT_KEY] = built
            st.session_state[ERROR_KEY] = None
            f = built.model.component_failure_probabilities(inputs.mission_time)
            st.markdown("**Weights used by the model**")
            st.dataframe(
                pd.DataFrame({
                    "η": [f"{e:g}" for e in built.model.etas],
                    "Raw": [f"{w:g}" for w in built.raw_weights],
                    "Normalised": [f"{w:.4f}" for w in built.normalized_weights],
                    "F(t | β, η)": [f"{x:.4f}" for x in f],
                }),
                hide_index=True, width="stretch",
            )
            if built.was_normalized:
                st.info(f"Raw weights sum to {built.raw_weight_sum:g}, so each was divided by "
                        f"{built.raw_weight_sum:g}. The normalised column is what the model uses.", icon="ℹ️")

        if st.button("Reset to example model", width="stretch"):
            for k in WIDGET_KEYS:
                st.session_state.pop(k, None)
            st.rerun()
