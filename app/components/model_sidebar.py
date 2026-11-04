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
    """Draw the sidebar and store the inputs and validated model in session state.

    Order follows the physics: Weibull shape and mission time, then the uncertain
    characteristic life. Run settings (seed, time-unit label) are grouped last.
    """
    with st.sidebar:
        st.markdown("#### TDDB model")
        st.caption("Shared by every page. Changing anything marks earlier results as out of date.")

        beta = st.number_input(
            "Weibull shape β", min_value=LIMITS.min_beta, max_value=LIMITS.max_beta,
            value=DEFAULT_INPUTS.beta, step=0.1, format="%.2f", key="sb_beta",
            help="How sharply failures cluster in time. β > 1: wear-out (failure rate rises with age).",
        )
        unit_now = st.session_state.get("sb_unit", DEFAULT_INPUTS.time_unit) or "?"
        mission_time = st.number_input(
            f"Mission time t [{unit_now}]", min_value=0.0, value=DEFAULT_INPUTS.mission_time, step=500.0,
            format="%.0f", key="sb_time", help="The time by which we ask: what fraction of devices has failed?",
        )

        st.markdown("#### Characteristic life η (uncertain)")
        st.caption(f"One row per possible η with a raw weight (any non-negative numbers). Up to "
                   f"{LIMITS.max_components} rows.")
        table = st.data_editor(
            _default_table(), num_rows="dynamic", key="sb_components", hide_index=True, width="stretch",
            column_config={
                # No min_value: the grid would silently drop a typed minus sign (-1 becomes 1).
                # Out-of-range values are reported by build_model with a readable message instead.
                ETA_COL: st.column_config.NumberColumn(
                    f"η [{unit_now}]", format="%g", width="small",
                    help=f"Characteristic life η, in {unit_now} (63 % have failed by t = η)"),
                RAW_COL: st.column_config.NumberColumn(
                    RAW_COL, format="%g", width="small",
                    help="Relative likelihood of this η. Any non-negative numbers; normalised to sum to 1."),
            },
        )
        st.caption(EDITOR_HINT)

        with st.expander("Run settings", expanded=False):
            unit = st.text_input("Time unit (label only)", value=DEFAULT_INPUTS.time_unit, key="sb_unit",
                                 help="Used for labels only. Values are never converted.")
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
            st.markdown("#### Weights in use")
            st.dataframe(
                pd.DataFrame({
                    "η": [f"{e:,.0f}" for e in built.model.etas],
                    "Raw": [f"{w:g}" for w in built.raw_weights],
                    "Normalised": [f"{w:.4f}" for w in built.normalized_weights],
                    "F(t)": [f"{x:.4f}" for x in f],
                }),
                hide_index=True, width="stretch",
                # Fixed pixel widths so all four columns fit the sidebar without truncating values.
                column_config={
                    "η": st.column_config.TextColumn(width=58),
                    "Raw": st.column_config.TextColumn(width=38),
                    "Normalised": st.column_config.TextColumn(width=72),
                    "F(t)": st.column_config.TextColumn(width=56),
                },
            )
            if built.was_normalized:
                st.info(f"Raw weights sum to {built.raw_weight_sum:g}, so each was divided by "
                        f"{built.raw_weight_sum:g}. The normalised column is what the model uses.")

        if st.button("Reset to example model", width="stretch"):
            for k in WIDGET_KEYS:
                st.session_state.pop(k, None)
            st.rerun()
