"""Classical Monte Carlo: the exact reference, then a sampling experiment that estimates it."""

import pandas as pd
import streamlit as st

from qsr.dashboard import labels, readout, services
from qsr.dashboard.config import MC_SAMPLE_CHOICES, DashboardInputError, build_model
from qsr.visualization import mc_results_table, plot_component_curves, plot_failure_and_reliability, plot_mc_vs_exact

from components.state import (
    cached_classical,
    model_summary,
    plot_theme,
    require_model,
    show_figure,
    stale_banner,
)
from components.style import overline

st.title("Classical Monte Carlo")
inputs, built = require_model()
model, t, unit, theme = built.model, inputs.mission_time, inputs.time_unit, plot_theme()
model_summary(inputs, built)

overline("1 · Reference: the exact model")
st.caption("Computed directly from the formula, with no sampling. The dotted vertical line marks the mission time.")
tab1, tab2 = st.tabs(["Failure probability and reliability", "Each η and the weighted mixture"])
with tab1:
    show_figure(plot_failure_and_reliability(model, time_unit=unit, mission_time=t, theme=theme))
with tab2:
    show_figure(plot_component_curves(model, time_unit=unit, mission_time=t, theme=theme))

overline("2 · Experiment: estimate p by sampling")
st.markdown(
    "Each sample picks an η according to its weight, draws a failure time from that Weibull "
    "distribution, and counts a failure if it falls before the mission time. The estimate is "
    "failures ÷ samples; its error shrinks roughly as 1/√N."
)
left, right = st.columns([2, 1], vertical_alignment="bottom")
with left:
    n = st.select_slider("Number of samples N", options=MC_SAMPLE_CHOICES, value=100_000, key="cl_n",
                         format_func=lambda v: f"{v:,}")
with right:
    pressed = st.button("Run Monte Carlo", type="primary", width="stretch")
current = services.fingerprint(inputs, n_samples=n)
if pressed:
    try:
        with st.spinner("Sampling…"):
            st.session_state["classical_run"] = cached_classical(inputs, n)
    except DashboardInputError as exc:
        st.error(str(exc))

run = st.session_state.get("classical_run")
overline("3 · Result")
if run is None:
    st.caption("No result yet. Press **Run Monte Carlo** to estimate p by sampling.")
else:
    stale_banner(run, current)
    r = run.result
    st.markdown(readout.readout_table([readout.reference_row(run.exact), readout.classical_row(r, run.exact)],
                                      interval_label="95 % Wilson interval"))
    st.caption(f"{r.n_failures:,} failures in {r.n_samples:,} samples · seed {r.seed} · the Wilson interval is a "
               "standard binomial interval that stays sensible near 0 and 1.")

    overline("4 · Convergence")
    st.caption("The same seed at increasing sample sizes. Error bars are 95 % Wilson intervals; the line is the "
               "exact value.")
    # Always draw a run against the model that produced it (matters when the run is stale).
    run_inputs = services.inputs_from_fingerprint(run.fingerprint)
    run_model = build_model(run_inputs).model
    show_figure(plot_mc_vs_exact(run_model, run.convergence, time_unit=run_inputs.time_unit, theme=theme))
    with st.expander("Convergence data table"):
        st.dataframe(pd.DataFrame(mc_results_table(run_model, run.convergence)), hide_index=True, width="stretch")
st.caption(labels.UNMATCHED_NOTE)
