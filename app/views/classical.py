"""Classical analysis: exact probability, curves and Monte Carlo."""

import pandas as pd
import streamlit as st

from qsr.dashboard import labels, services
from qsr.dashboard.config import MC_SAMPLE_CHOICES, DashboardInputError, build_model
from qsr.visualization import mc_results_table, plot_component_curves, plot_failure_and_reliability, plot_mc_vs_exact

from components.state import (
    cached_classical,
    detail_line,
    model_summary,
    plot_theme,
    require_model,
    show_figure,
    stale_banner,
)

st.title("Classical analysis")
inputs, built = require_model()
model, t, unit, theme = built.model, inputs.mission_time, inputs.time_unit, plot_theme()
model_summary(inputs, built)

st.subheader("Exact curves")
st.caption("Computed directly from the model. The dotted line marks the mission time.")
tab1, tab2 = st.tabs(["Failure probability and reliability", "Each η and the weighted mixture"])
with tab1:
    show_figure(plot_failure_and_reliability(model, time_unit=unit, mission_time=t, theme=theme))
with tab2:
    show_figure(plot_component_curves(model, time_unit=unit, mission_time=t, theme=theme))

st.subheader(labels.MC)
st.markdown(f"{labels.MC_HELP} Each sample picks an η according to its weight, draws a failure time, and checks "
            "whether it is before the mission time.")
n = st.select_slider("Number of samples", options=MC_SAMPLE_CHOICES, value=100_000, key="cl_n",
                     format_func=lambda v: f"{v:,}")
current = services.fingerprint(inputs, n_samples=n)
if st.button("Run Monte Carlo", type="primary"):
    try:
        with st.spinner("Sampling…"):
            st.session_state["classical_run"] = cached_classical(inputs, n)
    except DashboardInputError as exc:
        st.error(str(exc))

run = st.session_state.get("classical_run")
if run is None:
    st.info("Press **Run Monte Carlo** to estimate p by sampling.")
else:
    stale_banner(run, current)
    r = run.result
    c1, c2, c3 = st.columns(3)
    c1.metric(labels.MC, services.format_probability(r.estimate), help=labels.MC_HELP)
    c2.metric(f"{labels.EXACT} (for this run)", services.format_probability(run.exact), help=labels.EXACT_HELP)
    c3.metric("|estimate − exact|", services.format_probability(abs(r.estimate - run.exact)))
    detail_line(f"**{labels.MC_CI}:** {services.format_interval(r.ci_low, r.ci_high)}",
                f"**Failures:** {r.n_failures:,} of {r.n_samples:,} samples")
    st.caption(f"Seed {r.seed} · Wilson interval: a standard binomial interval that stays sensible near 0 and 1.")
    st.markdown("**Convergence**: the same seed at increasing sample sizes; error bars are 95 % Wilson intervals.")
    # Always draw a run against the model that produced it (matters when the run is stale).
    run_inputs = services.inputs_from_fingerprint(run.fingerprint)
    run_model = build_model(run_inputs).model
    show_figure(plot_mc_vs_exact(run_model, run.convergence, time_unit=run_inputs.time_unit, theme=theme))
    with st.expander("Convergence table"):
        st.dataframe(pd.DataFrame(mc_results_table(run_model, run.convergence)), hide_index=True, width="stretch")
st.caption(labels.UNMATCHED_NOTE)
