"""Comparison: budget-matched QAE vs Monte Carlo, and the error-versus-oracle-calls study."""

import pandas as pd
import streamlit as st

from qsr.dashboard import labels, services
from qsr.dashboard.config import LIMITS, SHOT_CHOICES, DashboardInputError

from components.state import (
    cached_comparison,
    interval_scope_notice,
    model_summary,
    plot_theme,
    require_model,
    scope_preview,
    show_figure,
    stale_banner,
)

st.title("Comparison")
inputs, built = require_model()
model_summary(inputs, built)

if not services.quantum_available():
    st.warning(labels.QISKIT_MISSING, icon="⚠️")
    st.markdown("The matched comparison needs the quantum estimator. Classical results are on the **Classical** page.")
    st.stop()

from qsr.visualization import plot_error_vs_queries  # noqa: E402

# --------------------------------------------------------------------------
st.subheader("1. Matched comparison")
st.markdown(labels.MATCHED_BUDGET)
c1, c2 = st.columns(2)
length = c1.slider("Grover schedule length", 1, LIMITS.max_schedule_length, 5, key="cmp_len")
shots = c2.select_slider("Shots per circuit", options=SHOT_CHOICES, value=100, key="cmp_shots")
sch = services.schedule_for(length)
st.caption(f"Budget: {shots} × {sum(2 * m + 1 for m in sch)} = {services.oracle_calls_for(length, shots):,} "
           "oracle queries for QAE = the same number of Monte Carlo samples.")
scope_preview(sch, shots)

current = services.fingerprint(inputs, schedule_length=length, shots=shots)
if st.button("Run matched comparison", type="primary"):
    try:
        with st.spinner("Running both methods with the same budget…"):
            st.session_state["comparison_run"] = cached_comparison(inputs, length, shots)
    except DashboardInputError as exc:
        st.error(str(exc))

run = st.session_state.get("comparison_run")
if run is None:
    st.info("Press **Run matched comparison**.")
else:
    stale_banner(run, current)
    comp = run.comparison
    st.markdown("**Accuracy at an equal number of model evaluations**")
    df = pd.DataFrame(services.comparison_rows(comp))
    st.dataframe(df, hide_index=True, width="stretch", column_config={
        "Value": st.column_config.NumberColumn(format="%.6f"),
        "Interval low": st.column_config.NumberColumn(format="%.6f"),
        "Interval high": st.column_config.NumberColumn(format="%.6f"),
        "|error| vs exact": st.column_config.NumberColumn(format="%.6f"),
        "Model evaluations": st.column_config.NumberColumn(format="%d"),
    })
    st.caption(f"Monte Carlo interval: {labels.MC_CI}. QAE interval: {labels.QAE_CI}.")
    interval_scope_notice(comp.qae)
    if comp.qae.n_ci_regions > 1:
        st.warning(labels.MULTI_REGION_WARNING.format(n=comp.qae.n_ci_regions), icon="⚠️")
    st.caption(labels.QUERY_VS_RUNTIME)

    st.markdown(f"**{labels.SIMULATOR_TIME}** (measured separately; not part of the accuracy comparison)")
    timing = services.qae_timing(comp.qae)
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Classical Monte Carlo", f"{comp.mc_time_s:.4f} s")
    r2.metric(labels.QAE_EXECUTION_TIME, f"{timing.execution_s:.4f} s")
    ratio = timing.execution_s / comp.mc_time_s if comp.mc_time_s > 0 else float("inf")
    r3.metric("QAE execution ÷ MC time", f"{ratio:,.0f}×")
    r4.metric(labels.QAE_COMPILE_TIME, f"{timing.diagnostics_compile_s:.4f} s")
    st.caption(labels.COMPILE_NOTE)
    st.caption(labels.RUNTIME_CAPTION)

# --------------------------------------------------------------------------
st.subheader("2. Error versus oracle queries")
st.markdown("Repeats both methods over several seeds and budgets, and plots the root-mean-square error against "
            "the number of oracle queries (QAE) or samples (MC). This is a **query-complexity** comparison.")
c1, c2, c3 = st.columns(3)
max_len = c1.slider("Largest schedule length", 2, LIMITS.max_schedule_length, 5, key="sc_len")
n_seeds = c2.slider("Seeds per point", 3, LIMITS.max_scaling_seeds, 10, key="sc_seeds")
sc_shots = c3.select_slider("Shots per circuit", options=SHOT_CHOICES, value=100, key="sc_shots")
st.caption(f"Seeds {inputs.seed} … {inputs.seed + n_seeds - 1}. This can take from a few seconds to about "
           "a minute; it runs only when you press the button.")

sc_current = services.fingerprint(inputs, max_schedule_length=max_len, shots=sc_shots, n_seeds=n_seeds)
if st.button("Run scaling study"):
    bar = st.progress(0.0, text="Starting…")
    try:
        st.session_state["scaling_run"] = services.run_scaling(
            inputs, max_len, sc_shots, n_seeds, progress=lambda f, msg: bar.progress(min(max(f, 0.0), 1.0), text=msg)
        )
    except DashboardInputError as exc:
        st.error(str(exc))
    finally:
        bar.empty()

srun = st.session_state.get("scaling_run")
if srun is None:
    st.info("Press **Run scaling study**.")
else:
    stale_banner(srun, sc_current)
    show_figure(plot_error_vs_queries(srun.rows, theme=plot_theme()))
    st.caption(labels.CRB_NOTE)
    with st.expander("Scaling table"):
        cols = ["max_grover_power", "oracle_calls", "qae_rmse", "qae_crb_se", "mc_rmse", "mc_binomial_se",
                "qae_ci_covered", "mc_ci_covered", "n_seeds", "qae_mean_time_s", "mc_mean_time_s"]
        st.dataframe(pd.DataFrame(srun.rows)[cols], hide_index=True, width="stretch")
        st.caption("Coverage counts over this few seeds are illustrative only; the calibration study with "
                   "60,000 trials is summarised on the Limitations page.")
st.caption(labels.NO_SPEEDUP)
