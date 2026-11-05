"""Comparison: budget-matched QAE vs Monte Carlo, and the error-versus-oracle-queries study.

Accuracy, query budget and runtime are shown in separate sections so they are
never read as one combined score.
"""

import pandas as pd
import streamlit as st

from qsr.dashboard import labels, readout, services
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
from components.style import overline

st.title("Comparison")
inputs, built = require_model()
model_summary(inputs, built)

if not services.quantum_available():
    st.warning(labels.QISKIT_MISSING)
    st.markdown("The matched comparison needs the quantum estimator. Classical results are on the "
                "**Classical Monte Carlo** page.")
    st.stop()

from qsr.visualization import plot_error_vs_queries, plot_estimates_vs_reference  # noqa: E402

st.markdown(labels.MATCHED_BUDGET)

# --------------------------------------------------------------------------
overline("1 · Matched comparison: one budget for both methods")
c1, c2 = st.columns(2, gap="large")
# Defaults match the documented example and demo guide (m up to 16, 100 shots: calibrated settings).
length = c1.slider("Grover schedule length (circuits)", 1, LIMITS.max_schedule_length, 6, key="cmp_len")
shots = c2.select_slider("Shots per circuit", options=SHOT_CHOICES, value=100, key="cmp_shots")
sch = services.schedule_for(length)
st.caption(f"Budget: {shots} × {sum(2 * m + 1 for m in sch)} = **{services.oracle_calls_for(length, shots):,}** "
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
    st.caption("No result yet. Press **Run matched comparison**.")
else:
    stale_banner(run, current)
    comp = run.comparison
    mc_row = readout.classical_row(comp.mc, comp.exact)
    q_row = readout.quantum_row(comp.qae, comp.exact)

    overline("2 · Accuracy at an equal number of model evaluations")
    estimates = [
        {"kind": "classical", "label": f"Monte Carlo · {comp.mc.n_samples:,} samples",
         "estimate": comp.mc.estimate, "low": comp.mc.ci_low, "high": comp.mc.ci_high,
         "detail": labels.MC_CI},
        {"kind": "quantum", "label": f"QAE (simulated) · {comp.qae.state_preparation_calls:,} queries",
         "estimate": comp.qae.estimate, "low": comp.qae.ci_low, "high": comp.qae.ci_high,
         "detail": labels.QAE_CI},
    ]
    try:
        show_figure(plot_estimates_vs_reference(comp.exact, estimates, theme=plot_theme(),
                                                title="Both estimates and their intervals against the exact value"))
    except ValueError as exc:  # the figure validates its input strictly; the tables below stay available
        st.caption(f"Interval figure not drawn ({exc}); the values are in the table below.")
    st.markdown(readout.readout_table([readout.reference_row(comp.exact), mc_row, q_row]))
    st.caption(f"Monte Carlo interval: {labels.MC_CI}. QAE interval: {labels.QAE_CI}. One run per method from "
               "one seed: a single comparison, not an average. The scaling study below repeats it over seeds.")
    interval_scope_notice(comp.qae)
    if comp.qae.n_ci_regions > 1:
        st.warning(labels.MULTI_REGION_WARNING.format(n=comp.qae.n_ci_regions))
    with st.expander("Data table"):
        df = pd.DataFrame(services.comparison_rows(comp))
        st.dataframe(df, hide_index=True, width="stretch", column_config={
            "Value": st.column_config.NumberColumn(format="%.6f"),
            "Interval low": st.column_config.NumberColumn(format="%.6f"),
            "Interval high": st.column_config.NumberColumn(format="%.6f"),
            "|error| vs exact": st.column_config.NumberColumn(format="%.6f"),
            "Model evaluations": st.column_config.NumberColumn(format="%d"),
        })

    overline("3 · Cost: query budget and wall-clock time, kept apart")
    timing = services.qae_timing(comp.qae)
    ratio = timing.execution_s / comp.mc_time_s if comp.mc_time_s > 0 else float("inf")
    budget_col, time_col = st.columns(2, gap="large")
    with budget_col:
        st.markdown("**Query budget** (theoretical cost measure)")
        st.markdown(readout.facts_table([
            ("Monte Carlo samples", f"{comp.mc.n_samples:,}"),
            ("QAE oracle queries", f"{comp.qae.state_preparation_calls:,}"),
            ("QAE circuits × shots", f"{len(comp.qae.evaluation_schedule)} × {comp.qae.shots}"),
        ], header=("Method", "Model evaluations")))
        st.caption(labels.QUERY_VS_RUNTIME)
    with time_col:
        st.markdown(f"**{labels.SIMULATOR_TIME}** (not part of the accuracy comparison)")
        st.markdown(readout.facts_table([
            ("Classical Monte Carlo", readout.seconds(comp.mc_time_s, 4)),
            (labels.QAE_EXECUTION_TIME, readout.seconds(timing.execution_s, 4)),
            ("QAE execution ÷ MC time", f"{ratio:,.0f}×"),
            (labels.QAE_COMPILE_TIME, readout.seconds(timing.diagnostics_compile_s, 4)),
        ], header=("Stage", "Time")))
        st.caption(labels.COMPILE_NOTE)
    st.caption(labels.RUNTIME_CAPTION)

# --------------------------------------------------------------------------
overline("4 · Error against oracle queries, over many seeds")
st.markdown("Repeats both methods over several seeds and budgets, and plots the root-mean-square error against "
            "the number of oracle queries (QAE) or samples (MC). This is a **query-complexity** comparison.")
c1, c2, c3 = st.columns(3)
max_len = c1.slider("Largest schedule length", 2, LIMITS.max_schedule_length, 6, key="sc_len")
n_seeds = c2.slider("Seeds per point", 3, LIMITS.max_scaling_seeds, 20, key="sc_seeds")
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
    st.caption("No result yet. Press **Run scaling study**.")
else:
    stale_banner(srun, sc_current)
    show_figure(plot_error_vs_queries(srun.rows, theme=plot_theme()))
    st.caption(labels.CRB_NOTE)
    with st.expander("Scaling data table"):
        cols = ["max_grover_power", "oracle_calls", "qae_rmse", "qae_crb_se", "mc_rmse", "mc_binomial_se",
                "qae_ci_covered", "mc_ci_covered", "n_seeds", "qae_mean_time_s", "mc_mean_time_s"]
        st.dataframe(pd.DataFrame(srun.rows)[cols], hide_index=True, width="stretch")
        st.caption("Coverage counts over this few seeds are illustrative only; the calibration study with "
                   "60,000 trials is summarised on the Limitations page.")
st.caption(labels.NO_SPEEDUP)
