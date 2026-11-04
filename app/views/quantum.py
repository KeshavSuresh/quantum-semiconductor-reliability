"""Quantum analysis: simulated MLAE estimate, interval, circuit diagnostics and costs."""

import math

import pandas as pd
import streamlit as st

from qsr.dashboard import labels, services
from qsr.dashboard.config import LIMITS, SHOT_CHOICES, DashboardInputError

from components.state import cached_quantum, detail_line, model_summary, require_model, stale_banner

st.title("Quantum analysis")
inputs, built = require_model()
model_summary(inputs, built)

if not services.quantum_available():
    st.warning(labels.QISKIT_MISSING, icon="⚠️")
    st.markdown("Without Qiskit this page cannot run the quantum estimator. The **Classical** page and the "
                "**Limitations** page work normally.")
    st.stop()

st.markdown(f"**{labels.QAE}.** {labels.QAE_HELP}")
with st.expander("How it works, in plain language"):
    st.markdown(
        "1. **Load the uncertainty.** A small register of qubits is put into a superposition where index *i* "
        "has probability wᵢ, one per η value.\n"
        "2. **Mark failures.** An extra *failure qubit* is rotated so that, for index *i*, it reads 1 with "
        "probability Fᵢ = F(t | β, ηᵢ). Overall it reads 1 with probability exactly p. This circuit is called **A**.\n"
        "3. **Amplify.** Repeating a 'Grover' step m times changes the chance of reading 1 to sin²((2m+1)θ), "
        "where p = sin²θ. Several values of m are run, each for a number of shots.\n"
        "4. **Estimate.** The θ that best explains all the counts (maximum likelihood) gives the estimate of p.\n\n"
        "The Fᵢ values are computed classically and loaded as rotation angles. This is an idealised, noiseless "
        "simulation on a classical computer."
    )

c1, c2 = st.columns(2)
length = c1.slider("Grover schedule length", 1, LIMITS.max_schedule_length, 5, key="q_len",
                   help="Number of circuits; powers m = 0, 1, 2, 4, …")
shots = c2.select_slider("Shots per circuit", options=SHOT_CHOICES, value=100, key="q_shots")
sch = services.schedule_for(length)  # never import qsr.quantum on a page (see services docstring)
calls = services.oracle_calls_for(length, shots)
st.caption(f"Grover powers m = {services.sequence_label(sch)} → oracle queries = {shots} × "
           f"{sum(2 * m + 1 for m in sch)} = **{calls:,}**. {labels.QUERY_CONVENTION}")

current = services.fingerprint(inputs, schedule_length=length, shots=shots)
if st.button("Run QAE on the simulator", type="primary"):
    try:
        with st.spinner("Simulating quantum circuits…"):
            st.session_state["quantum_run"] = cached_quantum(inputs, length, shots)
    except DashboardInputError as exc:
        st.error(str(exc))

run = st.session_state.get("quantum_run")
if run is None:
    st.info("Press **Run QAE on the simulator** to estimate p with the quantum algorithm.")
    st.stop()

stale_banner(run, current)
r = run.result
c1, c2, c3 = st.columns(3)
c1.metric(labels.QAE, services.format_probability(r.estimate), help=labels.QAE_HELP)
c2.metric(f"{labels.EXACT} (for this run)", services.format_probability(run.exact), help=labels.EXACT_HELP)
c3.metric("|estimate − exact|", services.format_probability(abs(r.estimate - run.exact)))
detail_line(f"**{labels.QAE_CI}:** {services.format_interval(r.ci_low, r.ci_high)}",
            f"**Failures observed:** {services.sequence_label(r.hits)} (per circuit, out of {r.shots} shots each)")
st.caption(labels.QAE_CI_CAPTION)
if r.n_ci_regions > 1:
    st.warning(labels.MULTI_REGION_WARNING.format(n=r.n_ci_regions), icon="⚠️")
se = r.std_error_theory
st.caption(f"Asymptotic standard error: "
           f"{labels.SE_NA if math.isnan(se) else services.format_probability(se)} · seed {r.seed} "
           "(each circuit gets its own seed derived from it).")
st.success(f"Circuit check: the exact (noiseless) statevector of A gives P(failure qubit = 1) = "
           f"{services.format_probability(run.statevector_probability)}, which matches the exact model probability "
           f"to {abs(run.statevector_probability - run.exact):.1e}.", icon="✅")

st.subheader("Cost of the quantum estimate")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Qubits", r.n_qubits)
k2.metric(labels.QUERY_COMPLEXITY, f"{r.state_preparation_calls:,}")
k3.metric("Grover steps (= phase-oracle calls)", f"{r.grover_applications:,}")
k4.metric("Largest circuit depth", f"{r.max_depth:,}" if r.max_depth is not None else "n/a")
st.caption(labels.QUERY_CONVENTION + " Depth and CX counts are after compiling to {rz, sx, x, cx} with no "
           "hardware connectivity limits; real devices would need more gates.")
st.dataframe(pd.DataFrame(services.circuit_rows(r)), hide_index=True, width="stretch",
             column_config={"Observed fraction": st.column_config.NumberColumn(format="%.3f")})

st.subheader(labels.SIMULATOR_TIME)
t1, t2, t3, t4, t5 = st.columns(5)
t1.metric("Build circuits", f"{r.build_time_s:.3f} s")
t2.metric("Compile (for depth)", f"{r.transpile_time_s:.3f} s")
t3.metric("Simulate", f"{r.sampling_time_s:.3f} s")
t4.metric("Estimate", f"{r.estimation_time_s:.3f} s")
t5.metric("Total", f"{r.total_time_s:.3f} s")
st.caption(labels.RUNTIME_CAPTION)
st.caption(labels.NO_SPEEDUP)
st.caption(labels.UNMATCHED_NOTE)
