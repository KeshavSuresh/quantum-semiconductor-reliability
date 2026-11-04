"""Quantum amplitude estimation: setup, result against the reference, then diagnostics and runtime."""

import math

import pandas as pd
import streamlit as st

from qsr.dashboard import labels, readout, services
from qsr.dashboard.config import LIMITS, SHOT_CHOICES, DashboardInputError

from components.state import (
    cached_quantum,
    interval_scope_notice,
    model_summary,
    require_model,
    scope_preview,
    stale_banner,
)
from components.style import overline

st.title("Quantum amplitude estimation")
inputs, built = require_model()
model_summary(inputs, built)

if not services.quantum_available():
    st.warning(labels.QISKIT_MISSING)
    st.markdown("Without Qiskit this page cannot run the quantum estimator. The **Classical Monte Carlo** "
                "page and the **Limitations** page work normally.")
    st.stop()

st.markdown(f"{labels.QAE_HELP} The circuit encodes the same mixture probability p; amplitude estimation "
            "recovers it from measurement counts.")
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

overline("1 · Setup")
c1, c2 = st.columns(2, gap="large")
# Default 6 (m up to 16, 100 shots): the documented example, inside the calibration study's settings.
length = c1.slider("Grover schedule length (circuits)", 1, LIMITS.max_schedule_length, 6, key="q_len",
                   help="Number of circuits; powers m = 0, 1, 2, 4, …")
shots = c2.select_slider("Shots per circuit", options=SHOT_CHOICES, value=100, key="q_shots")
sch = services.schedule_for(length)  # never import qsr.quantum on a page (see services docstring)
calls = services.oracle_calls_for(length, shots)
st.caption(f"Grover powers m = {services.sequence_label(sch)} → oracle queries = {shots} × "
           f"{sum(2 * m + 1 for m in sch)} = **{calls:,}**. {labels.QUERY_CONVENTION}")
scope_preview(sch, shots)
pressed = st.button("Run QAE on the simulator", type="primary")

current = services.fingerprint(inputs, schedule_length=length, shots=shots)
if pressed:
    try:
        with st.spinner("Simulating quantum circuits…"):
            st.session_state["quantum_run"] = cached_quantum(inputs, length, shots)
    except DashboardInputError as exc:
        st.error(str(exc))

run = st.session_state.get("quantum_run")
overline("2 · Result")
if run is None:
    st.caption("No result yet. Press **Run QAE on the simulator** to estimate p with the quantum algorithm.")
    st.stop()

stale_banner(run, current)
r = run.result
st.markdown(readout.readout_table([readout.reference_row(run.exact), readout.quantum_row(r, run.exact)],
                                  interval_label="Approx. 95 % interval"))
interval_scope_notice(r)
if r.n_ci_regions > 1:
    st.warning(labels.MULTI_REGION_WARNING.format(n=r.n_ci_regions))
se = r.std_error_theory
st.caption(f"Interval: {labels.QAE_CI} · failures observed per circuit "
           f"{services.sequence_label(r.hits)} (of {r.shots} shots each) · asymptotic standard error "
           f"{labels.SE_NA if math.isnan(se) else services.format_probability(se)} · seed {r.seed} "
           "(each circuit gets its own seed derived from it).")
st.caption(f"Circuit check: the exact (noiseless) statevector of A gives P(failure qubit = 1) = "
           f"{services.format_probability(run.statevector_probability)}, which matches the exact model probability "
           f"to {abs(run.statevector_probability - run.exact):.1e}.")

overline("3 · Diagnostics: circuit cost")
cost, cost_note = st.columns([3, 2], gap="large")
cost.markdown(readout.facts_table([
    ("Qubits", f"{r.n_qubits}"),
    (labels.QUERY_COMPLEXITY, f"{r.state_preparation_calls:,}"),
    ("Grover steps (= phase-oracle calls)", f"{r.grover_applications:,}"),
    ("Largest circuit depth", f"{r.max_depth:,}" if r.max_depth is not None else "n/a"),
]))
cost_note.caption("Oracle queries are counted as defined under Setup. Depth and CX counts are after compiling to "
                  "{rz, sx, x, cx} with no hardware connectivity limits; real devices would need more gates.")
with st.expander("Per-circuit counts, depth and CX gates"):
    st.dataframe(pd.DataFrame(services.circuit_rows(r)), hide_index=True, width="stretch",
                 column_config={"Observed fraction": st.column_config.NumberColumn(format="%.3f")})

overline(f"4 · Diagnostics: {labels.SIMULATOR_TIME.lower()}")
timing = services.qae_timing(r)
times, times_note = st.columns([3, 2], gap="large")
times.markdown(readout.facts_table([
    ("Build circuits", readout.seconds(r.build_time_s, 3)),
    ("Simulate", readout.seconds(r.sampling_time_s, 3)),
    ("Estimate", readout.seconds(r.estimation_time_s, 3)),
    ("**Execution total** (build + simulate + estimate)", f"**{readout.seconds(timing.execution_s, 3)}**"),
    ("Compile (transpile), for circuit diagnostics only", readout.seconds(timing.diagnostics_compile_s, 3)),
    ("Everything incl. compile", readout.seconds(timing.total_s, 3)),
], header=("Stage", "Wall-clock time")))
times_note.caption(labels.COMPILE_NOTE)
times_note.caption(labels.RUNTIME_CAPTION)
st.caption(labels.NO_SPEEDUP)
st.caption(labels.UNMATCHED_NOTE)
