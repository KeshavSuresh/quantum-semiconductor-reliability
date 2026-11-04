"""Overview: research question, model, current configuration and the three estimates of p."""

import streamlit as st

from qsr.dashboard import labels, readout, services

from components.state import current_model

st.title("Quantum Semiconductor Reliability Explorer")
st.caption("Student research prototype · gate-oxide TDDB reliability · classical Monte Carlo and simulated "
           "quantum amplitude estimation on the same model")

st.markdown(
    "> **Research question.** For a reliability model with an uncertain parameter, how does the error of "
    "quantum amplitude estimation scale with its number of oracle queries, compared with classical Monte "
    "Carlo at the same number of model evaluations, and what does it cost on a simulated quantum computer?"
)

left, right = st.columns([1.05, 1], gap="large")
with left:
    st.subheader("Model")
    st.markdown("Time-dependent dielectric breakdown: a thin gate oxide degrades under voltage until it "
                "conducts. One population of devices fails by time *t* with the Weibull probability")
    st.latex(r"F(t \mid \beta, \eta) = 1 - e^{-(t/\eta)^{\beta}}")
    st.markdown("The characteristic life η is uncertain, so it takes values ηᵢ with weights wᵢ. "
                "Every method on this dashboard estimates the same mixture probability")
    st.latex(r"p(t) = \sum_i w_i \, F(t \mid \beta, \eta_i)")

inputs, built = current_model()
with right:
    st.subheader("Current configuration")
    if built is None:
        st.error("The sidebar model is not valid yet. Fix the highlighted input to see values.")
    else:
        exact = services.exact_values(inputs)
        st.markdown(readout.reliability_table(exact.failure_probability, exact.reliability,
                                              inputs.mission_time, inputs.time_unit))
        st.caption(readout.model_line(inputs, built))

st.subheader("Three estimates of the same p")
st.markdown(
    "| | How it is obtained | Uncertainty reported |\n|:--|:--|:--|\n"
    f"| {labels.BADGE_REFERENCE} {labels.EXACT} | Evaluated directly from the formula above | None: exact |\n"
    f"| {labels.BADGE_CLASSICAL} {labels.MC} | Random failure times drawn from the same model | 95 % Wilson interval |\n"
    f"| {labels.BADGE_QUANTUM} {labels.QAE} | Amplitude estimation on a noiseless classical simulator of a "
    "3-qubit circuit | Approximate 95 % likelihood-ratio interval |"
)
st.markdown(
    "**Workflow.** Set the model in the sidebar → **Classical Monte Carlo**: sample it → "
    "**Quantum amplitude estimation**: estimate it from circuit measurements → **Comparison**: both at the "
    "same budget, and error against budget → **Limitations**: what the results do and do not show."
)
if not services.quantum_available():
    st.info(labels.QISKIT_MISSING)
st.caption(labels.NO_SPEEDUP)
