"""Overview: purpose, plain-language model explanation, workflow."""

import streamlit as st

from qsr.dashboard import labels, services

from components.state import current_model, model_summary

st.title("Quantum Semiconductor Reliability Explorer")
st.markdown(
    "A student research prototype that estimates how likely a chip's insulating layer is to break down "
    "by a given time, and compares **three ways of getting that number**: the exact formula, classical "
    "Monte Carlo sampling, and a simulated quantum algorithm (Quantum Amplitude Estimation)."
)

st.subheader("The reliability question")
st.markdown(
    "Transistor gates are separated from the channel by a very thin oxide. Under voltage it slowly "
    "degrades until it suddenly conducts: **time-dependent dielectric breakdown (TDDB)**. We ask: "
    "*what fraction of devices has failed by the mission time t?*"
)
c1, c2 = st.columns(2)
with c1:
    st.markdown("**1. Weibull failure law.** One population of devices fails by time t with probability")
    st.latex(r"F(t \mid \beta, \eta) = 1 - e^{-(t/\eta)^{\beta}}")
    st.caption("β is the shape (how sharply failures cluster); η is the characteristic life (63 % have failed by t = η).")
with c2:
    st.markdown("**2. Uncertain η.** We are not sure of η, so we allow several values ηᵢ with weights wᵢ:")
    st.latex(r"p = \sum_i w_i \, F(t \mid \beta, \eta_i)")
    st.caption("This mixture probability p is the single target that every method on this dashboard estimates.")

st.subheader("Three ways to get p")
st.markdown(
    f"- **{labels.EXACT}**: {labels.EXACT_HELP}\n"
    f"- **{labels.MC}**: {labels.MC_HELP}\n"
    f"- **{labels.QAE}**: {labels.QAE_HELP}"
)

st.subheader("How to use this dashboard")
st.markdown(
    "1. Set the model in the **sidebar**: shape β, the η values and their weights, the mission time and a seed.\n"
    "2. **Classical**: see the exact probability and curves, then run Monte Carlo.\n"
    "3. **Quantum**: run the simulated quantum estimator and inspect its circuits and costs.\n"
    "4. **Comparison**: a fair, budget-matched side-by-side, plus an error-versus-cost study.\n"
    "5. **Limitations**: read what these results do and do not show."
)

inputs, built = current_model()
st.subheader("Current model")
if built is None:
    st.error("The sidebar model is not valid yet. Fix the highlighted input to see results.")
else:
    model_summary(inputs, built)
if not services.quantum_available():
    st.info(labels.QISKIT_MISSING, icon="ℹ️")
st.caption(labels.NO_SPEEDUP)
