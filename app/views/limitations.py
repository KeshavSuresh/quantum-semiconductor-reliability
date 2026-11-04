"""Scientific limitations of the comparison, in plain language."""

import streamlit as st

from qsr.dashboard import labels

st.title("Scientific limitations")
st.markdown("Read this before drawing conclusions from the other pages. Full details, with the evidence, are in "
            "`docs/qae_design.md` (sections 7 and 8) in the project repository.")

st.error(f"**No quantum speed-up is shown.** {labels.NO_SPEEDUP}", icon="🚫")

st.subheader("1. Everything quantum here is a classical simulation")
st.markdown(
    "- The circuits run on a **statevector simulator**: a classical program that tracks every amplitude "
    "exactly. Its cost grows exponentially with the number of qubits.\n"
    "- The simulation is **noiseless**: no decoherence, gate errors or readout errors. Real hardware has all three.\n"
    f"- {labels.RUNTIME_CAPTION}"
)

st.subheader("2. Oracle queries are not runtime")
st.markdown(
    f"- {labels.QUERY_CONVENTION}\n"
    f"- {labels.QUERY_VS_RUNTIME}\n"
    "- In theory, QAE's error falls like 1/queries and Monte Carlo's like 1/√samples. That advantage concerns "
    "query counts on ideal hardware only.\n"
    "- One application of A on hardware costs many gates (see circuit depth and CX counts), while one Monte "
    "Carlo sample costs a few arithmetic operations. Matching queries to samples compares sample efficiency, "
    "not cost."
)

st.subheader("3. State preparation and the oracle")
st.markdown(
    "- Loading arbitrary weights wᵢ needs a number of gates that grows exponentially with the number of index "
    "qubits. The dashboard limits the model to 8 η values (4 qubits).\n"
    "- The per-component failure probabilities Fᵢ are **computed classically** and loaded as rotation angles. "
    "Any advantage would apply only to estimating the weighted sum, not to computing the Fᵢ.\n"
    "- Circuit depths are reported after compiling to {rz, sx, x, cx} **without** hardware connectivity limits. "
    "They grow linearly with the Grover power (hundreds of layers for m = 16), far beyond today's noisy devices."
)

st.subheader("4. The QAE confidence interval is approximate")
st.markdown(
    f"- {labels.QAE_CI_CAPTION}\n"
    f"- The study tested only {labels.CALIBRATED_SHOTS} shots per circuit and Grover powers up to "
    f"m = {labels.CALIBRATED_MAX_GROVER_POWER}. For any other settings the coverage is **unknown**; the Quantum "
    "and Comparison pages warn you when you use them.\n"
    f"- The worst probability/schedule combination, re-run with 20,000 trials, had coverage "
    f"{labels.WORST_CELL_COVERAGE}.\n"
    "- Most misses come from the χ² approximation behind the interval being too narrow at 100 shots per "
    "circuit. Others come from the likelihood having several peaks, which happens with particular "
    "probability and schedule combinations.\n"
    "- When the likelihood supports several separate ranges, the dashboard shows their outer hull and warns "
    "you. The hull also contains values the data reject.\n"
    "- The Monte Carlo interval (Wilson) is a standard binomial interval; it was not part of this calibration."
)

st.subheader("5. Estimator accuracy")
st.markdown(
    "- The QAE estimate has a small measurable bias at 100 shots. Its error was 0.83 to 1.48 times the "
    "Cramér–Rao bound depending on the schedule, so the bound is an asymptotic reference, not a guarantee.\n"
    f"- {labels.CRB_NOTE}"
)

st.subheader("6. Fair comparisons only")
st.markdown(f"- {labels.MATCHED_BUDGET}\n- {labels.UNMATCHED_NOTE}\n"
            "- A seed makes each method reproducible. The two methods' random streams are unrelated, so runs "
            "are not paired.")
