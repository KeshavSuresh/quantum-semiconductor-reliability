"""Every scientific label and caption shown in the dashboard, in one place.

Keeping the wording here (a) makes it reviewable, and (b) lets tests check
that the numbers quoted match ``docs/qae_design.md`` section 8.
"""

from __future__ import annotations

# --- calibration facts quoted from docs/qae_design.md section 8 (tested) ---
POOLED_COVERAGE = "0.943"
POOLED_COVERAGE_COUNTS = "56,602 / 60,000"
COVERAGE_RANGE = ("0.900", "0.994")
WORST_CELL_COVERAGE = "0.9286"

# --- the five quantities that must never be confused ---
EXACT = "Exact model probability"
EXACT_HELP = "Computed directly from the model: p = Σᵢ wᵢ · F(t | β, ηᵢ). No sampling, no randomness."
MC = "Classical Monte Carlo estimate"
MC_HELP = "Random sampling of failure times from the same model. Gets closer to the exact value as samples grow."
QAE = "Quantum MLAE estimate (simulated)"
QAE_HELP = (
    "Maximum-Likelihood Amplitude Estimation run on a classical statevector simulator of a "
    "noiseless quantum computer. Not run on quantum hardware."
)
QUERY_COMPLEXITY = "Oracle queries (theoretical cost measure)"
SIMULATOR_TIME = "Simulator wall-clock time"

# --- intervals ---
MC_CI = "95 % Wilson confidence interval"
QAE_CI = "Approximate 95 % likelihood-ratio interval"
QAE_CI_CAPTION = (
    "This interval is **approximate**, not a guaranteed 95 % interval. In the project's calibration "
    f"study, its measured coverage was {POOLED_COVERAGE} overall ({POOLED_COVERAGE_COUNTS} trials) and "
    f"ranged from {COVERAGE_RANGE[0]} to {COVERAGE_RANGE[1]} depending on the probability and schedule "
    "(100 shots per circuit). See the Limitations page."
)
MULTI_REGION_WARNING = (
    "The data are ambiguous: the likelihood supports {n} separate ranges of p. The interval shown "
    "is their outer hull, which also contains values the data reject."
)
SE_NA = "n/a (too close to p = 0 or 1 for the delta method)"

# --- costs and runtime ---
QUERY_CONVENTION = (
    "One oracle query = one application of the state-preparation circuit A (or its inverse A†) in one shot. "
    "A circuit Qᵐ·A uses 2m + 1 of them, so the total is shots × Σ(2m + 1)."
)
RUNTIME_CAPTION = (
    "Times are measured on this computer's **classical simulator** of a quantum computer. They are not "
    "quantum-hardware times and are not evidence for or against a quantum speed-up. Classical Monte "
    "Carlo is much faster here."
)
QUERY_VS_RUNTIME = (
    "Query counts measure how many model evaluations each method needs (sample efficiency). "
    "They are a theoretical comparison, separate from how long anything takes to run."
)
NO_SPEEDUP = (
    "No practical quantum speed-up is shown or claimed. Lower error at an equal number of oracle "
    "queries is a statement about query complexity on an idealised, noiseless simulator only."
)
MATCHED_BUDGET = (
    "Matched comparison: same model, mission time and seed; Monte Carlo gets exactly as many samples "
    "as QAE uses oracle queries."
)
UNMATCHED_NOTE = (
    "Runs on the Classical and Quantum pages use independent settings and are **not** a matched "
    "comparison. Use the Comparison page for a fair side-by-side."
)
CRB_NOTE = (
    "Dashed lines are theory: the asymptotic Cramér–Rao bound for QAE and the binomial standard "
    "error for Monte Carlo. The finite-sample QAE estimator can fall above or below its bound."
)

# --- state ---
STALE = "These results are out of date: they were computed with different settings. Changed: {changed}. Run again to update."
QISKIT_MISSING = (
    "Quantum features need the optional Qiskit extra, which is not installed in this environment. "
    "Install it with: `pip install -e \".[quantum]\"`. Classical results remain fully available."
)

#: Captions that must appear wherever the corresponding estimate is shown (checked by tests).
REQUIRED_ON_QUANTUM_RESULTS = (QAE, QAE_CI, NO_SPEEDUP)
