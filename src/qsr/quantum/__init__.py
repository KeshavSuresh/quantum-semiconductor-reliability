"""Quantum Amplitude Estimation (QAE) for the shared discrete eta-mixture model.

Implements the design in ``docs/qae_design.md``. The circuit encodes exactly
the quantity used by classical Monte Carlo,

    p = sum_i w_i * F(t | beta, eta_i),

as the probability that the failure ancilla is measured in ``|1>``.

Qiskit is an optional dependency: ``pip install -e ".[quantum]"``. The rest of
``qsr`` (model, Monte Carlo, plots) never imports this package.
"""

try:
    import qiskit  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised in a subprocess test
    raise ImportError(
        "qsr.quantum requires Qiskit. Install the optional extra with: pip install -e \".[quantum]\""
    ) from exc

from qsr.quantum.amplitude_estimation import (
    MLAEFit,
    QAEResult,
    build_amplification_circuits,
    circuit_seeds,
    estimate_failure_probability_qae,
    exponential_schedule,
    mlae_estimate,
)
from qsr.quantum.comparison import QAEComparison, compare_with_monte_carlo, error_scaling_study
from qsr.quantum.state_preparation import (
    build_state_preparation,
    exact_ancilla_probability,
    n_index_qubits,
    padded_weights,
    rotation_angles,
)

__all__ = [
    "MLAEFit",
    "QAEComparison",
    "QAEResult",
    "build_amplification_circuits",
    "build_state_preparation",
    "circuit_seeds",
    "compare_with_monte_carlo",
    "error_scaling_study",
    "estimate_failure_probability_qae",
    "exact_ancilla_probability",
    "exponential_schedule",
    "mlae_estimate",
    "n_index_qubits",
    "padded_weights",
    "rotation_angles",
]
