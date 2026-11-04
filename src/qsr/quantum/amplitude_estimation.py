"""Maximum-Likelihood Amplitude Estimation (MLAE) on core Qiskit.

Reference: Y. Suzuki et al., "Amplitude estimation without phase estimation",
Quantum Information Processing 19, 75 (2020).

Write ``a = P(anc = 1) = sin^2(theta)`` with ``theta`` in [0, pi/2]. With the
Grover operator ``Q = A S_0 A^dagger S_good`` (``S_good`` = Z on the ancilla),

    P(anc = 1 after Q^m A|0>) = sin^2((2m + 1) theta).

For each power ``m_k`` in the evaluation schedule we run ``shots`` shots and
count ``h_k`` ancilla-ones. The estimate maximises

    log L(theta) = sum_k h_k log sin^2(M_k theta) + (N - h_k) log cos^2(M_k theta),  M_k = 2 m_k + 1.

Confidence interval: likelihood-ratio interval, i.e. all ``theta`` with
``2 (log L(theta_hat) - log L(theta)) <= chi2_1(confidence)``, mapped through
``sin^2`` (monotone on [0, pi/2]). If the likelihood has several peaks, the
interval spans all of them (conservative). Suzuki et al. recommend this
interval because the Fisher (Wald) interval under-covers for short schedules.

The Fisher information ``I(theta) = 4 N sum_k M_k^2`` (exact for this model,
independent of theta) is still reported as the asymptotic standard error
``sin(2 theta_hat) / sqrt(I)``.

Query accounting (reported, not hidden):

* ``grover_applications = N * sum_k m_k``: uses of ``Q``.
* ``state_preparation_calls = N * sum_k (2 m_k + 1)``: uses of ``A`` or ``A^dagger``.
  This is the budget compared with classical Monte Carlo, where one sample
  corresponds to one evaluation of the model.

Only core Qiskit is used: ``grover_operator`` (not the deprecated
``GroverOperator`` class) and the seeded ``StatevectorSampler``.
"""

from __future__ import annotations

import math
import numbers
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.circuit.library import grover_operator
from qiskit.primitives import StatevectorSampler
from scipy.optimize import minimize_scalar
from scipy.special import xlogy
from scipy.stats import chi2

from qsr.models.uncertainty import DiscreteEtaModel
from qsr.quantum.state_preparation import _check_mission_time, _check_model, build_state_preparation
from qsr.validation import check_open_unit_interval, check_positive_int

#: Basis used only to report circuit depth and two-qubit gate counts.
DEPTH_BASIS_GATES = ("rz", "sx", "x", "cx")
MEASURE_REGISTER = "c"
DEFAULT_SCHEDULE_LENGTH = 5


@dataclass(frozen=True)
class QAEResult:
    """Outcome of one MLAE run. Timing fields are measured wall-clock seconds."""

    estimate: float
    ci_low: float
    ci_high: float
    confidence: float
    theta_hat: float
    std_error_theory: float
    """Asymptotic standard error of the estimate a: sin(2 theta_hat) / sqrt(I)."""
    evaluation_schedule: tuple[int, ...]
    shots: int
    hits: tuple[int, ...]
    grover_applications: int
    state_preparation_calls: int
    n_qubits: int
    circuit_depths: tuple[int, ...]
    """Depth of each Q^m A circuit after transpiling to DEPTH_BASIS_GATES (empty if skipped)."""
    two_qubit_gate_counts: tuple[int, ...]
    seed: int
    mission_time: float
    build_time_s: float
    transpile_time_s: float
    sampling_time_s: float
    estimation_time_s: float

    @property
    def total_time_s(self) -> float:
        return self.build_time_s + self.transpile_time_s + self.sampling_time_s + self.estimation_time_s

    @property
    def max_depth(self) -> int | None:
        return max(self.circuit_depths) if self.circuit_depths else None


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

def _check_int(name: str, value: int, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def _check_schedule(schedule: Sequence[int]) -> tuple[int, ...]:
    if isinstance(schedule, (str, bytes)) or not isinstance(schedule, Sequence):
        raise TypeError("evaluation_schedule must be a sequence of non-negative integers")
    powers = tuple(_check_int("evaluation_schedule element", m, 0) for m in schedule)
    if not powers:
        raise ValueError("evaluation_schedule must not be empty")
    if any(b <= a for a, b in zip(powers, powers[1:])):
        raise ValueError("evaluation_schedule must be strictly increasing")
    return powers


def _check_seed(seed: int) -> int:
    if seed is None:
        raise TypeError("seed must be an integer (required for reproducibility)")
    return _check_int("seed", seed, 0)


# --------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------

def exponential_schedule(length: int = DEFAULT_SCHEDULE_LENGTH) -> tuple[int, ...]:
    """Schedule ``(0, 1, 2, 4, ..., 2**(length-2))`` from Suzuki et al. ``length=1`` gives ``(0,)``."""
    n = check_positive_int("length", length)
    return (0,) + tuple(2**j for j in range(n - 1))


def build_amplification_circuits(
    state_preparation: QuantumCircuit, evaluation_schedule: Sequence[int], *, measure: bool = True
) -> list[QuantumCircuit]:
    """Circuits ``Q^m A`` for each ``m``; if ``measure`` is True, the ancilla (last qubit) is measured."""
    if not isinstance(state_preparation, QuantumCircuit):
        raise TypeError("state_preparation must be a QuantumCircuit")
    powers = _check_schedule(evaluation_schedule)
    anc = state_preparation.num_qubits - 1
    oracle = QuantumCircuit(state_preparation.num_qubits, name="S_good")
    oracle.z(anc)
    q_op = grover_operator(oracle, state_preparation=state_preparation, name="Q")
    circuits = []
    for m in powers:
        qc = state_preparation.copy(name=f"Q^{m} A")
        for _ in range(m):
            qc.compose(q_op, inplace=True)
        if measure:
            creg = ClassicalRegister(1, MEASURE_REGISTER)
            qc.add_register(creg)
            qc.measure(anc, creg[0])
        circuits.append(qc)
    return circuits


def _log_likelihood(theta: NDArray[np.float64] | float, hits, shots, factors) -> NDArray[np.float64]:
    th = np.atleast_1d(np.asarray(theta, dtype=float))[:, np.newaxis]
    angles = th * factors[np.newaxis, :]
    s2 = np.sin(angles) ** 2
    c2 = np.cos(angles) ** 2
    return np.sum(xlogy(hits, s2) + xlogy(shots - hits, c2), axis=1)


def mlae_estimate(
    hits: Sequence[int], shots: int, evaluation_schedule: Sequence[int], confidence: float = 0.95
) -> tuple[float, float, float, float, float]:
    """Maximum-likelihood estimate from measured counts.

    Returns ``(theta_hat, a_hat, ci_low, ci_high, std_error_a)``. Pure classical
    post-processing, independent of how the counts were produced.
    """
    powers = _check_schedule(evaluation_schedule)
    n = check_positive_int("shots", shots)
    h = np.array([_check_int("hits element", x, 0) for x in hits], dtype=float)
    if h.size != len(powers):
        raise ValueError("hits and evaluation_schedule must have the same length")
    if np.any(h > n):
        raise ValueError("hits cannot exceed shots")
    conf = check_open_unit_interval("confidence", confidence)

    factors = 2.0 * np.array(powers, dtype=float) + 1.0
    # Global grid search (the likelihood is multimodal), then local refinement.
    n_grid = int(min(2_000_000, max(20_001, 400 * factors.max())))
    grid = np.linspace(0.0, math.pi / 2.0, n_grid)
    ll = _log_likelihood(grid, h, n, factors)
    best = int(np.argmax(ll))
    theta_hat, best_ll = float(grid[best]), float(ll[best])
    step = grid[1] - grid[0]
    lo, hi = max(0.0, theta_hat - step), min(math.pi / 2.0, theta_hat + step)
    if hi > lo:
        res = minimize_scalar(
            lambda x: -_log_likelihood(x, h, n, factors)[0], bounds=(lo, hi), method="bounded",
            options={"xatol": 1e-13},
        )
        if res.success and -res.fun >= best_ll:
            theta_hat = float(res.x)

    best_ll = max(best_ll, float(_log_likelihood(theta_hat, h, n, factors)[0]))
    threshold = best_ll - 0.5 * float(chi2.ppf(conf, df=1))
    inside = grid[ll >= threshold]
    # Widen by one grid step on each side so grid discretisation never shrinks the interval.
    th_lo = max(0.0, min(float(inside.min()) if inside.size else theta_hat, theta_hat) - step)
    th_hi = min(math.pi / 2.0, max(float(inside.max()) if inside.size else theta_hat, theta_hat) + step)

    fisher = 4.0 * n * float(np.sum(factors**2))
    se_theta = 1.0 / math.sqrt(fisher)
    a_hat = math.sin(theta_hat) ** 2
    return theta_hat, a_hat, math.sin(th_lo) ** 2, math.sin(th_hi) ** 2, abs(math.sin(2 * theta_hat)) * se_theta


# --------------------------------------------------------------------------
# end-to-end estimator
# --------------------------------------------------------------------------

def estimate_failure_probability_qae(
    model: DiscreteEtaModel,
    mission_time: float,
    *,
    seed: int,
    evaluation_schedule: Sequence[int] | None = None,
    shots: int = 100,
    confidence: float = 0.95,
    report_depth: bool = True,
) -> QAEResult:
    """Estimate ``p = sum_i w_i F(t | beta, eta_i)`` with MLAE on a seeded statevector sampler.

    ``report_depth=False`` skips transpilation (used in large studies); depth
    and two-qubit counts are then reported as empty tuples.
    """
    model = _check_model(model)
    t = _check_mission_time(mission_time)
    seed = _check_seed(seed)
    powers = _check_schedule(exponential_schedule() if evaluation_schedule is None else evaluation_schedule)
    n_shots = check_positive_int("shots", shots)
    conf = check_open_unit_interval("confidence", confidence)

    t0 = time.perf_counter()
    a_circuit = build_state_preparation(model, t)
    circuits = build_amplification_circuits(a_circuit, powers, measure=True)
    t1 = time.perf_counter()

    depths: tuple[int, ...] = ()
    twoq: tuple[int, ...] = ()
    if report_depth:
        compiled = transpile(
            circuits, basis_gates=list(DEPTH_BASIS_GATES), optimization_level=1, seed_transpiler=0
        )
        depths = tuple(int(c.depth()) for c in compiled)
        twoq = tuple(int(c.count_ops().get("cx", 0)) for c in compiled)
    t2 = time.perf_counter()

    job = StatevectorSampler(seed=seed).run(circuits, shots=n_shots).result()
    hits = tuple(int(job[i].data[MEASURE_REGISTER].get_counts().get("1", 0)) for i in range(len(circuits)))
    t3 = time.perf_counter()

    theta_hat, a_hat, lo, hi, se_a = mlae_estimate(hits, n_shots, powers, conf)
    t4 = time.perf_counter()

    return QAEResult(
        estimate=a_hat,
        ci_low=lo,
        ci_high=hi,
        confidence=conf,
        theta_hat=theta_hat,
        std_error_theory=se_a,
        evaluation_schedule=powers,
        shots=n_shots,
        hits=hits,
        grover_applications=n_shots * sum(powers),
        state_preparation_calls=n_shots * sum(2 * m + 1 for m in powers),
        n_qubits=a_circuit.num_qubits,
        circuit_depths=depths,
        two_qubit_gate_counts=twoq,
        seed=seed,
        mission_time=t,
        build_time_s=t1 - t0,
        transpile_time_s=t2 - t1,
        sampling_time_s=t3 - t2,
        estimation_time_s=t4 - t3,
    )
