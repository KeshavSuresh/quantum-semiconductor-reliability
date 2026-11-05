"""Maximum-Likelihood Amplitude Estimation (MLAE) on core Qiskit.

Reference: Y. Suzuki et al., "Amplitude estimation without phase estimation",
Quantum Information Processing 19, 75 (2020).

Model
-----
Write ``a = P(anc = 1) = sin^2(theta)`` with ``theta`` in [0, pi/2]. With the
Grover iterate ``Q = A S_0 A^dagger S_good`` (``S_good`` = Z on the failure
ancilla, ``S_0`` = reflection about |0...0>),

    P(anc = 1 after Q^m A|0>) = sin^2((2m + 1) theta).

For each power ``m_k`` of the evaluation schedule, the circuit ``Q^{m_k} A`` is
run for ``N`` shots, and ``h_k`` = number of shots with the ancilla in ``|1>``
(failure). Different circuits use independent random streams, so the ``h_k``
are independent ``Binomial(N, sin^2(M_k theta))`` with ``M_k = 2 m_k + 1``.
The log-likelihood is

    log L(theta) = sum_k [ h_k log sin^2(M_k theta) + (N - h_k) log cos^2(M_k theta) ].

Maximum-likelihood search
-------------------------
The likelihood is multimodal (period pi / M_k per term), so a local optimiser
alone is unsafe. The procedure is:

1. Evaluate ``log L`` on a uniform grid over [0, pi/2] with step at most
   ``GRID_STEP_PER_SE`` x the Fisher standard error of theta, and at least
   ``GRID_POINTS_PER_PERIOD`` points per shortest period pi / M_max. Each peak is
   therefore resolved by about 20 or more grid points.
2. Refine the ``N_REFINE`` highest grid local maxima with bounded Brent
   optimisation inside the neighbouring grid cells, and keep the best.
   Exact ties (a genuinely ambiguous likelihood, e.g. no ``m = 0`` circuit) are
   broken towards the smallest theta. Ambiguity is then visible in the
   confidence interval (see ``n_ci_regions``).
3. Boundaries: theta = 0 (p = 0) and theta = pi/2 (p = 1) are grid points, so
   boundary maxima are found exactly.

Confidence interval
-------------------
Likelihood-ratio set ``{theta : 2 [log L(theta_hat) - log L(theta)] <= chi2_1(confidence)}``.
Its endpoints are refined by Brent root finding between grid points. If the
set consists of several disjoint regions, the reported interval is their hull
(conservative) and ``n_ci_regions > 1`` flags it. It is mapped to ``a`` through
``sin^2`` (monotone on [0, pi/2]). The chi-square calibration is ASYMPTOTIC.
Coverage at finite shots was checked by simulation (docs/qae_design.md §8),
not proved.

Standard error
--------------
For this model the Fisher information is exactly ``I(theta) = 4 N sum_k M_k^2``
(independent of theta), so ``se_theta = 1 / sqrt(I)`` is the asymptotic
Cramer-Rao standard error. The delta-method SE in ``a``,
``sin(2 theta_hat) se_theta``, is reported only when ``theta_hat`` is more
than 3 ``se_theta`` from both boundaries. Otherwise it is NaN, because the
delta method fails at a boundary.

Query accounting (convention, stated once and used everywhere)
--------------------------------------------------------------
* ONE ORACLE QUERY = one application of ``A`` or ``A^dagger`` in one shot.
  Circuit ``Q^m A`` contains ``2m + 1`` of them (verified by counting gates).
  ``state_preparation_calls = N * sum_k (2 m_k + 1)`` is the "oracle calls"
  budget, and classical Monte Carlo is given exactly this many samples.
* Reported separately, not added in: ``grover_applications = N * sum_k m_k``
  uses of ``Q``, which equals the number of phase-oracle ``S_good`` applications,
  and ``total_shots = N * K``.

Only core Qiskit is used: ``grover_operator`` (not the deprecated
``GroverOperator`` class) and ``StatevectorSampler`` with one independent
seed per circuit.
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
from scipy.optimize import brentq, minimize_scalar
from scipy.special import xlogy
from scipy.stats import chi2

from qsr.models.uncertainty import DiscreteEtaModel
from qsr.quantum.state_preparation import _check_mission_time, _check_model, build_state_preparation
from qsr.validation import check_open_unit_interval, check_positive_int

#: Basis used only to report circuit depth and two-qubit gate counts.
DEPTH_BASIS_GATES = ("rz", "sx", "x", "cx")
MEASURE_REGISTER = "c"
DEFAULT_SCHEDULE_LENGTH = 5

#: Grid step is at most this fraction of the Fisher standard error of theta.
GRID_STEP_PER_SE = 0.05
#: Grid has at least this many points per shortest likelihood period pi / M_max.
GRID_POINTS_PER_PERIOD = 200
MIN_GRID_POINTS = 20_001
#: Refuse problems that would need a larger grid (memory bound, ~40 MB per array).
MAX_GRID_POINTS = 5_000_000
#: Number of highest grid local maxima refined by the local optimiser.
N_REFINE = 5
#: Delta-method SE in ``a`` is reported only this many se_theta away from a boundary.
BOUNDARY_SE_MARGIN = 3.0
_HALF_PI = math.pi / 2.0


@dataclass(frozen=True)
class MLAEFit:
    """Result of maximum-likelihood post-processing of measured counts."""

    theta_hat: float
    estimate: float
    ci_low: float
    ci_high: float
    n_ci_regions: int
    """Number of disjoint likelihood-ratio regions; >1 means the CI is their (conservative) hull."""
    se_theta: float
    """Asymptotic (Cramer-Rao) standard error of theta: 1 / sqrt(4 N sum_k M_k^2)."""
    std_error_a: float
    """Delta-method SE of the estimate; NaN within 3 se_theta of p = 0 or p = 1."""
    grid_points: int
    grid_step: float


@dataclass(frozen=True)
class QAEResult:
    """Outcome of one MLAE run. Timing fields are measured wall-clock seconds on a simulator."""

    estimate: float
    ci_low: float
    ci_high: float
    confidence: float
    theta_hat: float
    n_ci_regions: int
    se_theta: float
    std_error_theory: float
    """Asymptotic delta-method SE of the estimate (NaN near p = 0 or 1)."""
    evaluation_schedule: tuple[int, ...]
    shots: int
    hits: tuple[int, ...]
    state_preparation_calls: int
    """Oracle queries: applications of A or A^dagger, summed over all circuits and shots."""
    grover_applications: int
    """Applications of Q, which equals the number of phase-oracle S_good applications."""
    total_shots: int
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


def circuit_seeds(seed: int, n_circuits: int) -> tuple[int, ...]:
    """Independent per-circuit sampler seeds derived from one user seed.

    ``StatevectorSampler(seed=<int>)`` re-seeds identically for every circuit in
    a job, which makes the counts of different circuits correlated. Spawning one
    child ``SeedSequence`` per circuit keeps them independent and reproducible.
    """
    s = _check_seed(seed)
    n = check_positive_int("n_circuits", n_circuits)
    return tuple(int(child.generate_state(1, dtype=np.uint64)[0] >> np.uint64(1))
                 for child in np.random.SeedSequence(s).spawn(n))


def _log_likelihood(theta, hits: NDArray[np.float64], shots: int, factors: NDArray[np.float64]) -> NDArray[np.float64]:
    th = np.atleast_1d(np.asarray(theta, dtype=float))[:, np.newaxis]
    angles = th * factors[np.newaxis, :]
    return np.sum(xlogy(hits, np.sin(angles) ** 2) + xlogy(shots - hits, np.cos(angles) ** 2), axis=1)


def _grid_size(se_theta: float, m_factor_max: float) -> int:
    by_se = math.ceil(_HALF_PI / (GRID_STEP_PER_SE * se_theta)) + 1
    by_period = math.ceil(GRID_POINTS_PER_PERIOD * m_factor_max / 2.0) + 1
    n = max(MIN_GRID_POINTS, by_se, by_period)
    if n > MAX_GRID_POINTS:
        raise ValueError(
            f"likelihood grid would need {n:,} points (> {MAX_GRID_POINTS:,}); reduce shots or the largest Grover power"
        )
    return n


def _ll_chunked(grid: NDArray[np.float64], h, n, factors, chunk: int = 250_000) -> NDArray[np.float64]:
    return np.concatenate([_log_likelihood(grid[i:i + chunk], h, n, factors) for i in range(0, grid.size, chunk)])


def mlae_estimate(
    hits: Sequence[int], shots: int, evaluation_schedule: Sequence[int], confidence: float = 0.95
) -> MLAEFit:
    """Maximum-likelihood estimate and likelihood-ratio CI from measured counts.

    Pure classical post-processing, independent of how the counts were produced.
    See the module docstring for the full procedure and its limitations.
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
    se_theta = 1.0 / math.sqrt(4.0 * n * float(np.sum(factors**2)))
    n_grid = _grid_size(se_theta, float(factors.max()))
    grid = np.linspace(0.0, _HALF_PI, n_grid)
    step = float(grid[1] - grid[0])
    ll = _ll_chunked(grid, h, n, factors)

    def f(x: float) -> float:
        return float(_log_likelihood(x, h, n, factors)[0])

    # 1-2. Global search: refine the highest grid local maxima.
    padded = np.concatenate([[-np.inf], ll, [-np.inf]])
    is_peak = (padded[1:-1] >= padded[:-2]) & (padded[1:-1] >= padded[2:]) & np.isfinite(ll)
    peaks = np.flatnonzero(is_peak)
    peaks = peaks[np.argsort(-ll[peaks], kind="stable")][:N_REFINE]
    best_theta, best_ll = float(grid[peaks[0]]), float(ll[peaks[0]])
    for i in peaks:
        lo, hi = float(grid[max(i - 1, 0)]), float(grid[min(i + 1, n_grid - 1)])
        cand, cand_ll = float(grid[i]), float(ll[i])
        res = minimize_scalar(lambda x: -f(x), bounds=(lo, hi), method="bounded", options={"xatol": 1e-12})
        if res.success and -res.fun > cand_ll:
            cand, cand_ll = float(res.x), float(-res.fun)
        # Strictly better wins; an exact tie keeps the smaller theta.
        if cand_ll > best_ll + 1e-9 or (abs(cand_ll - best_ll) <= 1e-9 and cand < best_theta):
            best_theta, best_ll = cand, max(cand_ll, best_ll)
    theta_hat = best_theta

    # Likelihood-ratio set, with endpoints refined by root finding.
    threshold = best_ll - 0.5 * float(chi2.ppf(conf, df=1))
    above = ll >= threshold
    above[np.searchsorted(grid, theta_hat).clip(0, n_grid - 1)] = True  # the maximiser is always inside
    edges = np.flatnonzero(np.diff(np.concatenate([[0], above.astype(np.int8), [0]])))
    starts, ends = edges[0::2], edges[1::2] - 1

    def g(x: float) -> float:
        return f(x) - threshold

    def left(s: int) -> float:
        if s == 0:
            return 0.0
        a, b = float(grid[s - 1]), float(grid[s])
        return brentq(g, a, b, xtol=1e-14) if g(a) < 0 <= g(b) else b

    def right(e: int) -> float:
        if e == n_grid - 1:
            return _HALF_PI
        a, b = float(grid[e]), float(grid[e + 1])
        return brentq(g, a, b, xtol=1e-14) if g(a) >= 0 > g(b) else a

    th_lo = min(left(int(starts[0])), theta_hat)
    th_hi = max(right(int(ends[-1])), theta_hat)

    near_boundary = theta_hat < BOUNDARY_SE_MARGIN * se_theta or theta_hat > _HALF_PI - BOUNDARY_SE_MARGIN * se_theta
    se_a = float("nan") if near_boundary else abs(math.sin(2.0 * theta_hat)) * se_theta
    return MLAEFit(
        theta_hat=theta_hat,
        estimate=math.sin(theta_hat) ** 2,
        ci_low=math.sin(th_lo) ** 2,
        ci_high=math.sin(th_hi) ** 2,
        n_ci_regions=int(starts.size),
        se_theta=se_theta,
        std_error_a=se_a,
        grid_points=n_grid,
        grid_step=step,
    )


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
    """Estimate ``p = sum_i w_i F(t | beta, eta_i)`` with MLAE on a seeded statevector simulator.

    Each circuit is sampled with its own seed from :func:`circuit_seeds`, so
    counts are independent across Grover powers and reproducible from ``seed``.
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

    hits = []
    for circuit, s in zip(circuits, circuit_seeds(seed, len(circuits))):
        counts = StatevectorSampler(seed=s).run([circuit], shots=n_shots).result()[0].data[MEASURE_REGISTER].get_counts()
        hits.append(int(counts.get("1", 0)))
    t3 = time.perf_counter()

    fit = mlae_estimate(hits, n_shots, powers, conf)
    t4 = time.perf_counter()

    return QAEResult(
        estimate=fit.estimate,
        ci_low=fit.ci_low,
        ci_high=fit.ci_high,
        confidence=conf,
        theta_hat=fit.theta_hat,
        n_ci_regions=fit.n_ci_regions,
        se_theta=fit.se_theta,
        std_error_theory=fit.std_error_a,
        evaluation_schedule=powers,
        shots=n_shots,
        hits=tuple(hits),
        state_preparation_calls=n_shots * sum(2 * m + 1 for m in powers),
        grover_applications=n_shots * sum(powers),
        total_shots=n_shots * len(powers),
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
