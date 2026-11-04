"""Thin wrappers that run the existing scientific code for the dashboard.

Every number comes from an existing function:

* exact values: ``DiscreteEtaModel.failure_probability`` / ``reliability``
* Monte Carlo: ``qsr.simulation.monte_carlo.estimate_failure_probability``
  (convergence via ``qsr.visualization.run_mc_convergence``)
* QAE: ``qsr.quantum.estimate_failure_probability_qae``
* matched comparison: ``qsr.quantum.compare_with_monte_carlo``
* scaling study: ``qsr.quantum.error_scaling_study``

Each run records a *fingerprint*: every parameter that produced it. The UI
compares it with the current settings to mark results as stale.

Plotly and Qiskit are imported lazily inside the functions that need them, so
this module imports with only the scientific core installed.

Qiskit thread affinity (workaround): with Qiskit 2.6.1, if Qiskit is first
imported on a thread that later exits, building any circuit on another
thread crashes the process (segmentation fault). Qiskit 2.1.0 is not
affected. Streamlit runs every script rerun on a new thread, so ALL Qiskit
work here (including the first import) runs on one persistent worker thread
via :func:`_on_qiskit_thread`. Pages must never import ``qsr.quantum``
directly.
"""

from __future__ import annotations

import importlib.util
import math
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from qsr.dashboard.config import CONFIDENCE, LIMITS, DashboardInputError, Limits, ModelInputs, build_model
from qsr.simulation.monte_carlo import MonteCarloResult

Fingerprint = tuple[tuple[str, Any], ...]

#: Human-readable names for fingerprint fields (used in "stale" messages).
FIELD_LABELS = {
    "beta": "shape β",
    "etas": "η values",
    "raw_weights": "weights",
    "mission_time": "mission time",
    "time_unit": "time unit",
    "seed": "seed",
    "n_samples": "Monte Carlo samples",
    "schedule_length": "Grover schedule",
    "shots": "shots per circuit",
    "max_schedule_length": "largest schedule",
    "n_seeds": "number of seeds",
    "confidence": "confidence level",
}


# --------------------------------------------------------------------------
# environment and fingerprints
# --------------------------------------------------------------------------

def quantum_available() -> bool:
    """True if Qiskit can be imported (the optional ``quantum`` extra is installed)."""
    try:
        return importlib.util.find_spec("qiskit") is not None
    except (ImportError, ValueError):  # e.g. sys.modules["qiskit"] is None
        return False


_QISKIT_EXECUTOR: ThreadPoolExecutor | None = None
_QISKIT_LOCK = threading.Lock()


def _on_qiskit_thread(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run ``fn`` on the single long-lived Qiskit worker thread and return its result.

    Exceptions raised by ``fn`` propagate to the caller unchanged.
    """
    global _QISKIT_EXECUTOR
    with _QISKIT_LOCK:
        if _QISKIT_EXECUTOR is None:
            _QISKIT_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qsr-qiskit")
        executor = _QISKIT_EXECUTOR
    return executor.submit(fn, *args, **kwargs).result()


def fingerprint(inputs: ModelInputs, **settings: Any) -> Fingerprint:
    """All parameters that determine a run, as a hashable, ordered tuple."""
    items = dict(inputs.as_dict())
    items["confidence"] = CONFIDENCE
    items.update(settings)
    return tuple(sorted(items.items()))


def changed_fields(old: Fingerprint, new: Fingerprint) -> list[str]:
    """Human-readable names of parameters that differ between two fingerprints."""
    a, b = dict(old), dict(new)
    keys = sorted(set(a) | set(b))
    return [FIELD_LABELS.get(k, k) for k in keys if a.get(k, None) != b.get(k, None)]


def inputs_from_fingerprint(fp: Fingerprint) -> ModelInputs:
    """The model inputs a run was computed with (so stale results are shown against *their* model)."""
    d = dict(fp)
    return ModelInputs(**{k: d[k] for k in ModelInputs.__dataclass_fields__})


def _science_error(exc: ValueError) -> DashboardInputError:
    return DashboardInputError(f"The calculation rejected these settings: {exc}")


def _check_choice(name: str, value: int, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DashboardInputError(f"{name} must be a whole number.")
    if not low <= value <= high:
        raise DashboardInputError(f"{name} must be between {low:,} and {high:,}.")
    return value


# --------------------------------------------------------------------------
# exact values
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ExactValues:
    failure_probability: float
    reliability: float
    component_failure_probabilities: tuple[float, ...]


def exact_values(inputs: ModelInputs) -> ExactValues:
    built = build_model(inputs)
    t = inputs.mission_time
    return ExactValues(
        failure_probability=built.model.failure_probability(t),
        reliability=built.model.reliability(t),
        component_failure_probabilities=tuple(float(x) for x in built.model.component_failure_probabilities(t)),
    )


# --------------------------------------------------------------------------
# classical Monte Carlo
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ClassicalRun:
    fingerprint: Fingerprint
    exact: float
    result: MonteCarloResult
    convergence: tuple[MonteCarloResult, ...]


def convergence_sizes(n_samples: int) -> tuple[int, ...]:
    """Roughly log-spaced, strictly increasing sample sizes ending at ``n_samples``."""
    steps = [m * 10**k for k in range(2, 7) for m in (1, 3)]
    sizes = [s for s in steps if s < n_samples]
    return tuple(sizes + [n_samples])


def run_classical(inputs: ModelInputs, n_samples: int, limits: Limits = LIMITS) -> ClassicalRun:
    """Monte Carlo estimate plus a convergence series (same seed for every size)."""
    from qsr.visualization.plots import run_mc_convergence  # lazy: needs plotly

    built = build_model(inputs)
    n = _check_choice("Monte Carlo samples", n_samples, limits.min_mc_samples, limits.max_mc_samples)
    try:
        series = run_mc_convergence(built.model, inputs.mission_time, convergence_sizes(n), inputs.seed, CONFIDENCE)
    except ValueError as exc:
        raise _science_error(exc) from exc
    return ClassicalRun(
        fingerprint=fingerprint(inputs, n_samples=n),
        exact=built.model.failure_probability(inputs.mission_time),
        result=series[-1],  # identical to estimate_failure_probability(model, t, n, seed)
        convergence=tuple(series),
    )


# --------------------------------------------------------------------------
# quantum MLAE
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class QuantumRun:
    fingerprint: Fingerprint
    exact: float
    statevector_probability: float
    """P(failure ancilla = 1) of circuit A from an exact statevector (no sampling)."""
    result: Any  # qsr.quantum.QAEResult (typed loosely so this module imports without Qiskit)


def _require_quantum() -> None:
    if not quantum_available():
        raise DashboardInputError("Qiskit is not installed; quantum features are unavailable.")


def _schedule_impl(length: int) -> tuple[int, ...]:
    from qsr.quantum import exponential_schedule

    return exponential_schedule(length)


def _schedule(length: int, limits: Limits) -> tuple[int, ...]:
    return _on_qiskit_thread(_schedule_impl, _check_choice("Schedule length", length, 1, limits.max_schedule_length))


def schedule_for(schedule_length: int, limits: Limits = LIMITS) -> tuple[int, ...]:
    """Grover powers for a schedule length (computed by ``qsr.quantum.exponential_schedule``)."""
    _require_quantum()
    return _schedule(schedule_length, limits)


def oracle_calls_for(schedule_length: int, shots: int, limits: Limits = LIMITS) -> int:
    """Oracle queries a run will use: shots x sum(2m + 1) (the documented convention)."""
    return shots * sum(2 * m + 1 for m in schedule_for(schedule_length, limits))


def run_quantum(inputs: ModelInputs, schedule_length: int, shots: int, limits: Limits = LIMITS) -> QuantumRun:
    """One MLAE run on the seeded statevector simulator, with circuit diagnostics."""
    _require_quantum()
    built = build_model(inputs)
    sch = _schedule(schedule_length, limits)
    n_shots = _check_choice("Shots per circuit", shots, limits.min_shots, limits.max_shots)
    t = inputs.mission_time

    def work() -> tuple[Any, float]:
        from qsr.quantum import build_state_preparation, estimate_failure_probability_qae, exact_ancilla_probability

        res = estimate_failure_probability_qae(
            built.model, t, seed=inputs.seed, evaluation_schedule=sch, shots=n_shots, confidence=CONFIDENCE
        )
        return res, exact_ancilla_probability(build_state_preparation(built.model, t))

    try:
        result, sv = _on_qiskit_thread(work)
    except ValueError as exc:
        raise _science_error(exc) from exc
    return QuantumRun(
        fingerprint=fingerprint(inputs, schedule_length=schedule_length, shots=n_shots),
        exact=built.model.failure_probability(t),
        statevector_probability=sv,
        result=result,
    )


def circuit_rows(result: Any) -> list[dict[str, Any]]:
    """Per-circuit table for a QAEResult: Grover power, A-type calls, counts, depth, CX."""
    rows = []
    for k, m in enumerate(result.evaluation_schedule):
        rows.append({
            "Grover power m": m,
            "A or A† per shot (2m+1)": 2 * m + 1,
            "Shots": result.shots,
            "Oracle queries": result.shots * (2 * m + 1),
            "Failures observed": result.hits[k],
            "Observed fraction": result.hits[k] / result.shots,
            "Circuit depth": result.circuit_depths[k] if result.circuit_depths else None,
            "CX gates": result.two_qubit_gate_counts[k] if result.two_qubit_gate_counts else None,
        })
    return rows


# --------------------------------------------------------------------------
# matched comparison and scaling study
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ComparisonRun:
    fingerprint: Fingerprint
    comparison: Any  # qsr.quantum.QAEComparison


def run_comparison(inputs: ModelInputs, schedule_length: int, shots: int, limits: Limits = LIMITS) -> ComparisonRun:
    """Budget-matched QAE vs Monte Carlo via ``compare_with_monte_carlo`` (no other pairing is used)."""
    _require_quantum()
    built = build_model(inputs)
    sch = _schedule(schedule_length, limits)
    n_shots = _check_choice("Shots per circuit", shots, limits.min_shots, limits.max_shots)

    def work() -> Any:
        from qsr.quantum import compare_with_monte_carlo

        return compare_with_monte_carlo(
            built.model, inputs.mission_time, seed=inputs.seed, evaluation_schedule=sch,
            shots=n_shots, confidence=CONFIDENCE,
        )

    try:
        comp = _on_qiskit_thread(work)
    except ValueError as exc:
        raise _science_error(exc) from exc
    return ComparisonRun(fingerprint=fingerprint(inputs, schedule_length=schedule_length, shots=n_shots), comparison=comp)


def comparison_rows(comp: Any) -> list[dict[str, Any]]:
    """Exact / MC / QAE rows with identical columns, for side-by-side display."""
    q, m = comp.qae, comp.mc
    return [
        {"Quantity": "Exact model probability", "Value": comp.exact, "Interval low": None,
         "Interval high": None, "|error| vs exact": 0.0, "Model evaluations": None},
        {"Quantity": "Classical Monte Carlo estimate", "Value": m.estimate, "Interval low": m.ci_low,
         "Interval high": m.ci_high, "|error| vs exact": comp.mc_abs_error, "Model evaluations": m.n_samples},
        {"Quantity": "Quantum MLAE estimate (simulated)", "Value": q.estimate, "Interval low": q.ci_low,
         "Interval high": q.ci_high, "|error| vs exact": comp.qae_abs_error,
         "Model evaluations": q.state_preparation_calls},
    ]


@dataclass(frozen=True)
class ScalingRun:
    fingerprint: Fingerprint
    rows: tuple[dict[str, Any], ...]
    seeds: tuple[int, ...]


def run_scaling(
    inputs: ModelInputs,
    max_schedule_length: int,
    shots: int,
    n_seeds: int,
    progress: Callable[[float, str], None] | None = None,
    limits: Limits = LIMITS,
) -> ScalingRun:
    """``error_scaling_study`` one schedule length at a time, so progress can be reported.

    Seeds are ``inputs.seed, inputs.seed + 1, ...``. The rows are identical to a
    single ``error_scaling_study`` call over all lengths with the same seeds.
    """
    _require_quantum()

    def study(length: int, seeds: tuple[int, ...], n_shots: int) -> list[dict[str, Any]]:
        from qsr.quantum import error_scaling_study

        return error_scaling_study(
            built.model, inputs.mission_time, schedule_lengths=(length,), shots=n_shots,
            seeds=seeds, confidence=CONFIDENCE,
        )

    built = build_model(inputs)
    top = _check_choice("Largest schedule length", max_schedule_length, 1, limits.max_schedule_length)
    n_shots = _check_choice("Shots per circuit", shots, limits.min_shots, limits.max_shots)
    n = _check_choice("Number of seeds", n_seeds, 1, limits.max_scaling_seeds)
    seeds = tuple(range(inputs.seed, inputs.seed + n))
    rows: list[dict[str, Any]] = []
    for length in range(1, top + 1):
        if progress is not None:
            progress((length - 1) / top, f"Schedule length {length} of {top} ({n} seeds)")
        try:
            # Progress is reported on the caller's thread; only the Qiskit work moves.
            rows.extend(_on_qiskit_thread(study, length, seeds, n_shots))
        except ValueError as exc:
            raise _science_error(exc) from exc
    if progress is not None:
        progress(1.0, "Done")
    return ScalingRun(
        fingerprint=fingerprint(inputs, max_schedule_length=top, shots=n_shots, n_seeds=n),
        rows=tuple(rows),
        seeds=seeds,
    )


def format_probability(p: float) -> str:
    """Compact, precision-preserving display of a probability."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "n/a"
    if p == 0:
        return "0"
    if abs(p) < 1e-4 or abs(1 - p) < 1e-6:
        return f"{p:.6e}"
    return f"{p:.6f}"


def format_interval(low: float, high: float) -> str:
    return f"[{format_probability(low)}, {format_probability(high)}]"


def sequence_label(values: Sequence[int]) -> str:
    return ", ".join(str(v) for v in values)
