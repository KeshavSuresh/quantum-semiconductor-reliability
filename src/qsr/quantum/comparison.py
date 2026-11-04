"""Compare QAE with the exact probability and with classical Monte Carlo.

Fairness rules:

* Both methods use the same ``DiscreteEtaModel``, mission time and seed.
* Classical MC gets ``n_samples = qae.state_preparation_calls``: one classical
  sample for every application of ``A`` or ``A^dagger`` in the quantum circuits.

Two kinds of cost are reported separately and must not be confused:

* **Query complexity (theory).** Estimation error vs. number of model
  evaluations. Ideal QAE error scales ~1/N_queries, MC ~1/sqrt(N_samples).
* **Wall-clock time (measured).** Seconds on this machine. QAE runs on a
  classical statevector simulator, whose cost grows exponentially with
  qubit count, so it is far slower than MC here. This is NOT evidence about
  quantum hardware, and nothing here shows a practical quantum speed-up.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from qsr.models.uncertainty import DiscreteEtaModel
from qsr.quantum.amplitude_estimation import (
    QAEResult,
    _check_schedule,
    estimate_failure_probability_qae,
    exponential_schedule,
)
from qsr.quantum.state_preparation import _check_mission_time, _check_model
from qsr.simulation.monte_carlo import MonteCarloResult, estimate_failure_probability
from qsr.validation import check_positive_int


@dataclass(frozen=True)
class QAEComparison:
    """One QAE run and one budget-matched classical MC run against the exact value."""

    exact: float
    qae: QAEResult
    mc: MonteCarloResult
    mc_time_s: float

    @property
    def qae_abs_error(self) -> float:
        return abs(self.qae.estimate - self.exact)

    @property
    def mc_abs_error(self) -> float:
        return abs(self.mc.estimate - self.exact)


def compare_with_monte_carlo(
    model: DiscreteEtaModel,
    mission_time: float,
    *,
    seed: int,
    evaluation_schedule: Sequence[int] | None = None,
    shots: int = 100,
    confidence: float = 0.95,
) -> QAEComparison:
    """Run QAE, then classical MC with a matching budget, and attach the exact probability."""
    model = _check_model(model)
    t = _check_mission_time(mission_time)
    qae = estimate_failure_probability_qae(
        model, t, seed=seed, evaluation_schedule=evaluation_schedule, shots=shots, confidence=confidence
    )
    t0 = time.perf_counter()
    mc = estimate_failure_probability(model, t, qae.state_preparation_calls, seed=seed, confidence=confidence)
    mc_time = time.perf_counter() - t0
    return QAEComparison(exact=model.failure_probability(t), qae=qae, mc=mc, mc_time_s=mc_time)


def error_scaling_study(
    model: DiscreteEtaModel,
    mission_time: float,
    *,
    schedule_lengths: Sequence[int] = (1, 2, 3, 4, 5, 6, 7),
    shots: int = 100,
    seeds: Iterable[int] = range(20),
    confidence: float = 0.95,
) -> list[dict[str, float | int]]:
    """RMSE over fixed seeds versus query budget, for QAE (exponential schedules) and budget-matched MC.

    Each row also gives the theoretical standard errors at the true probability:

    * QAE: ``sin(2 theta) / sqrt(4 N sum_k (2 m_k + 1)^2)`` (Fisher bound)
    * MC:  ``sqrt(p (1 - p) / N_samples)``

    and the measured mean wall-clock time per run for each method.
    """
    model = _check_model(model)
    t = _check_mission_time(mission_time)
    n_shots = check_positive_int("shots", shots)
    seed_list = list(seeds)
    if not seed_list:
        raise ValueError("seeds must not be empty")
    lengths = [check_positive_int("schedule_lengths element", k) for k in schedule_lengths]
    if not lengths:
        raise ValueError("schedule_lengths must not be empty")

    exact = model.failure_probability(t)
    theta = math.asin(math.sqrt(exact))
    rows: list[dict[str, float | int]] = []
    for length in lengths:
        powers = _check_schedule(exponential_schedule(length))
        q_err, m_err, q_cov, m_cov, q_time, m_time = [], [], 0, 0, 0.0, 0.0
        budget = n_shots * sum(2 * m + 1 for m in powers)
        for s in seed_list:
            q = estimate_failure_probability_qae(
                model, t, seed=s, evaluation_schedule=powers, shots=n_shots,
                confidence=confidence, report_depth=False,
            )
            q_time += q.total_time_s
            t0 = time.perf_counter()
            m = estimate_failure_probability(model, t, budget, seed=s, confidence=confidence)
            m_time += time.perf_counter() - t0
            q_err.append(q.estimate - exact)
            m_err.append(m.estimate - exact)
            q_cov += q.ci_low <= exact <= q.ci_high
            m_cov += m.ci_low <= exact <= m.ci_high
        factors = 2.0 * np.array(powers, dtype=float) + 1.0
        n_runs = len(seed_list)
        rows.append({
            "schedule_length": length,
            "max_grover_power": max(powers),
            "oracle_calls": budget,
            "qae_rmse": float(np.sqrt(np.mean(np.square(q_err)))),
            "mc_rmse": float(np.sqrt(np.mean(np.square(m_err)))),
            "qae_theory_se": abs(math.sin(2 * theta)) / math.sqrt(4.0 * n_shots * float(np.sum(factors**2))),
            "mc_theory_se": math.sqrt(exact * (1.0 - exact) / budget),
            "qae_ci_coverage": q_cov / n_runs,
            "mc_ci_coverage": m_cov / n_runs,
            "qae_mean_time_s": q_time / n_runs,
            "mc_mean_time_s": m_time / n_runs,
            "n_seeds": n_runs,
        })
    return rows
