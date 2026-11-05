"""Classical Monte Carlo estimation of the mixture failure probability.

Sampling follows exactly the generative story of
:class:`qsr.models.uncertainty.DiscreteEtaModel`:

1. Draw component index ``I ~ Categorical(w_0, ..., w_{K-1})``.
2. Draw ``U ~ Uniform[0, 1)`` and set ``T = eta_I * (-ln(1 - U))^(1/beta)``.
3. Record a failure if ``T <= t_mission``.

The fraction of failures ``k / n`` is an unbiased estimator of
``p(t_mission) = sum_i w_i F(t_mission | beta, eta_i)``, the exact value
returned by ``DiscreteEtaModel.failure_probability``.

Uncertainty is reported with the Wilson score interval, which stays inside
[0, 1] and remains informative when ``k = 0`` or ``k = n`` (common for
rare TDDB failures), unlike the normal (Wald) interval.

Units: ``mission_time`` must use the same time unit as the model's ``etas``.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from qsr.models.tddb_weibull import weibull_inverse_cdf
from qsr.models.uncertainty import DiscreteEtaModel
from qsr.validation import check_nonnegative_finite, check_open_unit_interval, check_positive_int


@dataclass(frozen=True)
class MonteCarloResult:
    """Outcome of a Monte Carlo failure-probability estimate."""

    estimate: float
    """Point estimate ``k / n``."""
    std_error: float
    """Plug-in standard error ``sqrt(p_hat (1 - p_hat) / n)``."""
    ci_low: float
    """Lower bound of the Wilson score interval."""
    ci_high: float
    """Upper bound of the Wilson score interval."""
    confidence: float
    """Nominal two-sided confidence level of the interval (e.g. 0.95)."""
    n_samples: int
    n_failures: int
    seed: int
    mission_time: float


def _check_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral):
        raise TypeError("seed must be an integer (required for reproducibility)")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    return int(seed)


def _check_model(model: DiscreteEtaModel) -> DiscreteEtaModel:
    if not isinstance(model, DiscreteEtaModel):
        raise TypeError("model must be a DiscreteEtaModel")
    return model


def _check_mission_time(mission_time: float) -> float:
    arr = check_nonnegative_finite("mission_time", mission_time)
    if arr.ndim != 0:
        raise ValueError("mission_time must be a scalar")
    return float(arr)


def wilson_interval(n_failures: int, n_samples: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion ``k / n``.

    ``z`` is the two-sided standard-normal quantile for ``confidence``
    (1.959964 for 95 %).
    """
    n = check_positive_int("n_samples", n_samples)
    if isinstance(n_failures, (bool, np.bool_)) or not isinstance(n_failures, numbers.Integral):
        raise TypeError("n_failures must be an integer")
    k = int(n_failures)
    if not 0 <= k <= n:
        raise ValueError("n_failures must satisfy 0 <= n_failures <= n_samples")
    conf = check_open_unit_interval("confidence", confidence)

    z = float(norm.ppf(0.5 + conf / 2.0))
    p_hat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p_hat + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n))
    low = 0.0 if k == 0 else max(0.0, centre - half)
    high = 1.0 if k == n else min(1.0, centre + half)
    return low, high


def sample_failure_times(
    model: DiscreteEtaModel,
    n_samples: int,
    seed: int,
    *,
    return_components: bool = False,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Draw ``n_samples`` failure times from the discrete-eta Weibull mixture.

    Returns failure times (same unit as ``model.etas``); with
    ``return_components=True`` also returns the sampled component indices.
    Identical ``seed`` values give identical outputs.
    """
    model = _check_model(model)
    n = check_positive_int("n_samples", n_samples)
    rng = np.random.default_rng(_check_seed(seed))
    components = rng.choice(model.n_components, size=n, p=model.weights)
    u = rng.random(n)
    times = np.asarray(weibull_inverse_cdf(u, model.beta, model.etas[components]), dtype=np.float64)
    if return_components:
        return times, components
    return times


def estimate_failure_probability(
    model: DiscreteEtaModel,
    mission_time: float,
    n_samples: int,
    seed: int,
    confidence: float = 0.95,
) -> MonteCarloResult:
    """Monte Carlo estimate of ``P(T <= mission_time)`` under ``model``."""
    model = _check_model(model)
    t_mission = _check_mission_time(mission_time)
    n = check_positive_int("n_samples", n_samples)
    seed = _check_seed(seed)
    conf = check_open_unit_interval("confidence", confidence)

    times = sample_failure_times(model, n, seed)
    k = int(np.count_nonzero(times <= t_mission))
    p_hat = k / n
    low, high = wilson_interval(k, n, conf)
    return MonteCarloResult(
        estimate=p_hat,
        std_error=math.sqrt(p_hat * (1.0 - p_hat) / n),
        ci_low=low,
        ci_high=high,
        confidence=conf,
        n_samples=n,
        n_failures=k,
        seed=seed,
        mission_time=t_mission,
    )
