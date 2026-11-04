"""Discrete uncertainty model over the Weibull scale parameter eta.

The characteristic life ``eta`` is uncertain (process variation, lot-to-lot
spread, model uncertainty). It is represented as a discrete distribution:

    P(eta = eta_i) = w_i,   i = 0 .. K-1,   w_i >= 0,   sum_i w_i = 1

with a shared Weibull shape ``beta``. The failure probability by mission
time ``t`` is the mixture

    p(t) = sum_i w_i * F(t | beta, eta_i)

This module computes ``p(t)`` exactly. It is the SINGLE source of truth used
by both the classical Monte Carlo estimator (Milestone 1) and the Quantum
Amplitude Estimation circuit (Milestone 3), so both methods estimate exactly
the same quantity.

Units: ``etas`` and ``t`` must use the same time unit (e.g. hours).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qsr.models.tddb_weibull import weibull_cdf, weibull_reliability
from qsr.validation import check_nonnegative_finite, check_positive_finite

#: Absolute tolerance for ``sum(weights) == 1``.
WEIGHT_SUM_TOL = 1e-9


def _readonly_1d(name: str, arr: NDArray[np.float64]) -> NDArray[np.float64]:
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    out = np.array(arr, dtype=np.float64, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class DiscreteEtaModel:
    """Weibull TDDB model with a discrete distribution over ``eta``.

    Parameters
    ----------
    beta:
        Shared Weibull shape parameter (> 0, dimensionless).
    etas:
        Candidate characteristic lives ``eta_i`` (> 0), in the chosen time unit.
    weights:
        Probabilities ``w_i`` (>= 0, summing to 1 within ``WEIGHT_SUM_TOL``).
        Use :meth:`from_unnormalized` to normalize raw weights.
    """

    beta: float
    etas: NDArray[np.float64] = field(repr=True)
    weights: NDArray[np.float64] = field(repr=True)

    def __post_init__(self) -> None:
        beta_arr = check_positive_finite("beta", self.beta)
        if beta_arr.ndim != 0:
            raise ValueError("beta must be a scalar")
        etas = _readonly_1d("etas", check_positive_finite("etas", self.etas))
        weights = _readonly_1d("weights", check_nonnegative_finite("weights", self.weights))
        if etas.shape != weights.shape:
            raise ValueError(
                f"etas and weights must have the same length (got {etas.size} and {weights.size})"
            )
        total = float(weights.sum())
        if abs(total - 1.0) > WEIGHT_SUM_TOL:
            raise ValueError(f"weights must sum to 1 (got {total!r}); use from_unnormalized() to normalize")
        object.__setattr__(self, "beta", float(beta_arr))
        object.__setattr__(self, "etas", etas)
        object.__setattr__(self, "weights", weights)

    @classmethod
    def from_unnormalized(cls, beta: float, etas: ArrayLike, raw_weights: ArrayLike) -> "DiscreteEtaModel":
        """Build a model after dividing ``raw_weights`` by their sum."""
        raw = check_nonnegative_finite("raw_weights", raw_weights)
        total = float(np.sum(raw))
        if total <= 0.0:
            raise ValueError("raw_weights must have a positive sum")
        return cls(beta=beta, etas=etas, weights=raw / total)

    @property
    def n_components(self) -> int:
        """Number of discrete ``eta`` values K."""
        return int(self.etas.size)

    def component_failure_probabilities(self, t: float) -> NDArray[np.float64]:
        """Vector ``F_i = F(t | beta, eta_i)`` for scalar mission time ``t``.

        These are the per-branch failure probabilities that the QAE oracle will
        encode as ancilla rotation angles in Milestone 3.
        """
        t_arr = check_nonnegative_finite("t", t)
        if t_arr.ndim != 0:
            raise ValueError("t must be a scalar")
        return np.asarray(weibull_cdf(float(t_arr), self.beta, self.etas), dtype=np.float64)

    def failure_probability(self, t: ArrayLike) -> float | NDArray[np.float64]:
        """Exact mixture failure probability ``p(t) = sum_i w_i F(t | beta, eta_i)``.

        ``t`` may be a scalar (returns float) or an array (returns array of same shape).
        """
        t_arr = check_nonnegative_finite("t", t)
        f = weibull_cdf(t_arr[..., np.newaxis], self.beta, self.etas)
        p = np.clip(np.asarray(f) @ self.weights, 0.0, 1.0)
        return float(p) if t_arr.ndim == 0 else p

    def reliability(self, t: ArrayLike) -> float | NDArray[np.float64]:
        """Exact mixture survival probability ``R(t) = sum_i w_i R(t | beta, eta_i)``."""
        t_arr = check_nonnegative_finite("t", t)
        r = weibull_reliability(t_arr[..., np.newaxis], self.beta, self.etas)
        out = np.clip(np.asarray(r) @ self.weights, 0.0, 1.0)
        return float(out) if t_arr.ndim == 0 else out
