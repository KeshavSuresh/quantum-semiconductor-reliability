"""Two-parameter Weibull model for time-dependent dielectric breakdown (TDDB).

    F(t | beta, eta) = 1 - exp(-(t / eta) ** beta)

* ``beta`` - shape parameter (Weibull slope), dimensionless, > 0.
* ``eta``  - scale parameter (characteristic life, 63.2 % failure point), > 0.
* ``t``    - stress time, >= 0.

Units: ``t`` and ``eta`` must use the SAME time unit (e.g. both in hours).
The functions are unit-agnostic; only the ratio ``t / eta`` matters.

Extension points for later milestones (not implemented here): area scaling,
electric-field and temperature acceleration all act by transforming ``eta``
before it is passed to these functions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qsr.validation import (
    check_nonnegative_finite,
    check_positive_finite,
    check_unit_interval_half_open,
)


def _scalar_or_array(x: NDArray[np.float64]) -> float | NDArray[np.float64]:
    return float(x) if x.ndim == 0 else x


def weibull_cdf(t: ArrayLike, beta: float, eta: ArrayLike) -> float | NDArray[np.float64]:
    """Probability of breakdown by time ``t``: ``1 - exp(-(t/eta)^beta)``.

    ``t`` and ``eta`` broadcast against each other with NumPy rules.
    Uses ``-expm1(-x)`` so very small failure probabilities keep full precision.
    """
    t_arr = check_nonnegative_finite("t", t)
    beta_arr = check_positive_finite("beta", beta)
    eta_arr = check_positive_finite("eta", eta)
    x = (t_arr / eta_arr) ** beta_arr
    return _scalar_or_array(-np.expm1(-x))


def weibull_reliability(t: ArrayLike, beta: float, eta: ArrayLike) -> float | NDArray[np.float64]:
    """Survival probability ``R(t) = 1 - F(t) = exp(-(t/eta)^beta)``."""
    t_arr = check_nonnegative_finite("t", t)
    beta_arr = check_positive_finite("beta", beta)
    eta_arr = check_positive_finite("eta", eta)
    return _scalar_or_array(np.exp(-((t_arr / eta_arr) ** beta_arr)))


def weibull_inverse_cdf(u: ArrayLike, beta: float, eta: ArrayLike) -> float | NDArray[np.float64]:
    """Quantile function ``t = eta * (-ln(1 - u))^(1/beta)`` for ``u`` in [0, 1).

    Used for inverse-transform sampling: if ``U ~ Uniform[0, 1)`` then
    ``weibull_inverse_cdf(U, beta, eta)`` is Weibull(beta, eta) distributed.
    Returned times are in the same unit as ``eta``.
    """
    u_arr = check_unit_interval_half_open("u", u)
    beta_arr = check_positive_finite("beta", beta)
    eta_arr = check_positive_finite("eta", eta)
    return _scalar_or_array(eta_arr * (-np.log1p(-u_arr)) ** (1.0 / beta_arr))
