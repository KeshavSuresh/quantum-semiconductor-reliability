"""Shared input-validation helpers.

All public model and simulation functions validate their inputs through these
helpers so that error messages are consistent across the package.
"""

from __future__ import annotations

import numbers

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_float_array(name: str, value: ArrayLike) -> NDArray[np.float64]:
    if value is None:
        raise TypeError(f"{name} must be numeric, not None")
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be numeric, not bool")
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if arr.dtype == np.bool_:
        raise TypeError(f"{name} must be numeric, not bool")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite (no NaN or inf)")
    return arr


def check_positive_finite(name: str, value: ArrayLike) -> NDArray[np.float64]:
    """Return ``value`` as a float array after checking every element is > 0 and finite."""
    arr = _as_float_array(name, value)
    if np.any(arr <= 0):
        raise ValueError(f"{name} must be strictly positive")
    return arr


def check_nonnegative_finite(name: str, value: ArrayLike) -> NDArray[np.float64]:
    """Return ``value`` as a float array after checking every element is >= 0 and finite."""
    arr = _as_float_array(name, value)
    if np.any(arr < 0):
        raise ValueError(f"{name} must be non-negative")
    return arr


def check_unit_interval_half_open(name: str, value: ArrayLike) -> NDArray[np.float64]:
    """Return ``value`` as a float array after checking every element is in [0, 1)."""
    arr = _as_float_array(name, value)
    if np.any((arr < 0) | (arr >= 1)):
        raise ValueError(f"{name} must lie in the half-open interval [0, 1)")
    return arr


def check_open_unit_interval(name: str, value: float) -> float:
    """Return ``value`` as a float after checking it is a scalar strictly inside (0, 1)."""
    arr = _as_float_array(name, value)
    if arr.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    v = float(arr)
    if not 0.0 < v < 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1")
    return v


def check_positive_int(name: str, value: int) -> int:
    """Return ``value`` after checking it is an integer >= 1 (bools and floats are rejected)."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return int(value)
