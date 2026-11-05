"""Dashboard inputs, defaults, limits and conversion to ``DiscreteEtaModel``.

The defaults equal the example model used throughout the README and
``docs/qae_design.md`` so the dashboard's default numbers can be checked
against the documentation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from qsr.models.uncertainty import DiscreteEtaModel


class DashboardInputError(ValueError):
    """Raised with a user-readable message when dashboard inputs are invalid."""


@dataclass(frozen=True)
class Limits:
    """UI limits that keep runs interactive on a laptop-class simulator."""

    max_components: int = 8  # at most 3 index qubits + 1 ancilla = 4 qubits
    min_mc_samples: int = 100
    max_mc_samples: int = 2_000_000
    min_shots: int = 10
    max_shots: int = 1_000
    max_schedule_length: int = 7  # Grover powers up to m = 32
    max_scaling_seeds: int = 30
    min_beta: float = 0.05
    max_beta: float = 20.0


LIMITS = Limits()

#: Monte Carlo sample-count choices offered in the UI.
MC_SAMPLE_CHOICES: tuple[int, ...] = (1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000, 2_000_000)
#: Shots-per-circuit choices offered in the UI.
SHOT_CHOICES: tuple[int, ...] = (10, 30, 100, 300, 1_000)
#: Confidence level of every interval in the dashboard. The MLAE calibration
#: study (docs/qae_design.md section 8) was run at 95 % only.
CONFIDENCE = 0.95


@dataclass(frozen=True)
class ModelInputs:
    """Everything the user sets in the sidebar. Hashable, so usable as a cache key."""

    beta: float
    etas: tuple[float, ...]
    raw_weights: tuple[float, ...]
    mission_time: float
    time_unit: str = "h"
    seed: int = 2026

    def as_dict(self) -> dict[str, object]:
        return {
            "beta": self.beta,
            "etas": self.etas,
            "raw_weights": self.raw_weights,
            "mission_time": self.mission_time,
            "time_unit": self.time_unit,
            "seed": self.seed,
        }


DEFAULT_INPUTS = ModelInputs(
    beta=2.0,
    etas=(1.0e4, 2.0e4, 4.0e4),
    raw_weights=(0.2, 0.5, 0.3),
    mission_time=5.0e3,
    time_unit="h",
    seed=2026,
)


@dataclass(frozen=True)
class BuiltModel:
    """A validated model plus the information the UI needs to show about weights."""

    inputs: ModelInputs
    model: DiscreteEtaModel
    raw_weights: tuple[float, ...]
    normalized_weights: tuple[float, ...]
    raw_weight_sum: float

    @property
    def was_normalized(self) -> bool:
        """True if the raw weights did not already sum to 1 (to 1e-9)."""
        return abs(self.raw_weight_sum - 1.0) > 1e-9


def _finite(name: str, value: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise DashboardInputError(f"{name} must be a number.") from exc
    if not math.isfinite(v):
        raise DashboardInputError(f"{name} must be a finite number.")
    return v


def _missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def parse_component_rows(rows: list[tuple[object, object]]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Turn editor rows ``(eta, raw_weight)`` into tuples, with row-specific messages.

    Completely empty rows (e.g. a freshly added row) are ignored. A row with only
    one of the two values filled in is reported by its 1-based row number.
    """
    etas: list[float] = []
    weights: list[float] = []
    for i, (eta, weight) in enumerate(rows, start=1):
        eta_missing, weight_missing = _missing(eta), _missing(weight)
        if eta_missing and weight_missing:
            continue
        if eta_missing or weight_missing:
            missing = "η" if eta_missing else "a raw weight"
            raise DashboardInputError(f"Row {i} is incomplete: enter {missing}, or delete the row.")
        etas.append(_finite(f"η in row {i}", eta))
        weights.append(_finite(f"Raw weight in row {i}", weight))
    return tuple(etas), tuple(weights)


def build_model(inputs: ModelInputs, limits: Limits = LIMITS) -> BuiltModel:
    """Validate sidebar inputs and build the model with ``DiscreteEtaModel.from_unnormalized``.

    Raw weights may be any non-negative numbers with a positive sum; they are
    normalised by the existing model code, and both versions are returned so
    the UI can display them side by side.
    """
    if not isinstance(inputs, ModelInputs):
        raise TypeError("inputs must be a ModelInputs")
    beta = _finite("Shape β", inputs.beta)
    if not limits.min_beta <= beta <= limits.max_beta:
        raise DashboardInputError(f"Shape β must be between {limits.min_beta} and {limits.max_beta}.")
    t = _finite("Mission time", inputs.mission_time)
    if t < 0:
        raise DashboardInputError("Mission time must be zero or positive.")
    if not isinstance(inputs.time_unit, str) or not inputs.time_unit.strip():
        raise DashboardInputError("Time unit must be a non-empty label, e.g. 'h'.")
    if isinstance(inputs.seed, bool) or not isinstance(inputs.seed, int) or inputs.seed < 0:
        raise DashboardInputError("Seed must be a non-negative whole number.")

    etas = tuple(_finite("Every η", e) for e in inputs.etas)
    raw = tuple(_finite("Every raw weight", w) for w in inputs.raw_weights)
    if len(etas) == 0:
        raise DashboardInputError("Add at least one η component.")
    if len(etas) != len(raw):
        raise DashboardInputError("Every η needs exactly one weight.")
    if len(etas) > limits.max_components:
        raise DashboardInputError(f"At most {limits.max_components} η components are supported in the dashboard.")
    if any(e <= 0 for e in etas):
        raise DashboardInputError("Every η (characteristic life) must be greater than zero.")
    if any(w < 0 for w in raw):
        raise DashboardInputError("Weights cannot be negative.")
    total = float(sum(raw))
    if total <= 0:
        raise DashboardInputError("At least one weight must be greater than zero.")

    try:
        model = DiscreteEtaModel.from_unnormalized(beta, etas, raw)
    except ValueError as exc:  # defensive: model validation is the source of truth
        raise DashboardInputError(str(exc)) from exc
    return BuiltModel(
        inputs=inputs,
        model=model,
        raw_weights=raw,
        normalized_weights=tuple(float(w) for w in np.asarray(model.weights)),
        raw_weight_sum=total,
    )
