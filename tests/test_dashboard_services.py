"""Tests for the Streamlit-free dashboard logic (``qsr.dashboard``).

These run without Streamlit. Quantum-dependent tests are skipped without Qiskit.
"""

import dataclasses
import importlib.util
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from qsr.dashboard import labels
from qsr.dashboard.config import (
    DEFAULT_INPUTS,
    LIMITS,
    DashboardInputError,
    ModelInputs,
    build_model,
)
from qsr.dashboard.services import (
    changed_fields,
    comparison_rows,
    convergence_sizes,
    exact_values,
    fingerprint,
    format_probability,
    quantum_available,
    run_classical,
)
from qsr.models.uncertainty import DiscreteEtaModel
from qsr.simulation.monte_carlo import estimate_failure_probability

ROOT = pathlib.Path(__file__).resolve().parents[1]
needs_qiskit = pytest.mark.skipif(not quantum_available(), reason="Qiskit not installed")
# run_classical builds its convergence series with the Plotly-backed visualization helper.
needs_plotly = pytest.mark.skipif(importlib.util.find_spec("plotly") is None, reason="Plotly not installed")


def _inputs(**kw):
    return dataclasses.replace(DEFAULT_INPUTS, **kw)


# --- model construction and weights ---------------------------------------------

def test_defaults_match_documented_example_model():
    built = build_model(DEFAULT_INPUTS)
    ref = DiscreteEtaModel(beta=2.0, etas=[1e4, 2e4, 4e4], weights=[0.2, 0.5, 0.3])
    assert built.model.failure_probability(5e3) == ref.failure_probability(5e3)
    assert f"{built.model.failure_probability(5e3):.6f}" == "0.079184"  # README / docs value
    assert not built.was_normalized


def test_raw_and_normalized_weights_are_both_reported():
    built = build_model(_inputs(raw_weights=(2.0, 5.0, 3.0)))
    assert built.raw_weights == (2.0, 5.0, 3.0)
    np.testing.assert_allclose(built.normalized_weights, (0.2, 0.5, 0.3))
    assert built.raw_weight_sum == 10.0 and built.was_normalized
    # Normalisation is done by the existing model code, not re-implemented.
    ref = DiscreteEtaModel.from_unnormalized(2.0, (1e4, 2e4, 4e4), (2.0, 5.0, 3.0))
    np.testing.assert_array_equal(built.model.weights, ref.weights)


@pytest.mark.parametrize(
    "kw, match",
    [
        (dict(raw_weights=(0.0, 0.0, 0.0)), "greater than zero"),
        (dict(raw_weights=(-0.1, 0.6, 0.5)), "negative"),
        (dict(etas=(1e4, 0.0, 4e4)), "η"),
        (dict(etas=(1e4, 2e4), raw_weights=(0.5, 0.3, 0.2)), "exactly one weight"),
        (dict(etas=(), raw_weights=()), "at least one"),
        (dict(etas=tuple(1e4 + i for i in range(9)), raw_weights=(1.0,) * 9), "At most 8"),
        (dict(beta=0.0), "β"),
        (dict(beta=float("nan")), "finite"),
        (dict(mission_time=-1.0), "Mission time"),
        (dict(time_unit=" "), "Time unit"),
        (dict(seed=-1), "Seed"),
        (dict(etas=(1e4, float("inf"), 4e4)), "finite"),
    ],
)
def test_invalid_inputs_give_readable_errors(kw, match):
    with pytest.raises(DashboardInputError, match=match):
        build_model(_inputs(**kw))


def test_parse_component_rows_ignores_empty_rows():
    from qsr.dashboard.config import parse_component_rows

    rows = [(1e4, 0.2), (None, None), (2e4, 0.5), (float("nan"), float("nan")), (4e4, 0.3)]
    assert parse_component_rows(rows) == ((1e4, 2e4, 4e4), (0.2, 0.5, 0.3))


@pytest.mark.parametrize(
    "rows, match",
    [
        ([(1e4, 0.2), (3e4, None)], "Row 2 is incomplete: enter a raw weight"),
        ([(1e4, 0.2), (float("nan"), 0.4)], "Row 2 is incomplete: enter η,"),
        ([(1e4, 0.2), ("", 0.4)], "Row 2 is incomplete"),
        ([(1e4, float("inf"))], "Raw weight in row 1 must be a finite number"),
    ],
)
def test_parse_component_rows_reports_the_row(rows, match):
    from qsr.dashboard.config import parse_component_rows

    with pytest.raises(DashboardInputError, match=match):
        parse_component_rows(rows)


def test_negative_values_are_reported_not_altered():
    from qsr.dashboard.config import parse_component_rows

    etas, weights = parse_component_rows([(1e4, -1.0), (2e4, 0.5)])
    assert weights == (-1.0, 0.5)  # passed through unchanged...
    with pytest.raises(DashboardInputError, match="Weights cannot be negative"):
        build_model(_inputs(etas=etas, raw_weights=weights))  # ...and rejected with a clear message
    etas, weights = parse_component_rows([(-5.0, 1.0)])
    with pytest.raises(DashboardInputError, match="greater than zero"):
        build_model(_inputs(etas=etas, raw_weights=weights))


def test_exact_values_delegate_to_model():
    ev = exact_values(DEFAULT_INPUTS)
    m = build_model(DEFAULT_INPUTS).model
    assert ev.failure_probability == m.failure_probability(5e3)
    assert ev.reliability == m.reliability(5e3)
    np.testing.assert_array_equal(ev.component_failure_probabilities, m.component_failure_probabilities(5e3))


# --- fingerprints / staleness --------------------------------------------------------

@pytest.mark.parametrize(
    "field, value",
    [("beta", 2.5), ("etas", (1e4, 2e4, 5e4)), ("raw_weights", (0.3, 0.4, 0.3)),
     ("mission_time", 6e3), ("time_unit", "days"), ("seed", 7)],
)
def test_any_model_input_change_marks_result_stale(field, value):
    old = fingerprint(DEFAULT_INPUTS, n_samples=1000)
    new = fingerprint(_inputs(**{field: value}), n_samples=1000)
    assert old != new
    assert len(changed_fields(old, new)) == 1


def test_run_setting_change_marks_result_stale():
    old = fingerprint(DEFAULT_INPUTS, shots=100, schedule_length=5)
    assert changed_fields(old, fingerprint(DEFAULT_INPUTS, shots=300, schedule_length=5)) == ["shots per circuit"]
    assert changed_fields(old, fingerprint(DEFAULT_INPUTS, shots=100, schedule_length=6)) == ["Grover schedule"]
    assert changed_fields(old, fingerprint(DEFAULT_INPUTS, shots=100, schedule_length=5)) == []


def test_raw_weight_scaling_is_still_a_change():
    """(2,5,3) and (0.2,0.5,0.3) give the same model, but the user changed an input: flag it."""
    old = fingerprint(DEFAULT_INPUTS)
    new = fingerprint(_inputs(raw_weights=(2.0, 5.0, 3.0)))
    assert changed_fields(old, new) == ["weights"]


# --- classical -----------------------------------------------------------------------

def test_convergence_sizes():
    assert convergence_sizes(100) == (100,)
    assert convergence_sizes(10_000) == (100, 300, 1_000, 3_000, 10_000)
    s = convergence_sizes(2_000_000)
    assert s[-1] == 2_000_000 and all(b > a for a, b in zip(s, s[1:]))


@needs_plotly
def test_run_classical_equals_direct_monte_carlo_call():
    run = run_classical(DEFAULT_INPUTS, 30_000)
    model = build_model(DEFAULT_INPUTS).model
    assert run.result == estimate_failure_probability(model, 5e3, 30_000, seed=2026, confidence=0.95)
    assert run.exact == model.failure_probability(5e3)
    assert [r.n_samples for r in run.convergence] == list(convergence_sizes(30_000))
    assert run.fingerprint == fingerprint(DEFAULT_INPUTS, n_samples=30_000)


@needs_plotly
def test_run_classical_reproduces_readme_numbers():
    r = run_classical(DEFAULT_INPUTS, 1_000_000).result
    assert f"{r.estimate:.6f}" == "0.079239"  # milestone1_demo.py / README, seed 2026


@needs_plotly
@pytest.mark.parametrize("n", [10, 3_000_000, 1000.0, True])
def test_run_classical_limits(n):
    with pytest.raises(DashboardInputError):
        run_classical(DEFAULT_INPUTS, n)


# --- quantum -------------------------------------------------------------------------

@needs_qiskit
def test_run_quantum_equals_direct_call():
    from qsr.dashboard.services import circuit_rows, run_quantum
    from qsr.quantum import estimate_failure_probability_qae, exponential_schedule

    run = run_quantum(DEFAULT_INPUTS, 4, 50)
    direct = estimate_failure_probability_qae(
        build_model(DEFAULT_INPUTS).model, 5e3, seed=2026, evaluation_schedule=exponential_schedule(4), shots=50
    )
    for f in ("estimate", "ci_low", "ci_high", "hits", "state_preparation_calls", "circuit_depths"):
        assert getattr(run.result, f) == getattr(direct, f)
    assert run.statevector_probability == pytest.approx(run.exact, abs=1e-12)
    rows = circuit_rows(run.result)
    assert [r["Grover power m"] for r in rows] == [0, 1, 2, 4]
    assert sum(r["Oracle queries"] for r in rows) == run.result.state_preparation_calls


@needs_qiskit
def test_run_quantum_reproduces_documented_single_run():
    from qsr.dashboard.services import run_quantum

    r = run_quantum(DEFAULT_INPUTS, 6, 100).result
    assert (f"{r.estimate:.6f}", r.state_preparation_calls, r.circuit_depths) == (
        "0.078668", 6_800, (25, 81, 137, 249, 473, 921))


@needs_qiskit
def test_comparison_is_budget_matched_and_rows_consistent():
    from qsr.dashboard.services import run_comparison

    comp = run_comparison(DEFAULT_INPUTS, 3, 30).comparison
    assert comp.mc.n_samples == comp.qae.state_preparation_calls
    assert comp.mc.seed == comp.qae.seed == 2026
    rows = comparison_rows(comp)
    assert [r["Quantity"] for r in rows] == [labels.EXACT, labels.MC, labels.QAE]
    assert rows[1]["Model evaluations"] == rows[2]["Model evaluations"]


@needs_qiskit
def test_run_scaling_matches_error_scaling_study_and_reports_progress():
    from qsr.dashboard.services import run_scaling
    from qsr.quantum import error_scaling_study

    seen = []
    run = run_scaling(DEFAULT_INPUTS, 3, 20, 3, progress=lambda f, msg: seen.append(f))
    direct = error_scaling_study(build_model(DEFAULT_INPUTS).model, 5e3, schedule_lengths=(1, 2, 3), shots=20,
                                 seeds=range(2026, 2029))
    strip = lambda rows: [{k: v for k, v in r.items() if not k.endswith("_time_s")} for r in rows]  # noqa: E731
    assert strip(run.rows) == strip(direct)  # timings differ run to run; everything else identical
    assert seen[0] == 0.0 and seen[-1] == 1.0 and seen == sorted(seen)
    assert run.seeds == (2026, 2027, 2028)


@needs_qiskit
@pytest.mark.parametrize("kw", [dict(schedule_length=0, shots=100), dict(schedule_length=8, shots=100),
                                dict(schedule_length=3, shots=5), dict(schedule_length=3, shots=5000)])
def test_quantum_limits(kw):
    from qsr.dashboard.services import run_quantum

    with pytest.raises(DashboardInputError):
        run_quantum(DEFAULT_INPUTS, **kw)


# --- calibration scope and timing split ---------------------------------------------

@pytest.mark.parametrize("schedule, shots, covered, fragments", [
    ((0, 1, 2, 4, 8, 16), 100, True, []),
    ((0,), 100, True, []),
    ((0, 1, 2, 4, 8, 16, 32), 100, False, ["m = 32"]),
    ((0, 1, 2), 30, False, ["30 shots per circuit instead of 100"]),
    ((0, 1, 2, 4, 8, 16, 32), 1000, False, ["1000 shots", "m = 32"]),
])
def test_calibration_status(schedule, shots, covered, fragments):
    from qsr.dashboard.services import calibration_status

    status = calibration_status(schedule, shots)
    assert status.covered is covered
    assert len(status.reasons) == len(fragments)
    for f in fragments:
        assert f in status.reason_text


def test_calibration_scope_matches_design_document():
    doc = (ROOT / "docs" / "qae_design.md").read_text(encoding="utf-8")
    assert labels.CALIBRATED_SHOTS == 100 and "N = 100 shots per circuit" in doc
    assert labels.CALIBRATED_MAX_GROVER_POWER == 16 and "m_max ∈ {0, 1, 2, 4, 8, 16}" in doc


def test_qae_timing_split_reads_measured_fields():
    from types import SimpleNamespace

    from qsr.dashboard.services import qae_timing

    r = SimpleNamespace(build_time_s=0.01, transpile_time_s=0.25, sampling_time_s=0.2, estimation_time_s=0.03)
    t = qae_timing(r)
    assert t.execution_s == pytest.approx(0.24)
    assert t.diagnostics_compile_s == 0.25
    assert t.total_s == pytest.approx(0.49)


# --- formatting, labels, isolation ---------------------------------------------------

def test_format_probability():
    assert format_probability(0.07918438) == "0.079184"
    assert format_probability(0.0) == "0"
    assert format_probability(1.33e-5) == "1.330000e-05"
    assert format_probability(float("nan")) == "n/a"


def test_quoted_calibration_numbers_match_design_document():
    doc = (ROOT / "docs" / "qae_design.md").read_text(encoding="utf-8")
    for s in (labels.POOLED_COVERAGE, labels.POOLED_COVERAGE_COUNTS, *labels.COVERAGE_RANGE, labels.WORST_CELL_COVERAGE):
        assert s in doc, s
    assert "approximate" in labels.QAE_CI.lower() and "not a guaranteed" in labels.QAE_CI_CAPTION


def test_limits_keep_quantum_circuits_small():
    from math import ceil, log2
    assert ceil(log2(LIMITS.max_components)) + 1 <= 4  # index qubits + ancilla


@needs_plotly
def test_dashboard_package_imports_without_streamlit_or_qiskit():
    code = (
        "import sys; sys.modules['streamlit'] = None; sys.modules['qiskit'] = None\n"
        "from qsr.dashboard import config, labels, services\n"
        "assert services.quantum_available() is False\n"
        "r = services.run_classical(config.DEFAULT_INPUTS, 1000)\n"
        "try:\n"
        "    services.run_quantum(config.DEFAULT_INPUTS, 3, 100)\n"
        "except config.DashboardInputError as e:\n"
        "    assert 'Qiskit' in str(e)\n"
        "else:\n"
        "    raise SystemExit('quantum ran without qiskit')\n"
        "print('OK')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=os.environ.copy())
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "OK"


_THREAD_PATTERN = """
import threading
{setup}
def first():  # like Streamlit's first script run: Qiskit is first imported on a thread that then exits
    {call}
def later():  # a later rerun on a brand-new thread
    {call}
for f in (first, later, later):
    t = threading.Thread(target=f); t.start(); t.join()
print("OK")
"""


@needs_qiskit
def test_quantum_runs_survive_streamlit_thread_pattern():
    """Regression: Qiskit 2.6.1 segfaults if first imported on a thread that later exits.

    The dashboard routes all Qiskit work through one persistent worker thread.
    """
    code = _THREAD_PATTERN.format(
        setup="from qsr.dashboard import config, services",
        call="services.run_quantum(config.DEFAULT_INPUTS, 2, 10); services.run_comparison(config.DEFAULT_INPUTS, 2, 10)",
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=os.environ.copy())
    assert out.returncode == 0, f"exit {out.returncode}: {out.stderr[-500:]}"
    assert out.stdout.strip() == "OK"


def test_model_inputs_is_hashable_for_caching():
    assert hash(DEFAULT_INPUTS) == hash(ModelInputs(**DEFAULT_INPUTS.as_dict()))
