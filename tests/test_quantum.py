"""Tests for the Qiskit QAE module. Skipped automatically when Qiskit is not installed."""

import math
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")

from qiskit.quantum_info import Statevector  # noqa: E402

from qsr.models.uncertainty import DiscreteEtaModel  # noqa: E402
from qsr.quantum import (  # noqa: E402
    QAEResult,
    build_amplification_circuits,
    build_state_preparation,
    compare_with_monte_carlo,
    error_scaling_study,
    estimate_failure_probability_qae,
    exact_ancilla_probability,
    exponential_schedule,
    mlae_estimate,
    n_index_qubits,
    padded_weights,
    rotation_angles,
)
from qsr.simulation.monte_carlo import estimate_failure_probability  # noqa: E402

# Time unit for all tests: hours.
MISSION_TIME = 5.0e3


@pytest.fixture
def model():
    return DiscreteEtaModel(beta=2.0, etas=[1.0e4, 2.0e4, 4.0e4], weights=[0.2, 0.5, 0.3])


MODELS = [
    DiscreteEtaModel(beta=2.0, etas=[1.0e4, 2.0e4, 4.0e4], weights=[0.2, 0.5, 0.3]),
    DiscreteEtaModel(beta=1.3, etas=[3.0e4], weights=[1.0]),                               # K = 1
    DiscreteEtaModel(beta=0.8, etas=[5e3, 1e4], weights=[0.6, 0.4]),                       # K = 2
    DiscreteEtaModel.from_unnormalized(3.0, [6e3, 8e3, 1e4, 1.5e4, 2e4], [1, 2, 4, 2, 1]),  # K = 5 (padded to 8)
]


# --- register sizing and padding --------------------------------------------

@pytest.mark.parametrize("k, n", [(1, 1), (2, 1), (3, 2), (4, 2), (5, 3), (8, 3), (9, 4)])
def test_n_index_qubits(k, n):
    assert n_index_qubits(k) == n


def test_padded_weights_and_angles(model):
    w = padded_weights(model)
    assert w.shape == (4,) and w[3] == 0.0 and w.sum() == pytest.approx(1.0, abs=1e-15)
    angles = rotation_angles(model, MISSION_TIME)
    f = model.component_failure_probabilities(MISSION_TIME)
    np.testing.assert_allclose(np.sin(angles[:3] / 2) ** 2, f, rtol=1e-12)
    assert angles[3] == 0.0


# --- state preparation correctness -------------------------------------------

@pytest.mark.parametrize("m", MODELS)
@pytest.mark.parametrize("t", [0.0, 1e3, 5e3, 2e4, 1e6])
def test_ancilla_probability_equals_exact_mixture(m, t):
    qc = build_state_preparation(m, t)
    assert qc.num_qubits == n_index_qubits(m.n_components) + 1
    assert exact_ancilla_probability(qc) == pytest.approx(m.failure_probability(t), abs=1e-12)


def test_branch_amplitudes_match_weights_and_component_probabilities(model):
    """|i>|1> has probability w_i F_i and |i>|0> has w_i (1 - F_i)."""
    qc = build_state_preparation(model, MISSION_TIME)
    probs = Statevector(qc).probabilities()  # index = i + 4 * ancilla (little-endian)
    w = padded_weights(model)
    f = np.zeros(4)
    f[:3] = model.component_failure_probabilities(MISSION_TIME)
    np.testing.assert_allclose(probs[4:], w * f, atol=1e-12)
    np.testing.assert_allclose(probs[:4], w * (1 - f), atol=1e-12)


@pytest.mark.parametrize("m", [0, 1, 2, 4, 8])
def test_grover_amplification_identity(model, m):
    a = model.failure_probability(MISSION_TIME)
    theta = math.asin(math.sqrt(a))
    (circ,) = build_amplification_circuits(build_state_preparation(model, MISSION_TIME), [m], measure=False)
    p1 = Statevector(circ).probabilities([circ.num_qubits - 1])[1]
    assert p1 == pytest.approx(math.sin((2 * m + 1) * theta) ** 2, abs=1e-10)


def test_measured_circuits_have_one_clbit(model):
    circs = build_amplification_circuits(build_state_preparation(model, MISSION_TIME), [0, 1, 2])
    assert [c.num_clbits for c in circs] == [1, 1, 1]


# --- MLAE post-processing -----------------------------------------------------

def test_exponential_schedule():
    assert exponential_schedule(1) == (0,)
    assert exponential_schedule(5) == (0, 1, 2, 4, 8)


def test_mlae_recovers_theta_from_expected_counts():
    a, schedule, shots = 0.0791843808, (0, 1, 2, 4, 8, 16), 100_000
    theta = math.asin(math.sqrt(a))
    hits = [round(shots * math.sin((2 * m + 1) * theta) ** 2) for m in schedule]
    th_hat, a_hat, lo, hi, se = mlae_estimate(hits, shots, schedule)
    assert a_hat == pytest.approx(a, abs=2e-6)
    assert lo <= a <= hi and se > 0


def test_mlae_single_power_is_classical_frequency():
    _, a_hat, lo, hi, _ = mlae_estimate([37], 100, [0])
    # Optimiser precision on a flat log-likelihood is ~1e-8; statistical error here is ~0.05.
    assert a_hat == pytest.approx(0.37, abs=1e-6)
    assert lo < 0.37 < hi


def test_mlae_zero_hits_gives_zero():
    _, a_hat, lo, hi, _ = mlae_estimate([0, 0, 0], 100, [0, 1, 2])
    assert a_hat == pytest.approx(0.0, abs=1e-12)
    assert lo == 0.0 and 0.0 < hi < 0.01


def test_mlae_ci_narrows_with_longer_schedule():
    a = 0.08
    theta = math.asin(math.sqrt(a))
    widths = []
    for length in (2, 4, 6):
        sch = exponential_schedule(length)
        hits = [round(100 * math.sin((2 * m + 1) * theta) ** 2) for m in sch]
        _, _, lo, hi, _ = mlae_estimate(hits, 100, sch)
        widths.append(hi - lo)
    assert widths[0] > widths[1] > widths[2]


@pytest.mark.parametrize(
    "hits, shots, schedule, conf, exc",
    [
        ([1, 2], 10, [0], 0.95, ValueError),        # length mismatch
        ([11], 10, [0], 0.95, ValueError),          # hits > shots
        ([-1], 10, [0], 0.95, ValueError),          # negative hits
        ([1.5], 10, [0], 0.95, TypeError),          # non-integer hits
        ([1], 0, [0], 0.95, ValueError),            # zero shots
        ([1], 10, [], 0.95, ValueError),            # empty schedule
        ([1, 1], 10, [1, 1], 0.95, ValueError),     # not strictly increasing
        ([1], 10, [-1], 0.95, ValueError),          # negative power
        ([1], 10, [0], 1.0, ValueError),            # bad confidence
    ],
)
def test_mlae_invalid_inputs(hits, shots, schedule, conf, exc):
    with pytest.raises(exc):
        mlae_estimate(hits, shots, schedule, conf)


# --- end-to-end estimator -----------------------------------------------------

def test_qae_estimate_close_to_exact(model):
    exact = model.failure_probability(MISSION_TIME)
    r = estimate_failure_probability_qae(model, MISSION_TIME, seed=2026, evaluation_schedule=exponential_schedule(6), shots=200)
    assert isinstance(r, QAEResult)
    assert abs(r.estimate - exact) < 0.005
    assert r.ci_low <= exact <= r.ci_high
    assert 0.0 <= r.ci_low <= r.estimate <= r.ci_high <= 1.0


def test_query_accounting_and_circuit_metrics(model):
    sch = (0, 1, 2, 4)
    r = estimate_failure_probability_qae(model, MISSION_TIME, seed=1, evaluation_schedule=sch, shots=50)
    assert r.grover_applications == 50 * (0 + 1 + 2 + 4)
    assert r.state_preparation_calls == 50 * (1 + 3 + 5 + 9)
    assert r.n_qubits == 3
    assert len(r.circuit_depths) == len(sch) and len(r.two_qubit_gate_counts) == len(sch)
    assert all(b > a for a, b in zip(r.circuit_depths, r.circuit_depths[1:]))
    assert r.max_depth == r.circuit_depths[-1]
    assert min(r.build_time_s, r.transpile_time_s, r.sampling_time_s, r.estimation_time_s) >= 0
    assert r.total_time_s > 0


def test_report_depth_false_skips_transpile(model):
    r = estimate_failure_probability_qae(model, MISSION_TIME, seed=1, shots=20, report_depth=False)
    assert r.circuit_depths == () and r.two_qubit_gate_counts == () and r.max_depth is None


_STATISTICAL_FIELDS = ("estimate", "ci_low", "ci_high", "theta_hat", "hits", "state_preparation_calls")


def test_reproducible_with_same_seed(model):
    a = estimate_failure_probability_qae(model, MISSION_TIME, seed=42, shots=100)
    b = estimate_failure_probability_qae(model, MISSION_TIME, seed=42, shots=100)
    for f in _STATISTICAL_FIELDS:
        assert getattr(a, f) == getattr(b, f)
    assert a.circuit_depths == b.circuit_depths  # transpiler is seeded too


def test_different_seeds_differ(model):
    a = estimate_failure_probability_qae(model, MISSION_TIME, seed=1, shots=100, report_depth=False)
    b = estimate_failure_probability_qae(model, MISSION_TIME, seed=2, shots=100, report_depth=False)
    assert a.hits != b.hits


def test_ci_coverage_over_fixed_seeds(model):
    """Likelihood-ratio CI; 100 seeds. Binomial SD of coverage ~0.022, accept [0.88, 1.0]."""
    exact = model.failure_probability(MISSION_TIME)
    hits = 0
    for s in range(100):
        r = estimate_failure_probability_qae(
            model, MISSION_TIME, seed=s, evaluation_schedule=exponential_schedule(4), shots=100, report_depth=False
        )
        hits += r.ci_low <= exact <= r.ci_high
    assert hits / 100 >= 0.88


def test_degenerate_probabilities():
    # t = 0 -> p = 0 exactly; very long t -> p = 1 (to double precision).
    m = DiscreteEtaModel(beta=2.0, etas=[1e4, 2e4], weights=[0.5, 0.5])
    r0 = estimate_failure_probability_qae(m, 0.0, seed=0, shots=50, report_depth=False)
    assert r0.estimate == pytest.approx(0.0, abs=1e-9) and r0.ci_low == 0.0
    r1 = estimate_failure_probability_qae(m, 1e7, seed=0, shots=50, report_depth=False)
    assert r1.estimate == pytest.approx(1.0, abs=1e-9) and r1.ci_high == pytest.approx(1.0)


@pytest.mark.parametrize(
    "kwargs, exc",
    [
        (dict(mission_time=-1.0), ValueError),
        (dict(mission_time=float("nan")), ValueError),
        (dict(seed=-1), ValueError),
        (dict(seed=None), TypeError),
        (dict(seed=1.5), TypeError),
        (dict(shots=0), ValueError),
        (dict(shots=10.0), TypeError),
        (dict(evaluation_schedule=[]), ValueError),
        (dict(evaluation_schedule=[0, 2, 1]), ValueError),
        (dict(evaluation_schedule="012"), TypeError),
        (dict(evaluation_schedule=[0, 1.0]), TypeError),
        (dict(confidence=0.0), ValueError),
    ],
)
def test_estimator_invalid_inputs(model, kwargs, exc):
    args = dict(mission_time=MISSION_TIME, seed=0, shots=10)
    args.update(kwargs)
    t = args.pop("mission_time")
    with pytest.raises(exc):
        estimate_failure_probability_qae(model, t, **args)


def test_invalid_model_and_circuit_types():
    with pytest.raises(TypeError):
        build_state_preparation({"beta": 2}, MISSION_TIME)
    with pytest.raises(TypeError):
        estimate_failure_probability_qae("model", MISSION_TIME, seed=0)
    with pytest.raises(TypeError):
        exact_ancilla_probability("circuit")
    with pytest.raises(TypeError):
        build_amplification_circuits("circuit", [0])


# --- comparison with classical Monte Carlo ------------------------------------

def test_comparison_uses_same_model_and_matched_budget(model):
    c = compare_with_monte_carlo(model, MISSION_TIME, seed=7, shots=50)
    assert c.exact == model.failure_probability(MISSION_TIME)
    assert c.mc.n_samples == c.qae.state_preparation_calls
    assert c.mc.seed == c.qae.seed == 7
    assert c.mc == estimate_failure_probability(model, MISSION_TIME, c.qae.state_preparation_calls, seed=7)
    assert c.qae_abs_error == pytest.approx(abs(c.qae.estimate - c.exact))
    assert c.mc_abs_error == pytest.approx(abs(c.mc.estimate - c.exact))
    assert c.mc_time_s >= 0


def test_error_scaling_study(model):
    rows = error_scaling_study(model, MISSION_TIME, schedule_lengths=(1, 3, 5), shots=100, seeds=range(15))
    assert [r["max_grover_power"] for r in rows] == [0, 2, 8]
    assert [r["oracle_calls"] for r in rows] == [100, 900, 3500]
    # With only m = 0, QAE is classical sampling: both theory SEs coincide.
    assert rows[0]["qae_theory_se"] == pytest.approx(rows[0]["mc_theory_se"], rel=1e-12)
    # Theory: QAE SE falls faster than MC SE as the budget grows.
    assert rows[-1]["qae_theory_se"] < rows[-1]["mc_theory_se"]
    # Measured: at the largest budget, QAE RMSE is below MC RMSE (statistically generous: factor 2).
    assert rows[-1]["qae_rmse"] * 2 < rows[-1]["mc_rmse"]
    for r in rows:
        assert 0 <= r["qae_ci_coverage"] <= 1 and r["n_seeds"] == 15


def test_error_scaling_study_invalid_inputs(model):
    with pytest.raises(ValueError):
        error_scaling_study(model, MISSION_TIME, seeds=[])
    with pytest.raises(ValueError):
        error_scaling_study(model, MISSION_TIME, schedule_lengths=[])
    with pytest.raises(ValueError):
        error_scaling_study(model, MISSION_TIME, schedule_lengths=[0])


# --- optional dependency ------------------------------------------------------

def test_classical_code_works_when_qiskit_is_unavailable(tmp_path):
    """Block 'import qiskit' in a fresh interpreter: classical code works; qsr.quantum gives a clear error."""
    script = textwrap.dedent(
        """
        import sys
        sys.modules["qiskit"] = None  # makes 'import qiskit' raise ImportError
        from qsr.models.uncertainty import DiscreteEtaModel
        from qsr.simulation.monte_carlo import estimate_failure_probability
        import qsr.visualization
        m = DiscreteEtaModel(beta=2.0, etas=[1e4], weights=[1.0])
        estimate_failure_probability(m, 5e3, 100, seed=0)
        try:
            import qsr.quantum
        except ImportError as exc:
            assert ".[quantum]" in str(exc), exc
            print("OK")
        else:
            raise SystemExit("qsr.quantum imported without qiskit")
        """
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=os.environ.copy())
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "OK"
