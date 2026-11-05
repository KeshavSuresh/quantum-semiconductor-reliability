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
    circuit_seeds,
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

def _brute_force_theta(hits, shots, schedule, n=2_000_001):
    """Independent dense evaluation of the log-likelihood (reference for the MLE search)."""
    from scipy.special import xlogy
    th = np.linspace(0.0, math.pi / 2, n)
    ll = np.zeros(n)
    for h, m in zip(hits, schedule):
        mk = 2 * m + 1
        ll += xlogy(h, np.sin(mk * th) ** 2) + xlogy(shots - h, np.cos(mk * th) ** 2)
    return th, ll


def _ll(theta, hits, shots, schedule):
    return sum(
        h * math.log(math.sin((2 * m + 1) * theta) ** 2) + (shots - h) * math.log(math.cos((2 * m + 1) * theta) ** 2)
        for h, m in zip(hits, schedule)
    )


def test_exponential_schedule():
    assert exponential_schedule(1) == (0,)
    assert exponential_schedule(5) == (0, 1, 2, 4, 8)


def test_likelihood_uses_amplified_probability():
    """For one power m with h hits, the MLE solves sin^2((2m+1) theta) = h/N (with m = 0 to fix the branch)."""
    fit = mlae_estimate([30, 0], 100, [0, 1])  # m=1: sin^2(3 theta) = 0 -> theta = pi/3 -> a = 0.75 conflicts
    # The two terms disagree; compare with brute force instead of a closed form.
    th, ll = _brute_force_theta([30, 0], 100, [0, 1])
    assert abs(fit.theta_hat - th[np.argmax(ll)]) < 1e-5
    single = mlae_estimate([37], 100, [0])
    # Optimiser precision on a flat log-likelihood is ~1e-8; statistical error here is ~0.05.
    assert single.estimate == pytest.approx(0.37, abs=1e-6)


@pytest.mark.parametrize(
    "a, schedule, shots",
    [(0.0791843808, (0, 1, 2, 4, 8, 16), 100), (0.3, (0, 1, 2, 4, 8), 100), (0.0791843808, exponential_schedule(7), 100_000)],
)
def test_mle_matches_brute_force_on_expected_counts(a, schedule, shots):
    theta = math.asin(math.sqrt(a))
    hits = [round(shots * math.sin((2 * m + 1) * theta) ** 2) for m in schedule]
    fit = mlae_estimate(hits, shots, schedule)
    th, ll = _brute_force_theta(hits, shots, schedule)
    assert abs(fit.theta_hat - th[np.argmax(ll)]) < 2 * (th[1] - th[0])
    assert _ll(fit.theta_hat, hits, shots, schedule) >= ll.max() - 1e-6
    assert fit.ci_low <= a <= fit.ci_high
    assert fit.grid_step <= 0.05 * fit.se_theta + 1e-15  # documented resolution


def test_multimodal_likelihood_global_maximum_and_regions():
    """Schedule (0, 8): many local maxima. The global one must be found; extra LR regions are flagged."""
    theta = math.asin(math.sqrt(0.3))
    hits = [30, round(100 * math.sin(17 * theta) ** 2)]
    fit = mlae_estimate(hits, 100, (0, 8))
    th, ll = _brute_force_theta(hits, 100, (0, 8))
    assert abs(fit.theta_hat - th[np.argmax(ll)]) < 1e-5
    # Number of local maxima of the likelihood on the dense grid (confirms the case is multimodal).
    assert np.sum((ll[1:-1] > ll[:-2]) & (ll[1:-1] > ll[2:])) >= 5
    assert fit.n_ci_regions == 2
    assert fit.ci_low <= 0.3 <= fit.ci_high


def test_exactly_ambiguous_likelihood_is_reported_not_hidden():
    """Only m = 2: sin^2(5 theta) = 0.4 has three equally likely solutions in [0, pi/2]."""
    fit = mlae_estimate([40], 100, (2,))
    roots = [(math.asin(math.sqrt(0.4)) + k * math.pi) / 5 for k in (0, 1, 2)] + [
        (math.pi - math.asin(math.sqrt(0.4)) + k * math.pi) / 5 for k in (0, 1)
    ]
    assert fit.theta_hat == pytest.approx(min(roots), abs=1e-6)  # documented tie-break: smallest theta
    assert fit.n_ci_regions > 1
    assert fit.ci_high - fit.ci_low > 0.9  # interval honestly spans the ambiguity


def test_lr_interval_endpoints_are_refined_roots():
    theta = math.asin(math.sqrt(0.0791843808))
    sch = exponential_schedule(5)
    hits = [round(100 * math.sin((2 * m + 1) * theta) ** 2) for m in sch]
    fit = mlae_estimate(hits, 100, sch)
    from scipy.stats import chi2
    target = chi2.ppf(0.95, 1)
    llmax = _ll(fit.theta_hat, hits, 100, sch)
    for a_end in (fit.ci_low, fit.ci_high):
        th_end = math.asin(math.sqrt(a_end))
        assert 2 * (llmax - _ll(th_end, hits, 100, sch)) == pytest.approx(target, abs=1e-6)


def test_boundary_p0_and_p1():
    z = mlae_estimate([0, 0, 0], 100, [0, 1, 2])
    assert z.estimate == 0.0 and z.ci_low == 0.0 and 0.0 < z.ci_high < 0.01
    assert math.isnan(z.std_error_a)  # delta method is invalid at the boundary
    o = mlae_estimate([100, 100, 100], 100, [0, 1, 2])
    assert o.estimate == pytest.approx(1.0, abs=1e-15) and o.ci_high == pytest.approx(1.0, abs=1e-15)
    assert 0.99 < o.ci_low < 1.0 and math.isnan(o.std_error_a)


def test_interior_standard_error_is_delta_method_fisher():
    fit = mlae_estimate([37], 100, [0])
    assert fit.se_theta == pytest.approx(1 / math.sqrt(4 * 100))
    assert fit.std_error_a == pytest.approx(abs(math.sin(2 * fit.theta_hat)) * fit.se_theta)
    # For m = 0 this equals the binomial SE sqrt(a (1 - a) / N).
    assert fit.std_error_a == pytest.approx(math.sqrt(0.37 * 0.63 / 100), rel=1e-6)


def test_ci_narrows_with_longer_schedule():
    a = 0.08
    theta = math.asin(math.sqrt(a))
    widths = []
    for length in (2, 4, 6):
        sch = exponential_schedule(length)
        hits = [round(100 * math.sin((2 * m + 1) * theta) ** 2) for m in sch]
        fit = mlae_estimate(hits, 100, sch)
        widths.append(fit.ci_high - fit.ci_low)
    assert widths[0] > widths[1] > widths[2]


def test_grid_too_large_is_refused():
    with pytest.raises(ValueError, match="grid"):
        mlae_estimate([0, 0], 10_000_000, [0, 2048])


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


def test_query_count_matches_executed_circuits(model):
    """Count A / A-dagger signatures (UCRY gates) and phase oracles (Z) inside each executed circuit."""
    sch = exponential_schedule(6)
    circuits = build_amplification_circuits(build_state_preparation(model, MISSION_TIME), sch)
    per_shot_a = []
    for m, c in zip(sch, circuits):
        names = [inst.operation.name for inst in c.data]
        n_a = sum(n in ("ucry", "ucry_dg") for n in names)
        assert n_a == 2 * m + 1
        assert sum(n in ("state_preparation", "state_preparation_dg") for n in names) == 2 * m + 1
        assert names.count("z") == m  # one S_good per Grover iterate
        per_shot_a.append(n_a)
    r = estimate_failure_probability_qae(model, MISSION_TIME, seed=0, evaluation_schedule=sch, shots=100, report_depth=False)
    assert r.state_preparation_calls == 100 * sum(per_shot_a) == 6_800
    assert r.grover_applications == 100 * sum(sch) == 3_100
    assert r.total_shots == 100 * len(sch)


def test_circuit_seeds_are_distinct_and_reproducible():
    a = circuit_seeds(2026, 6)
    assert a == circuit_seeds(2026, 6) and len(set(a)) == 6
    assert a != circuit_seeds(2027, 6)
    assert circuit_seeds(2026, 3) == a[:3]


def test_counts_are_independent_across_circuits(model):
    """Regression: an integer sampler seed reused per circuit made counts correlated across Grover powers."""
    h0, h1 = [], []
    for s in range(300, 500):
        r = estimate_failure_probability_qae(model, MISSION_TIME, seed=s, evaluation_schedule=(0, 1), shots=50, report_depth=False)
        h0.append(r.hits[0])
        h1.append(r.hits[1])
    corr = np.corrcoef(h0, h1)[0, 1]
    assert abs(corr) < 0.25  # independent: SD of sample correlation ~ 1/sqrt(200) = 0.07


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


def test_ci_coverage_smoke(model):
    """Smoke test only (100 runs, seeds 5000-5099, disjoint from any tuning runs).

    Calibration evidence with thousands of trials is in examples/milestone3_ci_calibration.py
    and docs/qae_design.md. With true coverage 0.95, P(< 88/100) is about 0.003.
    """
    exact = model.failure_probability(MISSION_TIME)
    hits = 0
    for s in range(5000, 5100):
        r = estimate_failure_probability_qae(
            model, MISSION_TIME, seed=s, evaluation_schedule=exponential_schedule(4), shots=100, report_depth=False
        )
        hits += r.ci_low <= exact <= r.ci_high
    assert hits / 100 >= 0.88


def test_weights_at_tolerance_edge_prepare_correctly():
    for eps in (9e-10, -9e-10):
        m = DiscreteEtaModel(beta=2.0, etas=[1e4, 2e4], weights=[0.5, 0.5 + eps])
        assert exact_ancilla_probability(build_state_preparation(m, MISSION_TIME)) == pytest.approx(
            m.failure_probability(MISSION_TIME), abs=2e-9
        )


def test_component_probabilities_exactly_zero_and_one():
    """F_1 = 1 (eta tiny), F_2 ~ 1e-18 (eta huge): P(anc=1) must equal w_1 exactly up to rounding."""
    m = DiscreteEtaModel(beta=2.0, etas=[1.0, 1e12], weights=[0.3, 0.7])
    f = m.component_failure_probabilities(1e3)
    assert f[0] == 1.0
    angles = rotation_angles(m, 1e3)
    assert angles[0] == pytest.approx(math.pi) and angles[1] == pytest.approx(0.0, abs=1e-8)
    assert exact_ancilla_probability(build_state_preparation(m, 1e3)) == pytest.approx(m.failure_probability(1e3), abs=1e-12)


@pytest.mark.parametrize("m_model", MODELS)
def test_endianness_branch_probabilities_for_all_models(m_model):
    t = 7e3
    qc = build_state_preparation(m_model, t)
    n = qc.num_qubits - 1
    probs = Statevector(qc).probabilities()
    w = padded_weights(m_model)
    f = np.zeros(2**n)
    f[: m_model.n_components] = m_model.component_failure_probabilities(t)
    np.testing.assert_allclose(probs[2**n:], w * f, atol=1e-12)          # ancilla = 1 block
    np.testing.assert_allclose(probs[: 2**n], w * (1 - f), atol=1e-12)   # ancilla = 0 block
    assert np.all(probs[m_model.n_components: 2**n] < 1e-15)            # padded indices carry nothing
    assert np.all(probs[2**n + m_model.n_components:] < 1e-15)


@pytest.mark.parametrize("m_model", MODELS)
@pytest.mark.parametrize("t", [2e3, 2e4])
@pytest.mark.parametrize("power", [1, 3, 8])
def test_amplification_identity_across_models(m_model, t, power):
    a = m_model.failure_probability(t)
    theta = math.asin(math.sqrt(a))
    (circ,) = build_amplification_circuits(build_state_preparation(m_model, t), [power], measure=False)
    assert Statevector(circ).probabilities([circ.num_qubits - 1])[1] == pytest.approx(
        math.sin((2 * power + 1) * theta) ** 2, abs=1e-10
    )


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
    # With only m = 0, QAE is classical sampling: the Cramer-Rao bound equals the binomial SE.
    assert rows[0]["qae_crb_se"] == pytest.approx(rows[0]["mc_binomial_se"], rel=1e-12)
    # Theory: QAE bound falls faster than MC SE as the budget grows.
    assert rows[-1]["qae_crb_se"] < rows[-1]["mc_binomial_se"]
    # Measured: at the largest budget, QAE RMSE is below MC RMSE (statistically generous: factor 2).
    assert rows[-1]["qae_rmse"] * 2 < rows[-1]["mc_rmse"]
    for r in rows:
        assert 0 <= r["qae_ci_coverage"] <= 1 and r["n_seeds"] == 15
        assert r["qae_ci_coverage"] == r["qae_ci_covered"] / 15 and r["mc_ci_coverage"] == r["mc_ci_covered"] / 15
        assert 0 <= r["qae_multi_region_cis"] <= 15
    # RMSE definition: recompute one row directly.
    exact = model.failure_probability(MISSION_TIME)
    errs = [estimate_failure_probability_qae(model, MISSION_TIME, seed=s, evaluation_schedule=exponential_schedule(3),
                                             shots=100, report_depth=False).estimate - exact for s in range(15)]
    assert rows[1]["qae_rmse"] == pytest.approx(math.sqrt(np.mean(np.square(errs))), rel=1e-12)


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
