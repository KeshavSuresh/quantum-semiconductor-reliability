import math

import numpy as np
import pytest
from scipy import stats

from qsr.models.uncertainty import DiscreteEtaModel
from qsr.simulation.monte_carlo import (
    MonteCarloResult,
    estimate_failure_probability,
    sample_failure_times,
    wilson_interval,
)

# Time unit for all tests: hours.
MISSION_TIME = 5.0e3


@pytest.fixture
def model():
    return DiscreteEtaModel(beta=2.0, etas=[1.0e4, 2.0e4, 4.0e4], weights=[0.2, 0.5, 0.3])


# --- convergence -----------------------------------------------------------

def test_large_sample_estimate_close_to_exact(model):
    exact = model.failure_probability(MISSION_TIME)
    res = estimate_failure_probability(model, MISSION_TIME, 200_000, seed=7)
    assert abs(res.estimate - exact) <= 4 * math.sqrt(exact * (1 - exact) / res.n_samples)


def test_rmse_across_seeds_decreases_like_one_over_sqrt_n(model):
    """RMSE over fixed seeds, not single-run monotonicity, measures convergence."""
    exact = model.failure_probability(MISSION_TIME)
    seeds = range(40)
    sizes = [1_000, 4_000, 16_000, 64_000]
    rmse = []
    for n in sizes:
        errors = [estimate_failure_probability(model, MISSION_TIME, n, seed=s).estimate - exact for s in seeds]
        rmse.append(math.sqrt(np.mean(np.square(errors))))
    rmse = np.array(rmse)
    theory = np.sqrt(exact * (1 - exact) / np.array(sizes))
    # Each RMSE is within a statistically generous band of the binomial theory.
    assert np.all(rmse / theory > 0.6) and np.all(rmse / theory < 1.5)
    # 64x more samples -> theory predicts 8x smaller RMSE; require at least 4x.
    assert rmse[0] / rmse[-1] > 4.0


def test_component_selection_frequencies_match_weights(model):
    _, comps = sample_failure_times(model, 100_000, seed=3, return_components=True)
    counts = np.bincount(comps, minlength=model.n_components)
    result = stats.chisquare(counts, f_exp=model.weights * counts.sum())
    assert result.pvalue > 0.01


def test_samples_follow_mixture_cdf(model):
    times = sample_failure_times(model, 20_000, seed=11)
    result = stats.kstest(times, lambda t: model.failure_probability(t))
    assert result.pvalue > 0.01


def test_mission_time_zero_gives_zero_failures(model):
    res = estimate_failure_probability(model, 0.0, 1_000, seed=0)
    assert res.n_failures == 0 and res.estimate == 0.0


# --- reproducibility -------------------------------------------------------

def test_same_seed_identical_samples(model):
    a = sample_failure_times(model, 5_000, seed=42)
    b = sample_failure_times(model, 5_000, seed=42)
    np.testing.assert_array_equal(a, b)


def test_same_seed_identical_result(model):
    a = estimate_failure_probability(model, MISSION_TIME, 10_000, seed=42)
    b = estimate_failure_probability(model, MISSION_TIME, 10_000, seed=42)
    assert a == b


def test_different_seeds_differ(model):
    a = sample_failure_times(model, 1_000, seed=1)
    b = sample_failure_times(model, 1_000, seed=2)
    assert not np.array_equal(a, b)


# --- confidence interval ---------------------------------------------------

def test_result_fields_consistent(model):
    res = estimate_failure_probability(model, MISSION_TIME, 10_000, seed=5)
    assert isinstance(res, MonteCarloResult)
    assert res.estimate == res.n_failures / res.n_samples
    assert 0.0 <= res.ci_low <= res.estimate <= res.ci_high <= 1.0
    assert res.std_error == pytest.approx(math.sqrt(res.estimate * (1 - res.estimate) / res.n_samples))
    assert res.confidence == 0.95 and res.seed == 5 and res.mission_time == MISSION_TIME


def test_wilson_matches_reference_values():
    # Reference values hand-computed from the Wilson score formula, k=10, n=100, z=1.959964.
    low, high = wilson_interval(10, 100, 0.95)
    assert low == pytest.approx(0.0552291, abs=1e-6)
    assert high == pytest.approx(0.1743657, abs=1e-6)


def test_wilson_edge_cases():
    low, high = wilson_interval(0, 50)
    assert low == 0.0 and 0.0 < high < 0.1
    low, high = wilson_interval(50, 50)
    assert high == 1.0 and 0.9 < low < 1.0


def test_ci_width_shrinks_like_one_over_sqrt_n():
    w1 = np.subtract(*wilson_interval(1_000, 10_000)[::-1])
    w2 = np.subtract(*wilson_interval(10_000, 100_000)[::-1])
    assert w2 < w1
    assert w1 / w2 == pytest.approx(math.sqrt(10), rel=0.02)


def test_higher_confidence_gives_wider_interval():
    lo90, hi90 = wilson_interval(30, 300, 0.90)
    lo99, hi99 = wilson_interval(30, 300, 0.99)
    assert lo99 < lo90 and hi99 > hi90


def test_ci_empirical_coverage_near_nominal(model):
    """1000 fixed seeds; binomial SD of coverage ~0.007, so [0.93, 0.97] is ~ +/-3 SD."""
    exact = model.failure_probability(MISSION_TIME)
    hits = 0
    n_runs = 1_000
    for s in range(n_runs):
        r = estimate_failure_probability(model, MISSION_TIME, 1_000, seed=s)
        hits += r.ci_low <= exact <= r.ci_high
    coverage = hits / n_runs
    assert 0.93 <= coverage <= 0.97


# --- validation ------------------------------------------------------------

@pytest.mark.parametrize("n", [0, -10])
def test_invalid_sample_count_value(model, n):
    with pytest.raises(ValueError):
        estimate_failure_probability(model, MISSION_TIME, n, seed=0)


@pytest.mark.parametrize("n", [100.0, True, "100"])
def test_invalid_sample_count_type(model, n):
    with pytest.raises(TypeError):
        estimate_failure_probability(model, MISSION_TIME, n, seed=0)


@pytest.mark.parametrize("t", [-1.0, np.nan, np.inf, [1.0, 2.0]])
def test_invalid_mission_time(model, t):
    with pytest.raises(ValueError):
        estimate_failure_probability(model, t, 100, seed=0)


@pytest.mark.parametrize("conf", [0.0, 1.0, -0.5, 1.5])
def test_invalid_confidence(model, conf):
    with pytest.raises(ValueError):
        estimate_failure_probability(model, MISSION_TIME, 100, seed=0, confidence=conf)


def test_invalid_seed(model):
    with pytest.raises(ValueError):
        estimate_failure_probability(model, MISSION_TIME, 100, seed=-1)
    for bad in (None, 1.5, True):
        with pytest.raises(TypeError):
            estimate_failure_probability(model, MISSION_TIME, 100, seed=bad)


def test_invalid_model_type():
    with pytest.raises(TypeError):
        estimate_failure_probability({"beta": 2.0}, MISSION_TIME, 100, seed=0)


def test_wilson_invalid_counts():
    with pytest.raises(ValueError):
        wilson_interval(11, 10)
    with pytest.raises(ValueError):
        wilson_interval(-1, 10)
    with pytest.raises(TypeError):
        wilson_interval(1.0, 10)
