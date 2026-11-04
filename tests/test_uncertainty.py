import numpy as np
import pytest

from qsr.models.tddb_weibull import weibull_cdf
from qsr.models.uncertainty import DiscreteEtaModel

# Time unit for all tests: hours.
BETA = 2.0
ETAS = [1.0e4, 2.0e4, 4.0e4]
WEIGHTS = [0.2, 0.5, 0.3]


@pytest.fixture
def model():
    return DiscreteEtaModel(beta=BETA, etas=ETAS, weights=WEIGHTS)


def test_valid_model_weights_sum_to_one(model):
    assert model.weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert model.n_components == 3


def test_arrays_are_readonly_copies(model):
    src = np.array(ETAS)
    m = DiscreteEtaModel(beta=BETA, etas=src, weights=WEIGHTS)
    src[0] = 1.0
    assert m.etas[0] == ETAS[0]
    with pytest.raises(ValueError):
        model.etas[0] = 5.0


@pytest.mark.parametrize(
    "etas, weights",
    [
        (ETAS, [0.2, 0.5, 0.2]),          # sums to 0.9
        (ETAS, [0.5, 0.5, 0.5]),          # sums to 1.5
        (ETAS, [-0.1, 0.8, 0.3]),         # negative weight
        (ETAS, [np.nan, 0.5, 0.5]),       # NaN weight
        (ETAS, [0.5, 0.5]),               # length mismatch
        ([], []),                         # empty
        ([1e4, 0.0, 4e4], WEIGHTS),       # non-positive eta
        ([1e4, np.inf, 4e4], WEIGHTS),    # infinite eta
        ([[1e4, 2e4, 4e4]], [WEIGHTS]),   # 2-D input
    ],
)
def test_invalid_models_rejected(etas, weights):
    with pytest.raises(ValueError):
        DiscreteEtaModel(beta=BETA, etas=etas, weights=weights)


@pytest.mark.parametrize("beta", [0.0, -2.0, np.nan, [1.0, 2.0]])
def test_invalid_beta_rejected(beta):
    with pytest.raises(ValueError):
        DiscreteEtaModel(beta=beta, etas=ETAS, weights=WEIGHTS)


def test_from_unnormalized_normalizes():
    m = DiscreteEtaModel.from_unnormalized(BETA, ETAS, [2, 5, 3])
    np.testing.assert_allclose(m.weights, WEIGHTS)
    assert m.weights.sum() == pytest.approx(1.0, abs=1e-15)


def test_from_unnormalized_rejects_zero_sum():
    with pytest.raises(ValueError):
        DiscreteEtaModel.from_unnormalized(BETA, ETAS, [0, 0, 0])


def test_exact_mixture_matches_hand_calculation(model):
    t = 5.0e3  # hours
    expected = (
        0.2 * (1 - np.exp(-(5e3 / 1e4) ** 2))
        + 0.5 * (1 - np.exp(-(5e3 / 2e4) ** 2))
        + 0.3 * (1 - np.exp(-(5e3 / 4e4) ** 2))
    )
    assert model.failure_probability(t) == pytest.approx(expected, rel=1e-12)


def test_component_probabilities_weighted_sum_equals_mixture(model):
    t = 1.2e4
    f_i = model.component_failure_probabilities(t)
    np.testing.assert_allclose(f_i, weibull_cdf(t, BETA, np.array(ETAS)))
    assert float(model.weights @ f_i) == pytest.approx(model.failure_probability(t), rel=1e-14)


def test_single_component_reduces_to_weibull():
    m = DiscreteEtaModel(beta=1.5, etas=[3e4], weights=[1.0])
    t = np.array([0.0, 1e3, 3e4, 1e5])
    np.testing.assert_allclose(m.failure_probability(t), weibull_cdf(t, 1.5, 3e4), rtol=1e-14)


def test_identical_etas_collapse_to_weibull():
    m = DiscreteEtaModel(beta=BETA, etas=[2e4, 2e4, 2e4], weights=WEIGHTS)
    assert m.failure_probability(7e3) == pytest.approx(weibull_cdf(7e3, BETA, 2e4), rel=1e-14)


def test_zero_weight_component_has_no_effect():
    with_zero = DiscreteEtaModel(beta=BETA, etas=[1e4, 1e2], weights=[1.0, 0.0])
    assert with_zero.failure_probability(5e3) == pytest.approx(weibull_cdf(5e3, BETA, 1e4), rel=1e-14)


def test_mixture_bounds_and_monotonicity(model):
    t = np.linspace(0.0, 2e5, 500)
    p = model.failure_probability(t)
    assert p.shape == t.shape
    assert p[0] == 0.0 and np.all((p >= 0) & (p <= 1))
    assert np.all(np.diff(p) >= 0)
    np.testing.assert_allclose(p + model.reliability(t), 1.0, atol=1e-15)


def test_scalar_returns_float(model):
    assert isinstance(model.failure_probability(1e4), float)
    assert isinstance(model.reliability(1e4), float)


@pytest.mark.parametrize("t", [-1.0, np.nan, np.inf])
def test_invalid_mission_time_rejected(model, t):
    with pytest.raises(ValueError):
        model.failure_probability(t)
    with pytest.raises(ValueError):
        model.component_failure_probabilities(t)
