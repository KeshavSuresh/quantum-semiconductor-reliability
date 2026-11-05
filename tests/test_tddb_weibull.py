import numpy as np
import pytest
from scipy import stats

from qsr.models.tddb_weibull import weibull_cdf, weibull_inverse_cdf, weibull_reliability

BETA = 2.0
ETA = 1.0e4  # hours


def test_cdf_at_eta_is_one_minus_inverse_e():
    assert weibull_cdf(ETA, BETA, ETA) == pytest.approx(1.0 - np.exp(-1.0), rel=1e-12)


def test_cdf_at_zero_is_zero():
    assert weibull_cdf(0.0, BETA, ETA) == 0.0


def test_cdf_plus_reliability_is_one():
    t = np.linspace(0.0, 5 * ETA, 101)
    np.testing.assert_allclose(weibull_cdf(t, BETA, ETA) + weibull_reliability(t, BETA, ETA), 1.0, atol=1e-15)


@pytest.mark.parametrize("beta", [0.7, 1.0, 1.5, 3.0])
def test_cdf_and_reliability_match_scipy(beta):
    t = np.linspace(0.0, 4 * ETA, 50)
    np.testing.assert_allclose(weibull_cdf(t, beta, ETA), stats.weibull_min.cdf(t, beta, scale=ETA), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(weibull_reliability(t, beta, ETA), stats.weibull_min.sf(t, beta, scale=ETA), rtol=1e-12)


def test_beta_one_is_exponential():
    t = np.array([10.0, 1e3, 3e4])
    np.testing.assert_allclose(weibull_cdf(t, 1.0, ETA), 1.0 - np.exp(-t / ETA), rtol=1e-12)


def test_cdf_monotonic_increasing():
    f = weibull_cdf(np.linspace(0.0, 10 * ETA, 1000), BETA, ETA)
    assert np.all(np.diff(f) >= 0)
    assert 0.0 <= f.min() and f.max() <= 1.0


def test_small_probability_precision():
    # (t/eta)^beta = 1e-20 -> F ~ 1e-20; naive 1 - exp(-x) would return 0.
    assert weibull_cdf(1e-10 * ETA, BETA, ETA) == pytest.approx(1e-20, rel=1e-9)


def test_broadcast_over_eta():
    etas = np.array([1e3, 1e4, 1e5])
    out = weibull_cdf(1e4, BETA, etas)
    assert out.shape == (3,)
    assert out[0] > out[1] > out[2]


def test_scalar_input_returns_float():
    assert isinstance(weibull_cdf(1.0, BETA, ETA), float)
    assert isinstance(weibull_inverse_cdf(0.5, BETA, ETA), float)


def test_inverse_cdf_round_trip():
    u = np.linspace(0.0, 0.999, 200)
    t = weibull_inverse_cdf(u, BETA, ETA)
    np.testing.assert_allclose(weibull_cdf(t, BETA, ETA), u, atol=1e-12)


def test_inverse_cdf_median():
    expected = ETA * np.log(2.0) ** (1.0 / BETA)
    assert weibull_inverse_cdf(0.5, BETA, ETA) == pytest.approx(expected, rel=1e-12)


def test_inverse_cdf_sampling_matches_weibull_distribution():
    rng = np.random.default_rng(12345)
    samples = weibull_inverse_cdf(rng.random(20_000), BETA, ETA)
    result = stats.kstest(samples, stats.weibull_min(BETA, scale=ETA).cdf)
    assert result.pvalue > 0.01


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(t=1.0, beta=0.0, eta=ETA),
        dict(t=1.0, beta=-1.0, eta=ETA),
        dict(t=1.0, beta=np.nan, eta=ETA),
        dict(t=1.0, beta=BETA, eta=0.0),
        dict(t=1.0, beta=BETA, eta=np.inf),
        dict(t=-1.0, beta=BETA, eta=ETA),
        dict(t=np.nan, beta=BETA, eta=ETA),
    ],
)
def test_cdf_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        weibull_cdf(**kwargs)
    with pytest.raises(ValueError):
        weibull_reliability(**kwargs)


@pytest.mark.parametrize("u", [1.0, -0.01, 1.5, np.nan])
def test_inverse_cdf_rejects_u_outside_unit_interval(u):
    with pytest.raises(ValueError):
        weibull_inverse_cdf(u, BETA, ETA)
