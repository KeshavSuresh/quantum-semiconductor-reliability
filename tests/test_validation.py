import numpy as np
import pytest

from qsr.validation import (
    check_nonnegative_finite,
    check_open_unit_interval,
    check_positive_finite,
    check_positive_int,
    check_unit_interval_half_open,
)


def test_positive_finite_accepts_scalars_and_arrays():
    assert check_positive_finite("x", 2.5) == 2.5
    np.testing.assert_array_equal(check_positive_finite("x", [1, 2]), [1.0, 2.0])


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf, [1.0, 0.0]])
def test_positive_finite_rejects(bad):
    with pytest.raises(ValueError):
        check_positive_finite("x", bad)


@pytest.mark.parametrize("bad", [True, "abc", None])
def test_non_numeric_rejected(bad):
    with pytest.raises(TypeError):
        check_positive_finite("x", bad)


def test_empty_rejected():
    with pytest.raises(ValueError):
        check_positive_finite("x", [])


def test_nonnegative_allows_zero():
    assert check_nonnegative_finite("t", 0.0) == 0.0
    with pytest.raises(ValueError):
        check_nonnegative_finite("t", -1e-12)


def test_unit_interval_half_open():
    check_unit_interval_half_open("u", [0.0, 0.5, 0.999])
    for bad in (1.0, -0.1):
        with pytest.raises(ValueError):
            check_unit_interval_half_open("u", bad)


def test_open_unit_interval():
    assert check_open_unit_interval("c", 0.95) == 0.95
    for bad in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError):
            check_open_unit_interval("c", bad)
    with pytest.raises(ValueError):
        check_open_unit_interval("c", [0.9, 0.95])


def test_positive_int():
    assert check_positive_int("n", 10) == 10
    assert check_positive_int("n", np.int64(3)) == 3
    for bad in (1.0, True, "5"):
        with pytest.raises(TypeError):
            check_positive_int("n", bad)
    for bad in (0, -5):
        with pytest.raises(ValueError):
            check_positive_int("n", bad)
