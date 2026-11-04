import numpy as np
import pytest

from qsr.visualization import DARK, plot_error_vs_queries


def _rows():
    calls = [100, 400, 900, 1800]
    return [
        {"oracle_calls": n, "qae_rmse": 3.0 / n, "mc_rmse": 0.3 / np.sqrt(n),
         "qae_theory_se": 2.7 / n, "mc_theory_se": 0.27 / np.sqrt(n)}
        for n in calls
    ]


def test_figure_structure_and_data():
    rows = _rows()
    fig = plot_error_vs_queries(rows)
    names = [tr.name for tr in fig.data]
    assert names == [
        "QAE (MLAE): measured RMSE", "QAE (MLAE): theory",
        "Classical Monte Carlo: measured RMSE", "Classical Monte Carlo: theory",
    ]
    np.testing.assert_array_equal(fig.data[0].x, [r["oracle_calls"] for r in rows])
    np.testing.assert_allclose(fig.data[0].y, [r["qae_rmse"] for r in rows])
    np.testing.assert_allclose(fig.data[3].y, [r["mc_theory_se"] for r in rows])
    assert fig.layout.xaxis.type == "log" and fig.layout.yaxis.type == "log"
    assert "not runtime" in fig.layout.title.text
    assert {a.text for a in fig.layout.annotations} == {"QAE", "MC"}


def test_zero_rmse_becomes_gap_on_log_axis():
    rows = _rows()
    rows[1]["qae_rmse"] = 0.0
    fig = plot_error_vs_queries(rows)
    assert np.isnan(fig.data[0].y[1])


def test_dark_theme():
    assert plot_error_vs_queries(_rows(), theme=DARK).layout.paper_bgcolor == DARK.surface


@pytest.mark.parametrize(
    "mutate",
    [
        lambda rs: rs.clear(),
        lambda rs: rs[0].pop("qae_rmse"),
        lambda rs: rs.reverse(),
        lambda rs: rs[0].update(oracle_calls=0),
        lambda rs: rs[0].update(mc_rmse=-1.0),
        lambda rs: rs[0].update(qae_theory_se=float("nan")),
    ],
)
def test_invalid_rows_rejected(mutate):
    rows = _rows()
    mutate(rows)
    with pytest.raises(ValueError):
        plot_error_vs_queries(rows)


def test_invalid_theme_rejected():
    with pytest.raises(TypeError):
        plot_error_vs_queries(_rows(), theme="dark")
