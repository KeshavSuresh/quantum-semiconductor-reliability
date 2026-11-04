import json
import subprocess
import sys

import numpy as np
import pytest

# Plotly is the optional ``viz`` extra: skip (not fail) when it is not installed.
go = pytest.importorskip("plotly.graph_objects")

from qsr.models.tddb_weibull import weibull_cdf
from qsr.models.uncertainty import DiscreteEtaModel
from qsr.simulation.monte_carlo import estimate_failure_probability
from qsr.visualization import (
    DARK,
    LIGHT,
    curve_data,
    default_time_grid,
    mc_results_table,
    plot_component_curves,
    plot_failure_and_reliability,
    plot_mc_vs_exact,
    run_mc_convergence,
)
from qsr.visualization.theme import MAX_CATEGORICAL

# Time unit for all tests: hours.
MISSION_TIME = 5.0e3
SIZES = [1_000, 10_000, 100_000]
SEED = 2026


@pytest.fixture
def model():
    return DiscreteEtaModel(beta=2.0, etas=[1.0e4, 2.0e4, 4.0e4], weights=[0.2, 0.5, 0.3])


@pytest.fixture
def results(model):
    return run_mc_convergence(model, MISSION_TIME, SIZES, seed=SEED)


def _traces_by_name(fig):
    return {tr.name: tr for tr in fig.data}


# --- data functions --------------------------------------------------------

def test_default_time_grid(model):
    t = default_time_grid(model)
    assert t[0] == 0.0 and t[-1] == pytest.approx(3 * 4.0e4) and t.size == 400
    assert np.all(np.diff(t) > 0)
    t2 = default_time_grid(model, t_max=1e4, n_points=11)
    np.testing.assert_allclose(t2, np.linspace(0, 1e4, 11))


@pytest.mark.parametrize("kwargs", [dict(t_max=0.0), dict(t_max=-1.0), dict(t_max=np.nan), dict(n_points=1)])
def test_default_time_grid_rejects_invalid(model, kwargs):
    with pytest.raises(ValueError):
        default_time_grid(model, **kwargs)


def test_curve_data_uses_model_as_source_of_truth(model):
    t = np.linspace(0, 1e5, 50)
    d = curve_data(model, t)
    np.testing.assert_array_equal(d["time"], t)
    np.testing.assert_array_equal(d["mixture_failure"], model.failure_probability(t))
    np.testing.assert_array_equal(d["mixture_reliability"], model.reliability(t))
    assert d["component_failure"].shape == (3, 50)
    for i, eta in enumerate(model.etas):
        np.testing.assert_allclose(d["component_failure"][i], weibull_cdf(t, model.beta, eta), rtol=1e-15)
    # Weighted sum of plotted component curves equals the plotted mixture curve.
    np.testing.assert_allclose(model.weights @ d["component_failure"], d["mixture_failure"], atol=1e-15)


@pytest.mark.parametrize(
    "times", [[0.0], [1.0, 1.0, 2.0], [3.0, 2.0, 1.0], [-1.0, 1.0], [0.0, np.nan], [[0.0, 1.0]]]
)
def test_curve_data_rejects_bad_times(model, times):
    with pytest.raises(ValueError):
        curve_data(model, times)


def test_run_mc_convergence_matches_direct_calls(model, results):
    assert [r.n_samples for r in results] == SIZES
    for r in results:
        assert r == estimate_failure_probability(model, MISSION_TIME, r.n_samples, seed=SEED)


@pytest.mark.parametrize("sizes", [[], [100, 100], [1000, 100], [0, 10]])
def test_run_mc_convergence_rejects_bad_sizes(model, sizes):
    with pytest.raises(ValueError):
        run_mc_convergence(model, MISSION_TIME, sizes, seed=SEED)


def test_run_mc_convergence_rejects_float_size(model):
    with pytest.raises(TypeError):
        run_mc_convergence(model, MISSION_TIME, [100.0], seed=SEED)


def test_mc_results_table(model, results):
    exact = model.failure_probability(MISSION_TIME)
    rows = mc_results_table(model, results)
    assert len(rows) == len(SIZES)
    for row, r in zip(rows, results):
        assert row["exact"] == exact
        assert row["estimate"] == r.estimate and row["ci_low"] == r.ci_low and row["ci_high"] == r.ci_high
        assert row["abs_error"] == pytest.approx(abs(r.estimate - exact))
        assert row["exact_in_ci"] == (r.ci_low <= exact <= r.ci_high)


def test_results_must_share_mission_time_and_confidence(model):
    a = estimate_failure_probability(model, 5e3, 100, seed=0)
    b = estimate_failure_probability(model, 6e3, 100, seed=0)
    c = estimate_failure_probability(model, 5e3, 200, seed=0, confidence=0.9)
    with pytest.raises(ValueError):
        mc_results_table(model, [a, b])
    with pytest.raises(ValueError):
        plot_mc_vs_exact(model, [a, c])
    with pytest.raises(ValueError):
        plot_mc_vs_exact(model, [])
    with pytest.raises(TypeError):
        plot_mc_vs_exact(model, [{"estimate": 0.1}])


# --- figures ---------------------------------------------------------------

def test_failure_and_reliability_figure(model):
    t = np.linspace(0, 1e5, 100)
    fig = plot_failure_and_reliability(model, t, mission_time=MISSION_TIME)
    assert isinstance(fig, go.Figure)
    traces = _traces_by_name(fig)
    assert set(traces) == {"Failure probability F(t)", "Reliability R(t)"}
    np.testing.assert_array_equal(traces["Failure probability F(t)"].y, model.failure_probability(t))
    np.testing.assert_array_equal(traces["Reliability R(t)"].y, model.reliability(t))
    np.testing.assert_allclose(traces["Failure probability F(t)"].y + traces["Reliability R(t)"].y, 1.0, atol=1e-15)
    assert fig.layout.xaxis.title.text == "Time (h)"
    assert any("mission time 5000 h" in (a.text or "") for a in fig.layout.annotations)
    assert {a.text for a in fig.layout.annotations} >= {"F(t)", "R(t)"}  # direct labels


def test_component_figure_traces_and_data(model):
    t = np.linspace(0, 1e5, 80)
    fig = plot_component_curves(model, t, time_unit="hours")
    assert len(fig.data) == model.n_components + 1
    for i, eta in enumerate(model.etas):
        np.testing.assert_allclose(fig.data[i].y, weibull_cdf(t, model.beta, eta), rtol=1e-15)
        assert f"{eta:g} hours" in fig.data[i].name
    mixture = fig.data[-1]
    assert mixture.name.startswith("Mixture")
    np.testing.assert_array_equal(mixture.y, model.failure_probability(t))
    assert mixture.line.width > fig.data[0].line.width
    colors = [tr.line.color for tr in fig.data[:-1]]
    assert colors == list(LIGHT.series[:3])  # fixed categorical order
    assert {a.text for a in fig.layout.annotations} == {"η₁", "η₂", "η₃", "Mixture"}
    assert fig.layout.xaxis.title.text == "Time (hours)"


def test_component_colours_follow_entity_not_rank():
    # Same eta in slot 1 keeps colour 1 regardless of what else is in the model.
    a = plot_component_curves(DiscreteEtaModel(beta=2, etas=[1e4, 2e4], weights=[0.5, 0.5]))
    b = plot_component_curves(DiscreteEtaModel(beta=2, etas=[1e4, 2e4, 3e4], weights=[0.4, 0.3, 0.3]))
    assert a.data[0].line.color == b.data[0].line.color == LIGHT.series[0]


def test_many_components_grouped_without_generated_hues():
    k = MAX_CATEGORICAL + 4
    m = DiscreteEtaModel.from_unnormalized(1.5, np.linspace(1e3, 1e4, k), np.ones(k))
    fig = plot_component_curves(m)
    comps = fig.data[:-1]
    assert len(comps) == k
    assert {tr.line.color for tr in comps} == {LIGHT.ink_muted}
    assert sum(bool(tr.showlegend) for tr in comps) == 1
    assert not fig.layout.annotations  # no direct labels for a crowd


def test_mc_vs_exact_figure(model, results):
    fig = plot_mc_vs_exact(model, results)
    exact_trace, mc_trace = fig.data
    exact = model.failure_probability(MISSION_TIME)
    assert exact_trace.name.startswith("Exact p")
    assert set(exact_trace.y) == {exact}
    np.testing.assert_array_equal(mc_trace.x, SIZES)
    np.testing.assert_array_equal(mc_trace.y, [r.estimate for r in results])
    est = np.array([r.estimate for r in results])
    np.testing.assert_allclose(est + np.asarray(mc_trace.error_y.array), [r.ci_high for r in results], atol=1e-15)
    np.testing.assert_allclose(est - np.asarray(mc_trace.error_y.arrayminus), [r.ci_low for r in results], atol=1e-15)
    assert mc_trace.error_y.symmetric is False
    assert fig.layout.xaxis.type == "log"
    assert "95%" in mc_trace.name
    assert "t = 5000 h" in fig.layout.title.text


def test_dark_theme_applied(model, results):
    for fig in (
        plot_failure_and_reliability(model, theme=DARK),
        plot_component_curves(model, theme=DARK),
        plot_mc_vs_exact(model, results, theme=DARK),
    ):
        assert fig.layout.paper_bgcolor == DARK.surface == fig.layout.plot_bgcolor
    assert plot_component_curves(model, theme=DARK).data[-1].line.color == DARK.ink_primary


def test_figures_are_json_serializable(model, results):
    for fig in (plot_failure_and_reliability(model), plot_component_curves(model), plot_mc_vs_exact(model, results)):
        json.loads(fig.to_json())


def test_plotting_does_not_mutate_model(model, results):
    before = (model.beta, model.etas.copy(), model.weights.copy())
    plot_failure_and_reliability(model)
    plot_component_curves(model)
    plot_mc_vs_exact(model, results)
    assert model.beta == before[0]
    np.testing.assert_array_equal(model.etas, before[1])
    np.testing.assert_array_equal(model.weights, before[2])


@pytest.mark.parametrize("builder", [plot_failure_and_reliability, plot_component_curves])
def test_builders_validate_inputs(model, builder):
    with pytest.raises(TypeError):
        builder({"beta": 2})
    with pytest.raises(ValueError):
        builder(model, time_unit="")
    with pytest.raises(ValueError):
        builder(model, mission_time=-1.0)
    with pytest.raises(TypeError):
        builder(model, theme="dark")


def test_no_qiskit_or_streamlit_imported():
    # Fresh interpreter: other test modules (e.g. quantum tests) may already have imported Qiskit here.
    code = (
        "import sys, qsr.models.uncertainty, qsr.simulation.monte_carlo, qsr.visualization; "
        "assert 'qiskit' not in sys.modules and 'streamlit' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
