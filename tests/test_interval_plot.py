import numpy as np
import pytest

pytest.importorskip("plotly")  # optional ``viz`` extra: skip when it is not installed

from qsr.visualization import LIGHT, plot_estimates_vs_reference  # noqa: E402

EST = [
    {"kind": "classical", "label": "Classical MC", "estimate": 0.077206, "low": 0.071098, "high": 0.083791},
    {"kind": "quantum", "label": "Quantum MLAE", "estimate": 0.078668, "low": 0.076968, "high": 0.080449, "detail": "x"},
]


def test_points_intervals_and_reference_line():
    fig = plot_estimates_vs_reference(0.079184, EST)
    mc, qae = fig.data
    assert mc.x[0] == 0.077206 and qae.x[0] == 0.078668
    # Asymmetric error bars reconstruct the given interval exactly.
    assert mc.x[0] + mc.error_x.array[0] == pytest.approx(0.083791) and mc.x[0] - mc.error_x.arrayminus[0] == pytest.approx(0.071098)
    assert mc.marker.color == LIGHT.classical and qae.marker.color == LIGHT.quantum
    [line] = [s for s in fig.layout.shapes if s.type == "line"]
    assert line.x0 == 0.079184 and line.line.dash == "dash"


def test_axis_range_contains_every_interval_and_the_reference():
    fig = plot_estimates_vs_reference(0.09, EST)  # reference outside both intervals
    lo, hi = fig.layout.xaxis.range
    assert lo < 0.071098 and hi > 0.09
    assert fig.layout.xaxis.type in (None, "linear", "-")  # linear: no distortion


@pytest.mark.parametrize("bad", [
    [],
    [{"kind": "other", "label": "x", "estimate": 0.1, "low": 0.0, "high": 0.2}],
    [{"kind": "classical", "label": "x", "estimate": 0.3, "low": 0.0, "high": 0.2}],
    [{"kind": "classical", "label": "x", "estimate": float("nan"), "low": 0.0, "high": 0.2}],
    [{"kind": "classical", "label": "x", "estimate": 0.1, "low": 0.0}],
])
def test_invalid_estimates_rejected(bad):
    with pytest.raises(ValueError):
        plot_estimates_vs_reference(0.1, bad)


def test_invalid_reference_and_theme():
    with pytest.raises(ValueError):
        plot_estimates_vs_reference(np.inf, EST)
    with pytest.raises(TypeError):
        plot_estimates_vs_reference(0.1, EST, theme="dark")


def test_row_labels_are_annotations_not_clipped_tick_labels():
    fig = plot_estimates_vs_reference(0.08, [
        {"kind": "classical", "label": "Monte Carlo · 6,800 samples", "estimate": 0.077, "low": 0.071, "high": 0.084},
        {"kind": "quantum", "label": "QAE (simulated) · 6,800 queries", "estimate": 0.0787, "low": 0.0770, "high": 0.0804},
    ])
    assert fig.layout.yaxis.showticklabels is False
    texts = {a.text for a in fig.layout.annotations}
    assert {"Monte Carlo · 6,800 samples", "QAE (simulated) · 6,800 queries"} <= texts
