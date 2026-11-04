"""Headless tests of the Streamlit dashboard (streamlit.testing AppTest).

Skipped automatically when Streamlit is not installed (e.g. the classical-only environment).
"""

import pathlib

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

from qsr.dashboard import labels, services  # noqa: E402

APP = str(pathlib.Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
needs_qiskit = pytest.mark.skipif(not services.quantum_available(), reason="Qiskit not installed")


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _text(at: AppTest) -> str:
    parts = []
    for group in (at.markdown, at.caption, at.warning, at.info, at.error, at.success, at.title, at.subheader):
        parts.extend(str(el.value) for el in group)
    parts.extend(f"{m.label}: {m.value}" for m in at.metric)
    return "\n".join(parts)


def _click(at: AppTest, label: str) -> None:
    [btn] = [b for b in at.button if b.label == label]
    btn.click().run()
    assert not at.exception, [e.value for e in at.exception]


def _metric(at: AppTest, label: str) -> str:
    [m] = [m for m in at.metric if m.label == label]
    return m.value


@pytest.mark.parametrize("page", ["views/overview.py", "views/classical.py", "views/quantum.py",
                                  "views/comparison.py", "views/limitations.py"])
def test_every_page_renders_without_exceptions(page):
    at = _app()
    at.switch_page(page).run()
    assert not at.exception, [e.value for e in at.exception]


def test_sidebar_shows_raw_and_normalized_weights_and_exact_value():
    at = _app()
    tables = [df.value for df in at.sidebar.dataframe]
    assert any({"Raw", "Normalised"} <= set(t.columns) for t in tables)
    assert _metric(at, labels.EXACT) == "0.079184"


def test_sidebar_shows_raw_weights_beside_normalized_weights(monkeypatch):
    """Start the app with unnormalised raw weights (2, 5, 3): both versions and a notice must be shown.

    AppTest cannot type into st.data_editor, so the sidebar's initial table is patched instead
    (test-only; the dashboard code is unchanged).
    """
    import dataclasses

    monkeypatch.syspath_prepend(str(pathlib.Path(APP).parent))
    import components.model_sidebar as sidebar

    monkeypatch.setattr(sidebar, "DEFAULT_INPUTS", dataclasses.replace(sidebar.DEFAULT_INPUTS, raw_weights=(2.0, 5.0, 3.0)))
    at = _app()
    [weights] = [df.value for df in at.sidebar.dataframe if "Normalised" in df.value.columns]
    assert list(weights["Raw"]) == ["2", "5", "3"]
    assert list(weights["Normalised"]) == ["0.2000", "0.5000", "0.3000"]
    assert any("sum to 10" in i.value for i in at.sidebar.info)
    assert _metric(at, labels.EXACT) == "0.079184"  # same model as (0.2, 0.5, 0.3)


def test_editor_focus_hint_is_shown_under_the_table(monkeypatch):
    monkeypatch.syspath_prepend(str(pathlib.Path(APP).parent))
    from components.model_sidebar import EDITOR_HINT

    for page in ("views/overview.py", "views/classical.py"):
        at = _app()
        at.switch_page(page).run()
        assert EDITOR_HINT in [c.value for c in at.sidebar.caption]


def test_classical_run_matches_documented_numbers_and_goes_stale():
    at = _app()
    at.switch_page("views/classical.py").run()
    _click(at, "Run Monte Carlo")
    assert _metric(at, labels.MC) == "0.080190"  # 100,000 samples, seed 2026 (README / milestone1 demo)
    assert f"{labels.MC_CI}:** [0.078523, 0.081889]" in _text(at)  # full interval, never truncated
    assert not any("out of date" in w.value for w in at.warning)
    at.sidebar.number_input(key="sb_beta").set_value(2.5).run()
    stale = [w.value for w in at.warning if "out of date" in w.value]
    assert stale and "shape β" in stale[0]
    at.select_slider(key="cl_n").set_value(10_000).run()
    stale = [w.value for w in at.warning if "out of date" in w.value]
    assert stale and "Monte Carlo samples" in stale[0]


def test_invalid_model_is_reported_on_pages():
    at = _app()
    at.sidebar.text_input(key="sb_unit").set_value(" ").run()
    assert any("Time unit" in e.value for e in at.sidebar.error)
    at.switch_page("views/classical.py").run()
    msgs = [e.value for e in at.error if "not valid" in e.value]
    assert msgs and ".." not in msgs[0] and "Time unit" in msgs[0]


@needs_qiskit
def test_quantum_page_labels_costs_and_staleness():
    at = _app()
    at.switch_page("views/quantum.py").run()
    at.slider(key="q_len").set_value(3).run()
    at.select_slider(key="q_shots").set_value(30).run()
    _click(at, "Run QAE on the simulator")
    text = _text(at)
    for required in labels.REQUIRED_ON_QUANTUM_RESULTS:
        assert required in text, required
    assert labels.RUNTIME_CAPTION in text and labels.QUERY_CONVENTION in text
    # 30 shots is outside the calibration study: unknown-coverage warning, no measured range quoted.
    assert any("outside the calibration study" in w.value and "30 shots" in w.value for w in at.warning)
    assert labels.QAE_CI_CAPTION not in text
    assert _metric(at, labels.QUERY_COMPLEXITY) == f"{30 * (1 + 3 + 5):,}"
    assert f"{labels.QAE_CI}:** [" in text  # interval shown in full on its own line
    assert any("Circuit check" in s.value for s in at.success)
    at.select_slider(key="q_shots").set_value(100).run()
    assert any("out of date" in w.value and "shots per circuit" in w.value for w in at.warning)


@needs_qiskit
def _quantum_text(at, length, shots):
    at.switch_page("views/quantum.py").run()
    at.slider(key="q_len").set_value(length).run()
    at.select_slider(key="q_shots").set_value(shots).run()
    _click(at, "Run QAE on the simulator")
    return _text(at)


@needs_qiskit
def test_quantum_calibrated_settings_quote_the_study():
    at = _app()
    text = _quantum_text(at, 3, 100)  # m up to 2, 100 shots: inside the tested settings
    assert labels.QAE_CI_CAPTION in text
    assert not any("outside the calibration study" in w.value for w in at.warning)
    assert "Outside the calibration study" not in text


@needs_qiskit
@pytest.mark.parametrize("length, shots, reason", [
    (3, 30, "30 shots per circuit instead of 100"),
    (7, 100, "Grover power up to m = 32, beyond m = 16"),
])
def test_quantum_uncalibrated_settings_warn_and_do_not_quote_coverage(length, shots, reason):
    at = _app()
    at.switch_page("views/quantum.py").run()
    at.slider(key="q_len").set_value(length).run()
    at.select_slider(key="q_shots").set_value(shots).run()
    assert any(reason in c.value and "coverage will be unknown" in c.value for c in at.caption)  # before running
    _click(at, "Run QAE on the simulator")
    text = _text(at)
    warnings = [w.value for w in at.warning if "outside the calibration study" in w.value]
    assert warnings and reason in warnings[0] and "unknown" in warnings[0]
    assert labels.QAE_CI_CAPTION not in text
    assert labels.COVERAGE_RANGE[0] not in text and labels.COVERAGE_RANGE[1] not in text


@needs_qiskit
def test_quantum_timing_separates_compilation():
    at = _app()
    _quantum_text(at, 3, 100)
    r = at.session_state["quantum_run"].result
    assert _metric(at, "Execution total") == f"{r.build_time_s + r.sampling_time_s + r.estimation_time_s:.3f} s"
    assert _metric(at, "Compile (transpile)") == f"{r.transpile_time_s:.3f} s"
    assert labels.COMPILE_NOTE in _text(at)


@needs_qiskit
@pytest.mark.parametrize("shots, calibrated", [(100, True), (30, False)])
def test_comparison_calibration_notice_and_timing_split(shots, calibrated):
    at = _app()
    at.switch_page("views/comparison.py").run()
    at.slider(key="cmp_len").set_value(2).run()
    at.select_slider(key="cmp_shots").set_value(shots).run()
    _click(at, "Run matched comparison")
    text = _text(at)
    uncal = [w.value for w in at.warning if "outside the calibration study" in w.value]
    if calibrated:
        assert labels.QAE_CI_CAPTION in text and not uncal
    else:
        assert uncal and f"{shots} shots" in uncal[0] and labels.QAE_CI_CAPTION not in text
    comp = at.session_state["comparison_run"].comparison
    execution = comp.qae.build_time_s + comp.qae.sampling_time_s + comp.qae.estimation_time_s
    assert _metric(at, labels.QAE_EXECUTION_TIME) == f"{execution:.4f} s"
    assert _metric(at, labels.QAE_COMPILE_TIME) == f"{comp.qae.transpile_time_s:.4f} s"
    assert _metric(at, "QAE execution ÷ MC time") == f"{execution / comp.mc_time_s:,.0f}×"
    assert labels.COMPILE_NOTE in text


def test_comparison_page_matched_and_scaling():
    at = _app()
    at.switch_page("views/comparison.py").run()
    at.slider(key="cmp_len").set_value(2).run()
    at.select_slider(key="cmp_shots").set_value(30).run()
    _click(at, "Run matched comparison")
    df = at.dataframe[0].value
    assert list(df["Quantity"]) == [labels.EXACT, labels.MC, labels.QAE]
    assert df["Model evaluations"].iloc[1] == df["Model evaluations"].iloc[2] == 30 * (1 + 3)
    text = _text(at)
    assert labels.MATCHED_BUDGET in text and labels.RUNTIME_CAPTION in text and labels.NO_SPEEDUP in text

    at.slider(key="sc_len").set_value(2).run()
    at.slider(key="sc_seeds").set_value(3).run()
    at.select_slider(key="sc_shots").set_value(10).run()
    _click(at, "Run scaling study")
    assert labels.CRB_NOTE in _text(at)
    assert len(at.get("plotly_chart")) == 1  # the error-vs-oracle-queries figure


@needs_qiskit
def test_scaling_study_runs_in_two_separate_sessions():
    """Regression: a cached scaling run replayed its progress-bar calls into another session and crashed."""
    for _ in range(2):  # two independent sessions in the same process, identical settings
        at = _app()
        at.switch_page("views/comparison.py").run()
        at.slider(key="sc_len").set_value(2).run()
        at.slider(key="sc_seeds").set_value(3).run()
        at.select_slider(key="sc_shots").set_value(10).run()
        _click(at, "Run scaling study")
        assert len(at.get("plotly_chart")) == 1


def test_quantum_pages_degrade_gracefully_without_qiskit(monkeypatch):
    monkeypatch.setattr(services, "quantum_available", lambda: False)
    at = _app()
    assert any(labels.QISKIT_MISSING in i.value for i in at.info)
    for page in ("views/quantum.py", "views/comparison.py"):
        at.switch_page(page).run()
        assert not at.exception
        assert any(labels.QISKIT_MISSING in w.value for w in at.warning)
        assert not [b for b in at.button if b.label.startswith("Run")]
    at.switch_page("views/classical.py").run()
    _click(at, "Run Monte Carlo")
    assert _metric(at, labels.MC) == "0.080190"


def test_limitations_page_states_key_caveats():
    at = _app()
    at.switch_page("views/limitations.py").run()
    text = _text(at)
    for s in (labels.NO_SPEEDUP, labels.QAE_CI_CAPTION, labels.RUNTIME_CAPTION, labels.QUERY_CONVENTION,
              labels.WORST_CELL_COVERAGE, "noiseless", "computed classically"):
        assert s in text, s
