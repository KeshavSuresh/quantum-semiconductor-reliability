"""Readout tables (pure formatting; no Streamlit needed)."""

from types import SimpleNamespace

import pytest

from qsr.dashboard import labels, readout
from qsr.dashboard.config import DEFAULT_INPUTS, build_model
from qsr.simulation.monte_carlo import estimate_failure_probability

EXACT = 0.07918437


def _rows(md: str) -> list[list[str]]:
    lines = [ln for ln in md.splitlines() if ln.startswith("|")]
    return [[c.strip() for c in ln.strip("|").split("|")] for ln in lines[2:]]  # skip header and rule


def test_readout_rows_carry_badges_values_and_budgets():
    model = build_model(DEFAULT_INPUTS).model
    mc = estimate_failure_probability(model, 5_000.0, 100_000, seed=2026)
    qae = SimpleNamespace(estimate=0.078668, ci_low=0.076968, ci_high=0.080449, state_preparation_calls=6_800)
    md = readout.readout_table([readout.reference_row(EXACT), readout.classical_row(mc, EXACT),
                                readout.quantum_row(qae, EXACT)])
    ref, cls, qnt = _rows(md)
    assert ref[0] == f"{labels.BADGE_REFERENCE} {labels.EXACT}" and ref[2:] == ["—"] * 4
    assert cls[0].startswith(labels.BADGE_CLASSICAL) and cls[1] == "**0.080190**"
    assert cls[2] == "[0.078523, 0.081889]" and cls[5] == "100,000 samples"
    assert qnt[0].startswith(labels.BADGE_QUANTUM) and qnt[5] == "6,800 oracle queries"
    # Width and error are simple differences of the values shown, formatted the same way.
    assert qnt[3] == "0.003481" and qnt[4] == f"{abs(0.078668 - EXACT):.6f}"


def test_badges_use_one_colour_per_kind():
    assert labels.BADGE_REFERENCE.startswith(":gray-badge[")
    assert labels.BADGE_CLASSICAL.startswith(":orange-badge[")
    assert labels.BADGE_QUANTUM.startswith(":blue-badge[") and "simulated" in labels.BADGE_QUANTUM


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown readout kind"):
        readout.readout_table([readout.ReadoutRow("other", "x", 0.1)])


def test_model_line_and_reliability_table():
    built = build_model(DEFAULT_INPUTS)
    line = readout.model_line(DEFAULT_INPUTS, built)
    assert line == ("Weibull β = 2 · η = 10,000 / 20,000 / 40,000 h · w = 0.20 / 0.50 / 0.30 · "
                    "mission time t = 5,000 h")
    rel = _rows(readout.reliability_table(EXACT, 1 - EXACT, 5_000.0, "h"))
    assert rel[0][1] == "**0.079184**" and rel[1][1] == "**0.920816**"


def test_facts_table_alignment_escaping_and_validation():
    md = readout.facts_table([("a | b", "1"), ("c", "2")])
    lines = md.splitlines()
    assert lines[0] == "| Quantity | Value |" and lines[1] == "|:--|--:|"
    assert lines[2] == "| a \\| b | 1 |"  # a pipe in a value cannot break the table
    with pytest.raises(ValueError):
        readout.facts_table([])
    with pytest.raises(ValueError):
        readout.facts_table([("only one cell",)])


def test_seconds_never_wraps_between_number_and_unit():
    assert readout.seconds(0.00071, 4) == "0.0007 s"
    assert readout.seconds(1.23456) == "1.235 s"
