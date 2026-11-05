"""Milestone 2 demonstration: write the three Plotly figures as standalone HTML files.

Time unit: HOURS for every time quantity (etas and mission time). This is the
same model as examples/milestone1_demo.py.

Run:
    python examples/milestone2_plots.py                 # light theme, writes build/plots/
    python examples/milestone2_plots.py --dark          # dark theme
    python examples/milestone2_plots.py --out-dir PATH  # choose the output folder

build/ is already git-ignored, so generated files are never committed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from qsr.models.uncertainty import DiscreteEtaModel
from qsr.visualization import (
    DARK,
    LIGHT,
    mc_results_table,
    plot_component_curves,
    plot_failure_and_reliability,
    plot_mc_vs_exact,
    run_mc_convergence,
)

BETA = 2.0                          # Weibull shape (dimensionless)
ETAS_HOURS = [1.0e4, 2.0e4, 4.0e4]  # candidate characteristic lives eta_i [h]
WEIGHTS = [0.2, 0.5, 0.3]           # P(eta = eta_i)
MISSION_TIME_HOURS = 5.0e3          # mission time t [h]
SEED = 2026
SAMPLE_SIZES = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]
TIME_UNIT = "h"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=Path("build/plots"), help="output folder (default: build/plots)")
    parser.add_argument("--dark", action="store_true", help="use the dark theme")
    parser.add_argument("--cdn", action="store_true", help="load plotly.js from a CDN (smaller files, needs internet)")
    args = parser.parse_args()

    theme = DARK if args.dark else LIGHT
    model = DiscreteEtaModel(beta=BETA, etas=ETAS_HOURS, weights=WEIGHTS)
    results = run_mc_convergence(model, MISSION_TIME_HOURS, SAMPLE_SIZES, seed=SEED)

    figures = {
        "failure_reliability": plot_failure_and_reliability(
            model, time_unit=TIME_UNIT, mission_time=MISSION_TIME_HOURS, theme=theme
        ),
        "component_curves": plot_component_curves(
            model, time_unit=TIME_UNIT, mission_time=MISSION_TIME_HOURS, theme=theme
        ),
        "mc_vs_exact": plot_mc_vs_exact(model, results, time_unit=TIME_UNIT, theme=theme),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    include_js = "cdn" if args.cdn else True
    for name, fig in figures.items():
        path = args.out_dir / f"{name}_{theme.name}.html"
        fig.write_html(path, include_plotlyjs=include_js, full_html=True)
        print(f"wrote {path}")

    print(f"\nMonte Carlo vs exact at t = {MISSION_TIME_HOURS:g} {TIME_UNIT} (seed = {SEED}, 95% Wilson CI)")
    print(f"{'n':>10} {'p_hat':>10} {'CI low':>10} {'CI high':>10} {'|err|':>10} {'exact in CI':>12}")
    for row in mc_results_table(model, results):
        print(
            f"{row['n_samples']:>10d} {row['estimate']:>10.6f} {row['ci_low']:>10.6f} "
            f"{row['ci_high']:>10.6f} {row['abs_error']:>10.6f} {str(row['exact_in_ci']):>12}"
        )
    print(f"exact p = {model.failure_probability(MISSION_TIME_HOURS):.6f}")


if __name__ == "__main__":
    main()
