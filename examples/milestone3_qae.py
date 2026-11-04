"""Milestone 3 demonstration: Quantum Amplitude Estimation vs exact and classical Monte Carlo.

Time unit: HOURS for every time quantity (etas and mission time). This is the
same model as the Milestone 1 and 2 examples.

Requires the optional quantum extra:  pip install -e ".[quantum,viz]"

Run:
    python examples/milestone3_qae.py              # full run (about 30-60 s on a laptop)
    python examples/milestone3_qae.py --quick      # fewer seeds, shorter schedules
    python examples/milestone3_qae.py --dark       # dark-theme figure

Writes build/plots/qae_error_vs_queries_<theme>.html (build/ is git-ignored).

IMPORTANT: "oracle calls" counts model evaluations (query complexity). The
wall-clock times are for a classical statevector SIMULATOR and say nothing
about quantum hardware. No practical quantum speed-up is shown or claimed.
"""

from __future__ import annotations

import argparse
import platform
from pathlib import Path

import qiskit

from qsr.models.uncertainty import DiscreteEtaModel
from qsr.quantum import (
    build_state_preparation,
    compare_with_monte_carlo,
    error_scaling_study,
    exact_ancilla_probability,
    exponential_schedule,
)
from qsr.visualization import DARK, LIGHT, plot_error_vs_queries

BETA = 2.0                          # Weibull shape (dimensionless)
ETAS_HOURS = [1.0e4, 2.0e4, 4.0e4]  # candidate characteristic lives eta_i [h]
WEIGHTS = [0.2, 0.5, 0.3]           # P(eta = eta_i)
MISSION_TIME_HOURS = 5.0e3          # mission time t [h]
SEED = 2026
SHOTS = 100


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=Path("build/plots"), help="output folder (default: build/plots)")
    parser.add_argument("--dark", action="store_true", help="use the dark theme for the figure")
    parser.add_argument("--quick", action="store_true", help="10 seeds and schedules up to length 5")
    parser.add_argument("--cdn", action="store_true", help="load plotly.js from a CDN (smaller file, needs internet)")
    args = parser.parse_args()

    model = DiscreteEtaModel(beta=BETA, etas=ETAS_HOURS, weights=WEIGHTS)
    exact = model.failure_probability(MISSION_TIME_HOURS)
    print(f"Environment: Python {platform.python_version()}, Qiskit {qiskit.__version__}, "
          f"StatevectorSampler (classical simulation, noiseless)")
    print(f"Model (hours): beta = {BETA}, etas = {ETAS_HOURS}, weights = {WEIGHTS}, t = {MISSION_TIME_HOURS:g} h")

    # 1. State preparation correctness.
    a_circuit = build_state_preparation(model, MISSION_TIME_HOURS)
    sv_p = exact_ancilla_probability(a_circuit)
    print("\n1. State preparation A (statevector, no sampling)")
    print(f"   qubits = {a_circuit.num_qubits} (index {a_circuit.num_qubits - 1} + 1 failure ancilla)")
    print(f"   P(ancilla = 1) = {sv_p:.15f}")
    print(f"   exact p        = {exact:.15f}")
    print(f"   |difference|   = {abs(sv_p - exact):.2e}")

    # 2. One budget-matched comparison.
    schedule = exponential_schedule(6)
    c = compare_with_monte_carlo(model, MISSION_TIME_HOURS, seed=SEED, evaluation_schedule=schedule, shots=SHOTS)
    q = c.qae
    print(f"\n2. Single run, seed = {SEED}, schedule m = {list(q.evaluation_schedule)}, shots/circuit = {q.shots}")
    print(f"   {'method':<22} {'estimate':>10} {'95% CI':>23} {'|error|':>10} {'oracle calls':>13} {'time [s]':>10}")
    print(f"   {'exact':<22} {exact:>10.6f}")
    print(f"   {'QAE (MLAE, simulated)':<22} {q.estimate:>10.6f} [{q.ci_low:.6f}, {q.ci_high:.6f}] "
          f"{c.qae_abs_error:>10.6f} {q.state_preparation_calls:>13,} {q.total_time_s:>10.4f}")
    print(f"   {'classical MC':<22} {c.mc.estimate:>10.6f} [{c.mc.ci_low:.6f}, {c.mc.ci_high:.6f}] "
          f"{c.mc_abs_error:>10.6f} {c.mc.n_samples:>13,} {c.mc_time_s:>10.4f}")
    print(f"   QAE Grover applications = {q.grover_applications:,}; "
          f"circuit depth per power (basis rz,sx,x,cx, opt level 1) = {list(q.circuit_depths)}")
    print(f"   CX count per power = {list(q.two_qubit_gate_counts)}")
    print(f"   QAE time split [s]: build {q.build_time_s:.4f}, transpile {q.transpile_time_s:.4f}, "
          f"simulate {q.sampling_time_s:.4f}, estimate {q.estimation_time_s:.4f}")

    # 3. Error scaling over fixed seeds.
    lengths = (1, 2, 3, 4, 5) if args.quick else (1, 2, 3, 4, 5, 6, 7)
    seeds = range(10) if args.quick else range(30)
    rows = error_scaling_study(
        model, MISSION_TIME_HOURS, schedule_lengths=lengths, shots=SHOTS, seeds=seeds
    )
    print(f"\n3. Error vs oracle calls, RMSE over {len(seeds)} fixed seeds, {SHOTS} shots per circuit")
    print(f"   {'m_max':>5} {'calls':>7} {'QAE RMSE':>10} {'QAE theory':>11} {'MC RMSE':>10} {'MC theory':>10} "
          f"{'QAE cov':>8} {'MC cov':>7} {'QAE s/run':>10} {'MC s/run':>9}")
    for r in rows:
        print(f"   {r['max_grover_power']:>5} {r['oracle_calls']:>7,} {r['qae_rmse']:>10.6f} {r['qae_theory_se']:>11.6f} "
              f"{r['mc_rmse']:>10.6f} {r['mc_theory_se']:>10.6f} {r['qae_ci_coverage']:>8.2f} {r['mc_ci_coverage']:>7.2f} "
              f"{r['qae_mean_time_s']:>10.4f} {r['mc_mean_time_s']:>9.5f}")
    print("   theory: QAE ~ 1/calls (Fisher bound), MC ~ 1/sqrt(calls). Times are simulator wall-clock, not hardware.")

    theme = DARK if args.dark else LIGHT
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"qae_error_vs_queries_{theme.name}.html"
    plot_error_vs_queries(rows, theme=theme).write_html(path, include_plotlyjs="cdn" if args.cdn else True)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
