"""Milestone 1 demonstration: exact mixture probability vs classical Monte Carlo.

Time unit: HOURS for every time quantity (etas and mission time).

Run:  python examples/milestone1_demo.py
"""

from __future__ import annotations

from qsr.models.uncertainty import DiscreteEtaModel
from qsr.simulation.monte_carlo import estimate_failure_probability

BETA = 2.0                          # Weibull shape (dimensionless)
ETAS_HOURS = [1.0e4, 2.0e4, 4.0e4]  # candidate characteristic lives eta_i [h]
WEIGHTS = [0.2, 0.5, 0.3]           # P(eta = eta_i)
MISSION_TIME_HOURS = 5.0e3          # mission time t [h]
SEED = 2026
SAMPLE_SIZES = [1_000, 10_000, 100_000, 1_000_000]


def main() -> None:
    model = DiscreteEtaModel(beta=BETA, etas=ETAS_HOURS, weights=WEIGHTS)
    exact = model.failure_probability(MISSION_TIME_HOURS)
    f_i = model.component_failure_probabilities(MISSION_TIME_HOURS)

    print("Discrete eta-mixture TDDB Weibull model (time unit: hours)")
    print(f"  beta = {model.beta}")
    for eta, w, f in zip(model.etas, model.weights, f_i):
        print(f"  eta_i = {eta:>9.1f} h   w_i = {w:.2f}   F(t | beta, eta_i) = {f:.6f}")
    print(f"  mission time t = {MISSION_TIME_HOURS:.1f} h")
    print(f"  exact p = sum_i w_i F_i = {exact:.6f}\n")

    print(f"Classical Monte Carlo (seed = {SEED}, 95% Wilson CI)")
    print(f"{'n':>10} {'p_hat':>10} {'std_err':>10} {'CI low':>10} {'CI high':>10} {'|err|':>10} {'exact in CI':>12}")
    for n in SAMPLE_SIZES:
        r = estimate_failure_probability(model, MISSION_TIME_HOURS, n, seed=SEED)
        inside = r.ci_low <= exact <= r.ci_high
        print(
            f"{n:>10d} {r.estimate:>10.6f} {r.std_error:>10.6f} {r.ci_low:>10.6f} "
            f"{r.ci_high:>10.6f} {abs(r.estimate - exact):>10.6f} {str(inside):>12}"
        )


if __name__ == "__main__":
    main()
