"""Calibration study for the MLAE likelihood-ratio confidence interval (95 % nominal).

Two parts, each with seed ranges used NOWHERE else in the project (no tuning on
these trials):

A. Count-level study (large). For each true probability a and schedule, draw
   h_k ~ Binomial(N, sin^2((2 m_k + 1) theta)) independently, then run
   ``mlae_estimate``. This is exactly the distribution the Qiskit pipeline
   produces: the amplification identity is verified to 1e-10 in the tests,
   and circuits use independent sampler seeds. Drawing counts directly makes
   thousands of trials practical.
B. End-to-end check (smaller). The full Qiskit pipeline
   (``estimate_failure_probability_qae``) on the example TDDB model.

For every cell the script reports covered / trials with a 95 % Wilson interval
for the coverage itself. If the true coverage were 0.95, the binomial SD of the
measured coverage would be sqrt(0.95 x 0.05 / n) (0.49 % for n = 2000,
0.97 % for n = 500).

Run:  python examples/milestone3_ci_calibration.py            (about 8 minutes)
      python examples/milestone3_ci_calibration.py --quick    (about 1 minute; preliminary)
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from qsr.models.uncertainty import DiscreteEtaModel
from qsr.quantum import estimate_failure_probability_qae, exponential_schedule, mlae_estimate
from qsr.simulation.monte_carlo import wilson_interval

COUNT_STUDY_SEED = 910_000          # SeedSequence entropy for part A (never used elsewhere)
PIPELINE_SEED_START = 700_000       # first seed for part B (never used elsewhere)
SHOTS = 100
CONFIDENCE = 0.95


def coverage_line(label: str, covered: int, n: int, extra: str = "") -> str:
    lo, hi = wilson_interval(covered, n, 0.95)
    return f"   {label:<32} {covered:>5}/{n:<5} = {covered / n:6.3f}   95% CI for coverage [{lo:.3f}, {hi:.3f}] {extra}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="fewer trials (preliminary only)")
    args = parser.parse_args()
    n_count = 300 if args.quick else 2000
    n_pipe = 100 if args.quick else 500

    probs = (0.01, 0.0792, 0.3, 0.7, 0.95)
    lengths = (1, 2, 3, 4, 5, 6)
    print(f"A. Count-level study: {n_count} trials per cell, shots = {SHOTS}, nominal {CONFIDENCE:.0%}, "
          f"SeedSequence({COUNT_STUDY_SEED})")
    t0 = time.perf_counter()
    total_cov = total_n = 0
    multi_region = 0
    root = np.random.SeedSequence(COUNT_STUDY_SEED)
    cells = root.spawn(len(probs) * len(lengths))
    for i, a in enumerate(probs):
        theta = math.asin(math.sqrt(a))
        for j, length in enumerate(lengths):
            sch = exponential_schedule(length)
            p_k = np.sin((2 * np.array(sch) + 1) * theta) ** 2
            rng = np.random.default_rng(cells[i * len(lengths) + j])
            covered = regions = 0
            for _ in range(n_count):
                hits = rng.binomial(SHOTS, p_k).tolist()
                fit = mlae_estimate(hits, SHOTS, sch, CONFIDENCE)
                covered += fit.ci_low <= a <= fit.ci_high
                regions += fit.n_ci_regions > 1
            total_cov += covered
            total_n += n_count
            multi_region += regions
            print(coverage_line(f"a = {a:<6} m_max = {max(sch):<3}", covered, n_count,
                                f"(multi-region CIs: {regions})"))
    print(coverage_line("ALL cells pooled", total_cov, total_n, f"(multi-region CIs: {multi_region})"))
    print(f"   elapsed {time.perf_counter() - t0:.0f} s")

    model = DiscreteEtaModel(beta=2.0, etas=[1e4, 2e4, 4e4], weights=[0.2, 0.5, 0.3])
    t_mission = 5e3
    exact = model.failure_probability(t_mission)
    print(f"\nB. End-to-end Qiskit pipeline: TDDB model, t = {t_mission:g} h, exact p = {exact:.6f}, "
          f"{n_pipe} trials per schedule, seeds {PIPELINE_SEED_START}..")
    t0 = time.perf_counter()
    for length in (2, 4, 6):
        sch = exponential_schedule(length)
        covered = 0
        sq = []
        for k in range(n_pipe):
            r = estimate_failure_probability_qae(
                model, t_mission, seed=PIPELINE_SEED_START + 10_000 * length + k,
                evaluation_schedule=sch, shots=SHOTS, confidence=CONFIDENCE, report_depth=False,
            )
            covered += r.ci_low <= exact <= r.ci_high
            sq.append((r.estimate - exact) ** 2)
        theta = math.asin(math.sqrt(exact))
        crb = abs(math.sin(2 * theta)) / math.sqrt(4 * SHOTS * sum((2 * m + 1) ** 2 for m in sch))
        rmse = math.sqrt(float(np.mean(sq)))
        print(coverage_line(f"m_max = {max(sch):<3}", covered, n_pipe,
                            f"RMSE {rmse:.6f} vs Cramer-Rao {crb:.6f} (ratio {rmse / crb:.2f})"))
    print(f"   elapsed {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
