"""Diagnostics for the MLAE likelihood-ratio interval (supports docs/qae_design.md §8).

All parts use fresh seeds disjoint from every other study in the project.

1. Failure analysis in selected cells (2,000 trials each). For every trial the
   likelihood l(theta) is ALSO evaluated by brute force on a dense grid
   (2,000,001 points), independently of ``mlae_estimate``. For each trial that
   is not covered:
     * Grid/optimiser check: ``mlae_estimate`` must reach the brute-force maximum
       (to 1e-6 in log-likelihood), its hull endpoints must match the brute-force
       hull to within two dense-grid steps, and its region count must match.
     * Peak classification against the actual likelihood: hill-climb on the
       brute-force l(theta) starting from the true theta0. If the local maximum
       reached is not the global maximum, the MLE is on a WRONG PEAK. Otherwise
       the MLE is on the right peak and the failure is because the LR statistic
       W0 = 2[l(theta_hat) - l(theta0)] exceeds the chi^2_1 cut (a calibration
       failure of the asymptotic chi^2 approximation).
2. Hull vs union of regions: for every trial whose LR set has more than one
   region, the brute-force union of regions is compared with the reported
   hull (coverage gained by the hull; width ratio hull / union).
3. Worst cell re-run with 20,000 trials (a = 0.01, m_max = 8).
4. Sampler fix: legacy behaviour (all circuits in one StatevectorSampler job
   with one integer seed, so every circuit reuses the same random stream) vs
   the current per-circuit seeds, with identical seeds, model and shots.
5. Estimator bias and RMSE vs the Cramer-Rao bound (4,000 count-level trials
   per schedule at p = 0.0792; SeedSequence(930000 + schedule length)).

Run:  python examples/milestone3_ci_diagnostics.py               (all parts, about 40 minutes)
      python examples/milestone3_ci_diagnostics.py --parts 4     (sampler comparison only, about 5 minutes)
      python examples/milestone3_ci_diagnostics.py --parts 5     (bias study only, about 1 minute)
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
from qiskit.primitives import StatevectorSampler
from scipy.special import xlogy
from scipy.stats import chi2

from qsr.models.uncertainty import DiscreteEtaModel
from qsr.quantum import (
    build_amplification_circuits,
    build_state_preparation,
    estimate_failure_probability_qae,
    exponential_schedule,
    mlae_estimate,
)
from qsr.simulation.monte_carlo import wilson_interval

SHOTS = 100
CUT = float(chi2.ppf(0.95, df=1))
DIAG_SEED = 951_000           # SeedSequence entropy for parts 1-3 (used nowhere else)
SAMPLER_SEED_START = 760_000  # seeds for part 4 (used nowhere else)
DENSE = np.linspace(0.0, math.pi / 2, 2_000_001)
DENSE_STEP = float(DENSE[1] - DENSE[0])


def loglik(theta, hits, sch):
    th = np.atleast_1d(np.asarray(theta, dtype=float))[:, None]
    m = 2 * np.asarray(sch, dtype=float)[None, :] + 1
    h = np.asarray(hits, dtype=float)[None, :]
    return np.sum(xlogy(h, np.sin(m * th) ** 2) + xlogy(SHOTS - h, np.cos(m * th) ** 2), axis=1)


def dense_ll(hits, sch):
    return np.concatenate([loglik(DENSE[i:i + 250_000], hits, sch) for i in range(0, DENSE.size, 250_000)])


def hill_climb(ll: np.ndarray, start: int) -> int:
    """Index of the local maximum of ``ll`` reached by steepest ascent from ``start``."""
    i = start
    n = ll.size
    while True:
        left = ll[i - 1] if i > 0 else -np.inf
        right = ll[i + 1] if i < n - 1 else -np.inf
        if left <= ll[i] and right <= ll[i]:
            return i
        # Jump along the monotone run in the ascending direction (fast on a dense grid).
        if right > left:
            j = i + 1
            while j < n - 1 and ll[j + 1] > ll[j]:
                j += 1
        else:
            j = i - 1
            while j > 0 and ll[j - 1] > ll[j]:
                j -= 1
        i = j


def regions(ll: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    above = ll >= threshold
    edges = np.flatnonzero(np.diff(np.concatenate([[0], above.astype(np.int8), [0]])))
    return list(zip(edges[0::2].tolist(), (edges[1::2] - 1).tolist()))


def wl(k: int, n: int) -> str:
    lo, hi = wilson_interval(k, n, 0.95)
    return f"{k}/{n} = {k / n:.4f} [Wilson 95% {lo:.4f}, {hi:.4f}]"


def schedule_upto(m_max: int) -> tuple[int, ...]:
    return tuple(m for m in exponential_schedule(10) if m <= m_max)


def analyse_cell(a: float, m_max: int, n_trials: int, rng, *, brute_all: bool = True) -> dict:
    theta0 = math.asin(math.sqrt(a))
    sch = schedule_upto(m_max)
    pk = np.sin((2 * np.array(sch) + 1) * theta0) ** 2
    i0 = int(round(theta0 / DENSE_STEP))
    s = dict(n=n_trials, cov_hull=0, cov_union=0, multi=0, fail=0, wrong_peak=0, right_peak=0,
             brute_checked=0, brute_mismatch=0, width_ratios=[], hull_gain=0, multi_checked=0,
             sch=sch, pk=pk, theta0=theta0)
    for _ in range(n_trials):
        hits = rng.binomial(SHOTS, pk).tolist()
        fit = mlae_estimate(hits, SHOTS, sch)
        covered = fit.ci_low <= a <= fit.ci_high
        s["cov_hull"] += covered
        multi = fit.n_ci_regions > 1
        s["multi"] += multi
        need_brute = (not covered) or multi
        if not need_brute:
            s["cov_union"] += 1  # single region: union == hull
            continue
        if not brute_all and covered:
            s["cov_union"] += 1
            continue
        ll = dense_ll(hits, sch)
        g = int(np.argmax(ll))
        regs = regions(ll, float(ll[g]) - CUT / 2)
        in_union = any(lo <= i0 <= hi for lo, hi in regs)
        s["cov_union"] += in_union
        if multi:
            s["multi_checked"] += 1
            union_len = sum(math.sin(DENSE[hi]) ** 2 - math.sin(DENSE[lo]) ** 2 for lo, hi in regs)
            hull_len = math.sin(DENSE[regs[-1][1]]) ** 2 - math.sin(DENSE[regs[0][0]]) ** 2
            if union_len > 0:
                s["width_ratios"].append(hull_len / union_len)
            s["hull_gain"] += covered and not in_union
        if not covered:
            s["fail"] += 1
            # Grid/optimiser validation against brute force.
            s["brute_checked"] += 1
            ll_hat = float(loglik(fit.theta_hat, hits, sch)[0])
            tol_a = 2 * DENSE_STEP  # |d sin^2 / d theta| <= 1, two dense-grid steps
            lo_b = math.sin(DENSE[regs[0][0]]) ** 2
            hi_b = math.sin(DENSE[regs[-1][1]]) ** 2
            bad = (ll_hat < float(ll[g]) - 1e-6 or abs(lo_b - fit.ci_low) > tol_a
                   or abs(hi_b - fit.ci_high) > tol_a or len(regs) != fit.n_ci_regions)
            s["brute_mismatch"] += bad
            # Peak classification against the actual likelihood.
            peak0 = hill_climb(ll, i0)
            if abs(peak0 - g) > 2:
                s["wrong_peak"] += 1
            else:
                s["right_peak"] += 1
    return s


def legacy_hits(circuits, seed: int) -> list[int]:
    """Pre-fix behaviour: one job, one integer seed -> every circuit reuses the same random stream."""
    res = StatevectorSampler(seed=seed).run(circuits, shots=SHOTS).result()
    return [int(r.data.c.get_counts().get("1", 0)) for r in res]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parts", default="12345", help="which parts to run, e.g. '4' or '12345' (default)")
    parts = parser.parse_args().parts
    streams = iter(np.random.SeedSequence(DIAG_SEED).spawn(64))
    first_stream, worst_stream = [next(streams) for _ in range(9)], next(streams)  # fixed assignment per part

    if "1" in parts or "2" in parts:
        run_parts_12(first_stream)
    if "3" in parts:
        run_part_3(worst_stream)
    if "4" in parts:
        run_part_4()
    if "5" in parts:
        run_part_5()


def run_part_5() -> None:
    print("\n5. Estimator bias and RMSE vs Cramer-Rao bound (p = 0.0792, 4,000 count-level trials per schedule)")
    a = 0.07918438087735861
    th = math.asin(math.sqrt(a))
    for length in (2, 5, 6):
        sch = exponential_schedule(length)
        pk = np.sin((2 * np.array(sch) + 1) * th) ** 2
        rng = np.random.default_rng(np.random.SeedSequence(930_000 + length))
        est = np.array([mlae_estimate(rng.binomial(SHOTS, pk).tolist(), SHOTS, sch).estimate for _ in range(4000)])
        crb = abs(math.sin(2 * th)) / math.sqrt(4 * SHOTS * sum((2 * m + 1) ** 2 for m in sch))
        bias, sd = est.mean() - a, est.std(ddof=1)
        rmse = math.sqrt(float(np.mean((est - a) ** 2)))
        print(f"   m_max={max(sch):>2}: bias {bias:+.2e} (SE {sd / math.sqrt(4000):.1e})  SD {sd:.6f}  "
              f"RMSE {rmse:.6f}  CRB {crb:.6f}  RMSE/CRB {rmse / crb:.3f}")


def run_parts_12(seed_streams) -> None:
    print("1-2. Failure and hull analysis (2,000 fresh trials per cell; every non-covered or multi-region "
          "trial re-evaluated by 2,000,001-point brute force)")
    cells = [(0.01, 8), (0.01, 16), (0.0792, 0), (0.0792, 2), (0.0792, 8), (0.3, 1), (0.7, 1), (0.95, 0), (0.3, 8)]
    t0 = time.perf_counter()
    for (a, m_max), stream in zip(cells, seed_streams):
        r = analyse_cell(a, m_max, 2000, np.random.default_rng(stream))
        wr = np.array(r["width_ratios"])
        width = f"hull/union width median {np.median(wr):.2f}, max {wr.max():.2f}" if wr.size else "no multi-region sets"
        print(f"   a={a:<6} m_max={m_max:<2} hull coverage {wl(r['cov_hull'], r['n'])}; union-of-regions coverage "
              f"{r['cov_union']}/{r['n']}; multi-region {r['multi']} (hull gained {r['hull_gain']}; {width})")
        print(f"      failures {r['fail']}: wrong peak {r['wrong_peak']}, right peak with W0 > chi2_1 {r['right_peak']}; "
              f"brute-force mismatches {r['brute_mismatch']}/{r['brute_checked']}")
    print(f"   elapsed {time.perf_counter() - t0:.0f} s")


def run_part_3(stream) -> None:
    print("\n3. Worst cell re-run: a = 0.01, m_max = 8, 20,000 fresh trials")
    t0 = time.perf_counter()
    # Brute force only for non-covered trials here (cost); hull/union widths are studied in part 2.
    r = analyse_cell(0.01, 8, 20_000, np.random.default_rng(stream), brute_all=False)
    print(f"   schedule {list(r['sch'])}; per-circuit P(anc=1) = {np.round(r['pk'], 4).tolist()}; theta0 = {r['theta0']:.5f}")
    print(f"   hull coverage {wl(r['cov_hull'], r['n'])}")
    print(f"   multi-region {r['multi']} (union-vs-hull not brute-forced for covered trials in this part)")
    print(f"   failures {r['fail']}: wrong peak {r['wrong_peak']}, right peak with W0 > chi2_1 {r['right_peak']}; "
          f"brute-force mismatches {r['brute_mismatch']}/{r['brute_checked']}")
    print(f"   elapsed {time.perf_counter() - t0:.0f} s")


def run_part_4() -> None:
    print("\n4. Sampler: legacy single-job integer seed vs per-circuit seeds (identical seeds 760000+, "
          "TDDB model, t = 5000 h, 400 trials, 100 shots)")
    model = DiscreteEtaModel(beta=2.0, etas=[1e4, 2e4, 4e4], weights=[0.2, 0.5, 0.3])
    exact = model.failure_probability(5e3)
    theta = math.asin(math.sqrt(exact))
    t0 = time.perf_counter()
    for length in (4, 6):
        sch = exponential_schedule(length)
        circuits = build_amplification_circuits(build_state_preparation(model, 5e3), sch)
        crb = abs(math.sin(2 * theta)) / math.sqrt(4 * SHOTS * sum((2 * m + 1) ** 2 for m in sch))
        for label in ("legacy", "current"):
            cov, sq, h1, h4 = 0, [], [], []
            for k in range(400):
                seed = SAMPLER_SEED_START + 1000 * length + k
                if label == "legacy":
                    hits = legacy_hits(circuits, seed)
                    fit = mlae_estimate(hits, SHOTS, sch)
                    est, lo, hi = fit.estimate, fit.ci_low, fit.ci_high
                else:
                    q = estimate_failure_probability_qae(model, 5e3, seed=seed, evaluation_schedule=sch,
                                                         shots=SHOTS, report_depth=False)
                    hits, est, lo, hi = list(q.hits), q.estimate, q.ci_low, q.ci_high
                cov += lo <= exact <= hi
                sq.append((est - exact) ** 2)
                h1.append(hits[1])  # m = 1
                h4.append(hits[3])  # m = 4 (both mid-range probabilities, so the correlation is informative)
            rmse = math.sqrt(float(np.mean(sq)))
            print(f"   m_max={max(sch):<2} {label:<7} coverage {wl(cov, 400)}  RMSE {rmse:.6f}  RMSE/CRB {rmse / crb:.2f}  "
                  f"corr(hits[m=1], hits[m=4]) = {np.corrcoef(h1, h4)[0, 1]:+.3f}")
    print(f"   elapsed {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
