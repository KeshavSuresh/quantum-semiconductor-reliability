# Milestone 3 design: Quantum Amplitude Estimation on the shared discrete model

Status: **implemented in Milestone 3** (`src/qsr/quantum/`). Sections 1–6 are
the original design, which is unchanged. Section 7 records the implementation
decisions and section 8 the verification results.

## 1. Shared target quantity

Both methods use `qsr.models.uncertainty.DiscreteEtaModel`:

- shared Weibull shape `beta > 0`
- discrete scale values `eta_i`, `i = 0 .. K-1`, with weights `w_i >= 0`, `sum_i w_i = 1`
- mission time `t` (same time unit as `eta_i`)

Per-component failure probability:

    F_i = F(t | beta, eta_i) = 1 - exp(-(t / eta_i)^beta)

Target:

    p = sum_i w_i * F_i

`DiscreteEtaModel.failure_probability(t)` returns `p` exactly, and
`DiscreteEtaModel.component_failure_probabilities(t)` returns the vector `F_i`
used for the quantum encoding below.

## 2. State to prepare

Registers: an index register of `n = ceil(log2 K)` qubits and one failure
ancilla. Weights are zero-padded to length `2^n`, with padded entries having
`w_i = 0` and any `F_i`.

The algorithm `A` maps `|0>_n |0>_anc` to

    |psi> = sum_i sqrt(w_i) |i>_n  ⊗  [ sqrt(1 - F_i) |0>_anc + sqrt(F_i) |1>_anc ]

Implementation plan:

1. **Uncertainty loading**: state preparation on the index register with
   amplitudes `sqrt(w_i)`, giving `sum_i sqrt(w_i)|i>`.
2. **Failure oracle**: an RY rotation on the ancilla, uniformly controlled by the
   index register, with angle `theta_i = 2 * arcsin(sqrt(F_i))`, so that
   `RY(theta_i)|0> = sqrt(1-F_i)|0> + sqrt(F_i)|1>`.

## 3. Why this gives p

The probability of measuring the ancilla in `|1>` is

    P(anc = 1) = sum_i |sqrt(w_i) * sqrt(F_i)|^2 = sum_i w_i F_i = p

So `p` is the amplitude-squared `a` that QAE estimates.

## 4. Estimation

- Grover operator `Q = A S_0 A^† S_good`, where `S_good` flips the phase of
  states with ancilla `|1>`.
- Use Iterative or Maximum-Likelihood Amplitude Estimation, which need no QFT and
  suit simulators. Their estimation error scales as `O(1/M)` in the number of
  oracle (`A`) queries `M`, compared with `O(1/sqrt(N))` for `N` classical samples.
- In Qiskit 2.x the amplitude-estimation algorithms live in the separate
  `qiskit-algorithms` package. The exact package and version choice will be
  confirmed at the start of Milestone 3. (Resolved in section 7: not used.)

## 5. Caveats to report honestly

- The `F_i` values are computed classically and loaded as rotation angles. The
  quantum speed-up claim concerns only the estimation of the weighted sum given
  that loading, not computing `F_i`.
- State preparation for arbitrary `w_i` costs `O(2^n)` gates. A fair comparison
  counts oracle queries, and also reports circuit depth separately.
- Simulators are noiseless unless a noise model is added explicitly.

## 6. Planned verification for Milestone 3

1. The statevector probability of `anc = 1` equals
   `model.failure_probability(t)` to within `1e-12`.
2. The QAE estimate falls within its reported confidence interval of the exact `p`.
3. Error vs. query count for QAE is plotted against error vs. sample count for
   classical MC (`qsr.simulation.monte_carlo`), using the same model, mission time
   and seeds.

## 7. Implementation decisions (Milestone 3, revised after technical review)

### 7.1 Engine
Own Maximum-Likelihood Amplitude Estimation (Suzuki et al. 2020) on core Qiskit
only: `StatePreparation` (loads `sqrt(w_i)`), `UCRYGate` (the uniformly
controlled RY oracle), the non-deprecated `grover_operator()` function, and
`StatevectorSampler`. `qiskit-algorithms` 0.4.0 was rejected because it relies
on APIs deprecated since Qiskit 2.1. The test suite passes with
`DeprecationWarning` and `PendingDeprecationWarning` treated as errors on both
Qiskit 2.1.0 and 2.6.1, which is the basis for the declared range `qiskit>=2.1,<3`.

Registers: qubits `0..n-1` hold the η index (Qiskit little-endian), and the
last qubit is the failure ancilla (`|1>` = failure). `S_good = Z` on the
ancilla, and `S_0` is the reflection about `|0…0>` built by `grover_operator`.

### 7.2 Query accounting (convention)
**One oracle query = one application of `A` or `A†` in one shot.**

| Quantity | Formula | Example (m = 0,1,2,4,8,16; N = 100) |
|---|---|---|
| A or A† per shot of circuit `Q^m A` | `2m + 1` (one A, plus one A and one A† per Q) | 1, 3, 5, 9, 17, 33 |
| `state_preparation_calls` (= "oracle calls") | `N · Σ(2m_k + 1)` | 100 × 68 = **6,800** |
| `grover_applications` (= phase-oracle `S_good` calls) | `N · Σ m_k` | 100 × 31 = 3,100 |
| `total_shots` | `N · K` | 100 × 6 = 600 |

These counts are checked against the executed circuits by counting the
`ucry`/`ucry_dg` and `state_preparation`/`state_preparation_dg` instructions
(2m+1 each) and the `z` gates (m) in every circuit (`test_query_count_matches_executed_circuits`).
The phase oracle and Q are reported separately and are **not** added to the
oracle-call count.

Classical MC receives `n_samples = state_preparation_calls`. This is the
standard sample-complexity comparison: one classical sample evaluates the
full model once, and one `A` evaluates it once coherently. It deliberately
ignores the very different cost of one `A` on hardware compared with one MC
sample. Both methods use the same model, mission time and integer seed. The
seed only makes each method reproducible; their random streams are unrelated
(not paired).

### 7.3 Sampling independence (bug found in review and fixed)
`StatevectorSampler(seed=<int>)` re-seeds identically for **every circuit** in
a job. Two identical circuits then return identical shot sequences, and the
counts of different Grover powers were correlated (r ≈ 0.32 to 0.40 measured).
That violated the independence assumption of the MLAE likelihood. Each circuit
now runs with its own seed from `circuit_seeds(seed, K)`, which uses
`numpy.random.SeedSequence(seed).spawn(K)`. Measured correlation is now
≈ 0.01, consistent with zero. A regression test fails on the old behaviour
(r = 0.321 on its fixed seeds) and passes on the new one. **All results
produced before this fix are superseded by the numbers in section 8.**

### 7.4 Maximum-likelihood search
- Log-likelihood `Σ_k [h_k log sin²(M_k θ) + (N − h_k) log cos²(M_k θ)]`,
  `M_k = 2m_k + 1`, with `h_k` = shots with ancilla `1`. It uses `xlogy`, so
  `0 · log 0 = 0` at the boundaries.
- Uniform grid on `[0, π/2]`. Step ≤ 0.05 × `se_θ` and ≥ 200 points per
  shortest period `π / M_max`, with at least 20,001 points. Problems that would
  need more than 5,000,000 points are refused with a clear error rather than
  silently under-resolved. Example: 24,287 points for m ≤ 16, N = 100; about
  1.5 M points for m ≤ 32, N = 10⁵.
- The 5 highest grid local maxima are refined by bounded Brent optimisation
  within the neighbouring grid cells, and the best is kept. Exact ties (a
  genuinely ambiguous likelihood, for example when there is no m = 0 circuit)
  go to the smallest θ.
- θ = 0 (p = 0) and θ = π/2 (p = 1) are grid points, so boundary maxima are exact.
- Checked against an independent 2,000,001-point brute-force search: the
  estimator agreed to within the brute-force grid step in all non-tied cases,
  including a case with ≥ 5 local maxima and N = 10⁵ shots.

### 7.5 Confidence interval
- Likelihood-ratio set `{θ : 2[ℓ(θ̂) − ℓ(θ)] ≤ χ²₁(0.95)}`. Endpoints are solved
  by Brent root finding between grid points, and tests confirm the endpoint
  statistic equals χ²₁ to 1e-6. An endpoint at 0 or π/2 is kept exactly.
- If the set is several disjoint regions, the interval is their **hull**
  (conservative), and `n_ci_regions` reports how many regions there were.
- The χ² calibration is **asymptotic**. Finite-sample coverage was measured
  in section 8 and is not guaranteed.

### 7.6 Standard error
For this model the Fisher information is exactly `I(θ) = 4N Σ M_k²`, for any
θ. So `se_θ = 1/√I` is the asymptotic Cramér–Rao standard error, and it is
used for the grid resolution and the theory curve. The delta-method SE in `p`,
`sin(2θ̂)·se_θ`, is reported only when θ̂ is more than 3 `se_θ` from both
boundaries. Otherwise it is NaN, because the delta method fails at p = 0 or 1.

### 7.7 Circuit metrics and timing
Depth and CX count come from
`transpile(basis_gates=[rz, sx, x, cx], optimization_level=1, seed_transpiler=0)`,
with no device topology. Wall-clock time is measured separately for build,
transpile, simulate and estimate. These timings come from a classical
simulator and are **not** quantum-hardware estimates.

## 8. Verification results

All numbers below come from the code after the review fixes. Commands:
`python examples/milestone3_qae.py` and `python examples/milestone3_ci_calibration.py`.

### 8.1 Circuit correctness (exact statevector, no sampling)
- Example model (β = 2, η = {10⁴, 2·10⁴, 4·10⁴} h, w = {0.2, 0.5, 0.3}, t = 5000 h):
  `P(anc = 1) = 0.079184380877359`. It matches `model.failure_probability(t)`
  with |difference| = 5.6e-17.
- Tested to 1e-12 for K = 1, 2, 3 and 5 (padded to 8), t from 0 to 10⁶ h, and
  F = 0 and F = 1 exactly (`RY(π)`). The weight sum at the ±9e-10 tolerance
  edge matches to 2e-9.
- Branch probabilities: |i⟩|1⟩ = wᵢFᵢ, |i⟩|0⟩ = wᵢ(1−Fᵢ), and padded
  indices < 1e-15 for every test model. This confirms the endianness.
- `P(anc = 1 after Q^m A) = sin²((2m+1)θ)` to 1e-10 for m ∈ {0, 1, 2, 3, 4, 8},
  across all test models and two mission times.

### 8.2 Confidence-interval calibration (95 % nominal, N = 100 shots per circuit)
**A. Count-level study.** 2,000 trials per cell, 30 cells (p ∈ {0.01, 0.0792,
0.3, 0.7, 0.95} × m_max ∈ {0, 1, 2, 4, 8, 16}), 60,000 trials in total,
`SeedSequence(910000)`, a seed range used nowhere else. Counts were drawn as
independent Binomial(N, sin²(M_kθ)), the distribution the pipeline produces
(see 8.1 and 7.3).

| p | m_max = 0 | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|
| 0.01 | 1958 | 1908 | 1899 | 1897 | 1833* | 1855* |
| 0.0792 | 1868 | 1921 | 1870* | 1890 | 1851* | 1988 |
| 0.3 | 1891 | 1832* | 1887 | 1891 | 1899 | 1890 |
| 0.7 | 1902 | 1801* | 1897 | 1877 | 1886 | 1897 |
| 0.95 | 1862 | 1879 | 1895 | 1901 | 1895 | 1882 |

Entries are covered trials out of 2,000. \* marks cells with ≥ 256 multi-region
likelihood-ratio sets.

- Pooled: **56,602 / 60,000 = 0.943** (Wilson 95 % [0.941, 0.945]).
- If the true coverage were 0.95, the sampling SD per cell would be
  √(0.95·0.05/2000) = 0.49 %.
- 10 of 30 cells are significantly below 0.95 (worst 1801/2000 = 0.900,
  [0.887, 0.913]), and 3 are significantly above (best 1988/2000 = 0.994).
- Under-coverage concentrates where the likelihood is multimodal. The 6 cells
  with ≥ 256 multi-region sets have coverage 0.900–0.935, while the 19 cells
  with none pool to 0.947. In those cells the asymptotic χ²₁ calibration of the
  likelihood-ratio statistic does not hold, and taking the hull does not
  compensate. The m = 0 cells (0.934–0.979) show the usual discreteness of any
  binomial interval.

**B. End-to-end Qiskit pipeline.** Example model, 500 trials per schedule,
seeds 700000+, which are disjoint from everything else.

| m_max | covered / 500 | Wilson 95 % | RMSE | Cramér–Rao bound | RMSE / CRB |
|---|---|---|---|---|---|
| 1 | 477 | [0.932, 0.969] | 0.008606 | 0.008539 | 1.01 |
| 4 | 480 | [0.939, 0.974] | 0.002473 | 0.002507 | 0.99 |
| 16 | 495 | [0.977, 0.996] | 0.000615 | 0.000699 | 0.88 |

These agree with part A for the same p within sampling error.

**Verdict.** The interval is an **approximate** 95 % interval. Measured
coverage is 0.90–0.99 depending on p and schedule (0.943 pooled). It should
not be described as having guaranteed or exact 95 % coverage.

### 8.3 Estimator accuracy vs the Cramér–Rao bound
In a 4,000-trial count-level check at p = 0.0792, the finite-sample MLE has
small but statistically significant bias: −3.9e-4 ± 0.3e-4 at m_max = 8, and
+7.9e-5 ± 0.9e-5 at m_max = 16. Its RMSE is 0.99×, 1.48× and 0.83× the CRB at
m_max = 1, 8 and 16. The CRB bounds unbiased estimators only and is reached
asymptotically, so it is plotted as a reference, not as a guarantee or a floor.
Before the sampling fix (7.3), correlated counts had produced RMSE values down
to 0.69× the CRB. Those results are withdrawn.

### 8.4 Error vs oracle calls (30 fixed seeds 0–29, N = 100; illustrative)

| m_max | oracle calls | QAE RMSE | CRB | MC RMSE | MC binomial SE | QAE s/run | MC s/run |
|---|---|---|---|---|---|---|---|
| 0 | 100 | 0.0237 | 0.0270 | 0.0283 | 0.0270 | 0.008 | 0.0003 |
| 4 | 1,800 | 0.00250 | 0.00251 | 0.00559 | 0.00637 | 0.078 | 0.0005 |
| 16 | 6,800 | 0.00063 | 0.00070 | 0.00359 | 0.00328 | 0.267 | 0.0006 |
| 32 | 13,300 | 0.00035 | 0.00036 | 0.00258 | 0.00234 | 0.509 | 0.0009 |

Measured RMSE follows ~1/calls for QAE and ~1/√calls for MC, which is
query-complexity scaling. **Wall-clock time per run is ≈ 590× higher for QAE
than for MC** at 13,300 calls on this classical simulator. These timings are
not hardware estimates, and **no practical quantum speed-up is shown or claimed.**

### 8.5 Single run (seed 2026, m = 0,1,2,4,8,16, N = 100)
- QAE: 0.078668, CI [0.076968, 0.080449], 1 region.
- MC: 0.077206, CI [0.071098, 0.083791], with the same 6,800-call budget.
- Exact: 0.079184.
- Circuit depth 25 → 921 and CX 5 → 261 (basis rz, sx, x, cx; opt level 1;
  no topology).
- QAE run time 0.42 s, MC run time 0.0006 s.
