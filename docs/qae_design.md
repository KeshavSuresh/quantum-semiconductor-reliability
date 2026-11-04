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

## 7. Implementation decisions (Milestone 3)

- **Engine: own Maximum-Likelihood Amplitude Estimation on core Qiskit.**
  `qiskit-algorithms` 0.4.0 runs on Qiskit 2.6.1 but relies on APIs deprecated
  since Qiskit 2.1 (`GroverOperator`, `MCXGate.get_num_ancilla_qubits`), so it was
  not used. The implementation uses only core Qiskit: `StatePreparation` (loads
  `sqrt(w_i)`), `UCRYGate` (the uniformly controlled RY oracle), the
  non-deprecated `grover_operator()` function, and the seeded
  `StatevectorSampler`. The test suite runs with `DeprecationWarning` treated as
  an error.
- **Registers:** qubits `0..n-1` hold the η index (Qiskit little-endian), and the
  last qubit is the failure ancilla. `S_good` is `Z` on the ancilla.
- **Schedule:** exponential, `m = 0, 1, 2, 4, …` (Suzuki et al. 2020), with the
  same shot count per circuit.
- **Estimator:** maximise `sum_k h_k log sin²(M_k θ) + (N − h_k) log cos²(M_k θ)`,
  `M_k = 2 m_k + 1`, with a global grid search followed by bounded local refinement.
- **Confidence interval:** likelihood-ratio interval (χ²₁ threshold), mapped
  through `sin²`. The Fisher (Wald) interval was tried first and under-covered
  for short schedules (77–90 % coverage at the 95 % level), so it is reported
  only as an asymptotic standard error `sin(2θ)/sqrt(4 N Σ M_k²)`.
- **Query accounting:**
  - `state_preparation_calls = N · Σ(2 m_k + 1)`: uses of `A` or `A†`. This is
    the "oracle calls" budget, and classical Monte Carlo is given exactly this
    many samples.
  - `grover_applications = N · Σ m_k` is reported as well.
- **Circuit metrics:** depth and CX count after
  `transpile(basis_gates=[rz, sx, x, cx], optimization_level=1, seed_transpiler=0)`.
  No hardware topology is applied, so real devices would need more gates.
- **Timing:** wall-clock time measured separately for build, transpile,
  simulate and estimate. These are simulator timings only.

## 8. Verification results (example model, t = 5000 h)

1. Statevector `P(anc = 1)` matches `model.failure_probability(t)` to 5.6e-17.
   It is tested to 1e-12 for K = 1, 2, 3 and 5 (padded to 8) and for t from 0 to 10⁶ h.
2. `P(anc = 1 after Q^m A) = sin²((2m+1)θ)` holds to 1e-10 for m up to 8 (tested).
3. Likelihood-ratio CI coverage over 100 fixed seeds: 0.88–0.99 across
   schedules (95 % nominal).
4. Error vs. oracle calls, RMSE over 30 seeds at 100 shots per circuit:

| m_max | calls | QAE RMSE | MC RMSE |
|---|---|---|---|
| 0 | 100 | 0.0308 | 0.0283 |
| 4 | 1,800 | 0.0022 | 0.0056 |
| 32 | 13,300 | 0.00039 | 0.0026 |

   Query-complexity scaling follows theory: about 1/calls for QAE and
   1/sqrt(calls) for MC. The m = 2 point is noisier than theory
   (likelihood multimodality at short schedules).
5. Wall-clock time per run on this simulator, at 13,300 calls: QAE ≈ 0.42 s,
   MC ≈ 0.001 s. **No practical quantum speed-up is shown or claimed.**
