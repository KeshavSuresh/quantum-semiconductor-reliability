# Milestone 3 design: Quantum Amplitude Estimation on the shared discrete model

Status: **design only**. No Qiskit code exists yet; this document fixes the
mathematics so that Milestone 3 estimates exactly the quantity that the
Milestone 1 classical Monte Carlo estimates.

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
  confirmed at the start of Milestone 3.

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
