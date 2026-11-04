# Quantum Semiconductor Reliability Explorer

*Student research prototype: classical Monte Carlo vs simulated Quantum Amplitude
Estimation for semiconductor gate-oxide reliability.*

Every number in this document comes from the commands listed in
[Reproducing every number](#reproducing-every-number), run on the project's
code. Wall-clock times are machine-dependent; everything else is
deterministic and reproduces exactly.

---

## 1. Research question

> For a semiconductor reliability model with an uncertain parameter, how does
> the estimation error of **Quantum Amplitude Estimation (QAE)** scale with its
> number of oracle queries, compared with **classical Monte Carlo (MC)** at the
> same number of model evaluations? And what does it cost to get that result
> on today's (simulated) quantum computers?

The prototype answers this on a small, fully checkable example and is explicit
about what the answer does and does not show.

## 2. Motivation: why reliability, why TDDB

Transistor gates are separated from the channel by an oxide only a few
nanometres thick. Under operating voltage, defects slowly accumulate in the
oxide until a conducting path forms and the device fails suddenly. This is
**time-dependent dielectric breakdown (TDDB)**. Reliability engineers must
estimate the fraction of devices that fail before a *mission time* (for
example, the warranty period), because a low failure probability cannot be
measured directly on a few test chips.

## 3. Model

### 3.1 Weibull TDDB failure law

The probability that one device has broken down by time *t* is

    F(t | β, η) = 1 − exp(−(t / η)^β),        R(t) = 1 − F(t)

* **β (shape):** how sharply failures cluster in time. β > 1 means wear-out.
* **η (characteristic life):** 63.2 % of devices have failed by t = η.

### 3.2 Uncertainty model

η is not known exactly (process variation, model uncertainty). It is
represented by K possible values ηᵢ with weights wᵢ ≥ 0 that are normalised to
sum to 1. The quantity every method estimates is the mixture probability

    p(t) = Σᵢ wᵢ · F(t | β, ηᵢ)

**Example used throughout** (time unit: hours):

| | η₁ | η₂ | η₃ |
|---|---|---|---|
| η (h) | 10,000 | 20,000 | 40,000 |
| weight w | 0.2 | 0.5 | 0.3 |
| F(5000 h \| β = 2, η) | 0.221199 | 0.060587 | 0.015504 |

With β = 2 and mission time t = 5,000 h:
**p = 0.2·0.221199 + 0.5·0.060587 + 0.3·0.015504 = 0.079184** (reliability
R = 0.920816). This exact value is the reference for both estimators.

## 4. Methods

### 4.1 Classical Monte Carlo

For each sample: pick ηᵢ with probability wᵢ, draw a Weibull failure time by
inverse-CDF sampling, and record a failure if it is before *t*. The estimate
is the failure fraction k/n, reported with a **95 % Wilson** binomial
interval. Error falls like **1/√n**.

### 4.2 Quantum Amplitude Estimation (MLAE)

1. **State preparation A.** Two index qubits are put into the superposition
   Σᵢ √wᵢ |i⟩. One *failure qubit* is then rotated, controlled by the index, so
   that it reads 1 with probability Fᵢ. Overall,
   P(failure qubit = 1) = Σᵢ wᵢ Fᵢ = p, exactly the mixture probability.
   The exact statevector reproduces p to 5.6 × 10⁻¹⁷.
2. **Amplification.** The Grover operator Q is applied m times. The chance of
   reading 1 becomes sin²((2m+1)θ), where p = sin²θ.
3. **Measurement.** Circuits for m = 0, 1, 2, 4, 8, 16 are each run for 100
   shots on a seeded, noiseless statevector simulator.
4. **Estimation.** Maximum-likelihood estimation finds the θ that best
   explains all the counts. The interval is a likelihood-ratio interval.

In theory, the estimation error falls like **1/(number of oracle queries)**
instead of 1/√n.

### 4.3 Fair comparison methodology

* **Same target:** both methods use the same model object and the same
  mission time.
* **One oracle query** = one application of A or A† in one shot. A circuit
  Qᵐ·A uses 2m + 1 of them. For m = 0, 1, 2, 4, 8, 16 and 100 shots, that is
  100 × 68 = **6,800 oracle queries**. These counts were verified by counting
  the gates in every executed circuit.
* **Matched budget:** Monte Carlo gets exactly as many samples as QAE uses
  oracle queries (6,800).
* **Reproducible:** fixed seeds. Each quantum circuit gets an independent seed
  derived from the user's seed.
* **Runtime is reported separately** and is never mixed into the accuracy
  comparison.

## 5. Results

### 5.1 Single matched run (seed 2026, 6,800 model evaluations each)

| Quantity | Estimate | 95 % interval | Error vs exact |
|---|---|---|---|
| Exact model probability | 0.079184 | — | — |
| Classical Monte Carlo (6,800 samples) | 0.077206 | [0.071098, 0.083791] (Wilson) | 0.001978 |
| Quantum MLAE, simulated (6,800 queries) | 0.078668 | [0.076968, 0.080449] (approximate) | 0.000516 |

The QAE interval is about **3.6× narrower** than the MC interval at the same
number of model evaluations (width 0.003481 vs 0.012693). This is a single
run; section 5.3 averages over seeds.

For reference, Monte Carlo with 100,000 samples gives 0.080190
[0.078523, 0.081889], and with 1,000,000 samples 0.079239 [0.078711, 0.079770].

### 5.2 Cost of the quantum estimate

* 3 qubits (2 index + 1 failure qubit).
* Circuit depth per Grover power, after compiling to {rz, sx, x, cx} with
  **no** hardware connectivity limits: 25, 81, 137, 249, 473, 921.
  CX gate count: 5, 21, 37, 69, 133, 261.
* Simulator wall-clock time: QAE execution (build + simulate + estimate) took
  about 0.25 s, against about 0.0006 s for Monte Carlo, **roughly 400×
  slower**. Compilation used only to report circuit depth adds about 0.15 s
  and is shown separately. Timings are machine-dependent.

### 5.3 Error versus oracle queries (dashboard defaults: seeds 2026–2045, 100 shots)

RMSE over 20 fixed seeds at each budget:

| Largest power m | Oracle queries | QAE RMSE | MC RMSE (same budget) |
|---|---|---|---|
| 0 | 100 | 0.02207 | 0.03067 |
| 1 | 400 | 0.00833 | 0.01372 |
| 2 | 900 | 0.00566 | 0.00838 |
| 4 | 1,800 | 0.00239 | 0.00566 |
| 8 | 3,500 | 0.00271 | 0.00484 |
| 16 | 6,800 | 0.00041 | 0.00288 |

* **Trend:** MC follows its theoretical binomial standard error (≈ 1/√queries).
  QAE follows its asymptotic Cramér–Rao bound (≈ 1/queries) at most points.
* **The m = 8 point is an honest exception.** There, QAE's error is about 2×
  its bound. At this probability the likelihood often has two peaks, and the
  estimator sometimes picks the wrong one. The 30-seed study in
  `docs/qae_design.md` §8.2 and §8.3 shows the same effect.

### 5.4 Are the QAE intervals trustworthy?

A calibration study used 60,000 trials on seeds never used for tuning. It
covered 5 probabilities × 6 schedules, at 100 shots and powers up to m = 16:

* **Pooled coverage:** 56,602 / 60,000 = **0.943**, Wilson interval
  [0.941, 0.945], against a nominal 0.95.
* **Per setting:** 0.900 to 0.994. The worst setting, re-run with 20,000
  trials, gave 0.9286 [0.925, 0.932].
* **Verdict:** the interval is **approximate**. It is not a guaranteed 95 %
  interval.
* **Untested settings:** for other shot counts, or powers above m = 16,
  coverage is **unknown**, and the dashboard warns about this.

## 6. Interpretation

* **Query efficiency:** at an equal number of model evaluations, QAE's error
  shrinks faster than Monte Carlo's, as theory predicts. On this example and
  budget range it reached about 7× lower RMSE at 6,800 queries.
* **Runtime:** on a classical computer simulating the quantum circuit, QAE is
  hundreds of times *slower* than Monte Carlo. The result is about **sample
  efficiency on an ideal machine**, not speed.
* **The workflow:** the prototype shows that a reliability question can be
  encoded exactly as a quantum amplitude, which is a necessary first step for
  any future hardware study.

## 7. Limitations and threats to validity

1. **No hardware.** Everything quantum ran on a noiseless classical simulator:
   no gate errors, decoherence or readout errors.
2. **State-preparation and oracle cost.** The Fᵢ are computed classically and
   loaded as rotation angles. Any advantage would apply only to estimating the
   weighted sum. Loading arbitrary weights needs a number of gates that grows
   exponentially with the number of index qubits.
3. **Circuit depth.** Up to 921 layers at m = 16, even without connectivity
   constraints, which is far beyond what today's noisy devices can run
   reliably.
4. **Small problem.** 3 qubits and 3 η values. Classical methods are trivially
   fast here, and the exact answer is known.
5. **Approximate intervals.** Measured coverage is 0.90 to 0.99, not
   guaranteed, and calibration covers only the tested settings.
6. **Finite-sample estimator behaviour.** The MLAE estimate has small
   measurable bias. RMSE ranges from 0.83× to 1.48× the asymptotic bound
   depending on the schedule.
7. **Counting convention.** Matching oracle queries to Monte Carlo samples
   compares sample efficiency and ignores that one quantum query costs many
   gates.
8. **Software issue, worked around.** Qiskit 2.6.1 can crash when Qiskit is
   first imported on a thread that later exits. The dashboard runs all Qiskit
   work on one persistent thread.

## 8. Conclusion and future work

The prototype gives an exact, test-verified pipeline: a TDDB model with
uncertainty, solved three ways (exact, Monte Carlo and simulated QAE) on the
same target with a fair, query-matched comparison. QAE shows the expected
query-efficiency advantage on an ideal simulator, while being far slower in
wall-clock time and having only approximately calibrated intervals.

**Proposed future work (not done):**

* Run shallow schedules (m ≤ 2) on real quantum hardware with readout-error
  mitigation, and compare against the exact value.
* Use a noise-aware likelihood that models depolarising errors in MLAE.
* Calibrate the intervals better, for example with bootstrap or profile
  likelihood, and extend calibration to other shot counts.
* Use larger uncertainty models with more η values, and state-preparation
  methods that avoid exponential cost.
* Include acceleration models (voltage, temperature, oxide area) so that η
  depends on operating conditions.

---

## What is genuinely quantum in this prototype?

* **Quantum:** the algorithm and the circuits. The uncertainty is loaded as
  quantum amplitudes. A controlled rotation writes each Fᵢ into a failure
  qubit, so the probability of measuring 1 *is* p. Grover iterations amplify
  that amplitude, and the estimate comes from measurement statistics of
  circuits of increasing depth. These circuits are written in Qiskit and could
  be sent to a quantum computer unchanged.
* **Not quantum:**
  - The circuits were *executed* on a classical statevector simulator.
  - The Fᵢ values are computed classically.
  - The maximum-likelihood estimation is classical post-processing.
  - No quantum hardware was used, and no quantum speed-up was measured.

## Why is the simulator slower than classical Monte Carlo?

1. A statevector simulator stores all 2ⁿ amplitudes and applies every gate as
   classical linear algebra. Each circuit Qᵐ·A has up to 921 layers, and the
   simulator runs all of them for each of six circuits.
2. QAE adds circuit construction and a maximum-likelihood grid search after
   measurement.
3. One Monte Carlo sample is a few vectorised arithmetic operations, so 6,800
   samples take well under a millisecond.
4. Any QAE advantage is in the *number of queries*, which only matters when
   each query is expensive, and only on real quantum hardware. Simulating that
   hardware classically cannot be faster than the classical method it
   replaces.

## What would be needed to test this on real quantum hardware (proposed)

* **Access:** an account on a cloud quantum service, such as IBM Quantum
  through Qiskit Runtime. Credentials must be kept out of the repository; the
  `.gitignore` already excludes `.env` and `.streamlit/secrets.toml`.
* **Compilation:** transpilation to the device's native gates and
  connectivity. This increases depth beyond the numbers reported here.
* **Shallow schedules:** probably m ≤ 1 or 2, because depth grows with m and
  noise destroys deep circuits.
* **Error handling:** readout-error mitigation, and a likelihood that models
  noise. The current MLAE likelihood assumes a noiseless device.
* **Validation:** repeated runs against the exact p, reporting both
  oracle-query counts and real device time, including queueing.
* **Expectation:** with today's noise levels, the estimate would likely be
  less accurate than Monte Carlo at the same budget. That would itself be a
  useful, honest result.

---

## Likely judge and audience questions

**Q: Did you show a quantum speed-up?**
No. On the simulator QAE is roughly 400× slower than Monte Carlo. What we show
is that QAE needs fewer *oracle queries* for the same accuracy on an ideal,
noiseless machine, which is a query-complexity result, not a speed result.

**Q: Why is that query advantage interesting, then?**
If each model evaluation were expensive, and a fault-tolerant quantum computer
could evaluate it coherently, fewer queries would mean less total work. Error
∝ 1/queries vs 1/√samples means that 10× lower error costs about 10× more
queries for QAE but about 100× more samples for Monte Carlo.

**Q: How do you know the quantum circuit computes the right thing?**
The exact statevector of the preparation circuit gives P(failure qubit = 1) =
0.079184380877359, equal to the exact formula to 5.6 × 10⁻¹⁷. The
amplification identity sin²((2m+1)θ) is tested to 10⁻¹⁰ for several models.

**Q: Is the comparison fair?**
Both methods use the same model, mission time and seed. Monte Carlo receives
exactly as many samples (6,800) as QAE uses oracle queries, counted by
inspecting the executed circuits. Runtime is reported separately.

**Q: Are the QAE confidence intervals reliable?**
They are approximate. Across 60,000 calibration trials, coverage was 0.943
overall (0.900 to 0.994 depending on the setting), against a nominal 0.95.
Outside the tested settings (100 shots, m ≤ 16), coverage is unknown and the
dashboard says so.

**Q: Why does the QAE error jump at 3,500 queries (m = 8)?**
At this probability the likelihood often has two competing peaks for that
schedule, and the estimator sometimes chooses the wrong one. It shows up as
about 2× the theoretical bound in both our 20-seed and 30-seed studies. We
report it rather than hide it.

**Q: Why only 3 qubits?**
To keep everything exactly checkable and fast to simulate. The state-preparation
cost and simulation cost both grow exponentially with the number of index
qubits.

**Q: Could this run on real hardware today?**
The circuits could be submitted, but at m = 16 they are about 921 layers deep
before mapping to the device. Current noise would dominate. A realistic first
experiment would use m ≤ 2 with error mitigation. That is proposed, not done.

**Q: Where does the uncertainty in η come from in practice?**
From process variation between wafers and lots, and from fitting accelerated
test data. Here the three η values and weights are an illustrative example,
not measured data.

**Q: What would you do next?**
Noise-aware estimation, shallow-schedule hardware tests, better-calibrated
intervals, and models in which η depends on voltage and temperature.

---

## Reproducing every number

Install as described in the README, then, from the repository root:

| Numbers | Command |
|---|---|
| Model, exact p, Monte Carlo convergence | `python examples/milestone1_demo.py` |
| Curves and Monte Carlo figures (HTML in `build/plots/`) | `python examples/milestone2_plots.py` |
| Single QAE run, matched MC, circuit depths, 30-seed scaling | `python examples/milestone3_qae.py` |
| Calibration study (about 8 min) | `python examples/milestone3_ci_calibration.py` |
| Interval diagnostics (about 40 min) | `python examples/milestone3_ci_diagnostics.py` |
| Everything interactively, with the dashboard defaults | `streamlit run app/streamlit_app.py` |

The 20-seed scaling table above is what the dashboard's Comparison page
produces with its default settings. The 30-seed table in
`docs/qae_design.md` §8.4 uses seeds 0–29, so its numbers differ slightly
while showing the same trends.
