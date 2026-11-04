# quantum-semiconductor-reliability

Quantum Semiconductor Reliability Explorer: time-dependent dielectric breakdown
(TDDB) reliability modelling that compares **classical Monte Carlo** with
**Quantum Amplitude Estimation (QAE)** on the **same** discrete uncertainty model.

## Status

| Milestone | Scope | State |
|---|---|---|
| 1 | Weibull model, discrete η-uncertainty model, exact reference, classical Monte Carlo, Wilson CI, tests | Done |
| 2 | Plotly visualizations of the shared model (curves, Monte Carlo vs exact) | Done |
| 3 | Qiskit Quantum Amplitude Estimation on the shared model ([design and results](docs/qae_design.md)) | Done |
| 4 | Streamlit application | Planned |
| 5 | CI and documentation | Planned |

## Model (Milestone 1)

Weibull TDDB failure probability by time `t`:

    F(t | β, η) = 1 − exp(−(t / η)^β)

Discrete uncertainty in the characteristic life η, with shared shape β:

    P(η = η_i) = w_i,   w_i ≥ 0,   Σ w_i = 1
    p(t) = Σ_i w_i · F(t | β, η_i)        (exact reference probability)

Classical Monte Carlo samples from the same model. It picks `η_i` with
probability `w_i`, draws a Weibull failure time by inverse-CDF sampling, counts
failures with `T ≤ t`, and reports `p̂ = k/n` together with a 95 % Wilson score
confidence interval. A fixed integer `seed` makes every run reproducible.

**Units:** all times (`η_i` and mission time `t`) must use the same unit. The
examples use **hours**. β and the weights are dimensionless.

## Setup

Requires Python ≥ 3.10 (developed on 3.13). The core model and Monte Carlo
need only NumPy and SciPy. Plotly (`viz`) and Qiskit (`quantum`) are optional
extras. Streamlit is **not** needed yet.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

`requirements-dev.txt` installs everything used in development
(`-e .[dev,viz,quantum]`). To install only what you need:

```bash
pip install -e .                    # model + Monte Carlo only
pip install -e ".[viz]"             # + Plotly figures
pip install -e ".[quantum]"         # + Qiskit amplitude estimation (tested with Qiskit 2.6.1)
```

Without Qiskit, the classical code and its tests still work, and the quantum
tests are skipped.

Run the tests:

```bash
pytest -q
```

Run the demonstration (exact p vs Monte Carlo at n = 10³ … 10⁶):

```bash
python examples/milestone1_demo.py
```

Generate the Milestone 2 figures as standalone HTML files in `build/plots/`
(git-ignored), in either light or dark mode:

```bash
python examples/milestone2_plots.py
```

```bash
python examples/milestone2_plots.py --dark
```

Open the `.html` files in any browser; hovering shows exact values. Add
`--cdn` for smaller files that load plotly.js from the internet, or use
`--out-dir PATH` to choose a different folder.

Run the Milestone 3 quantum demonstration, which takes about 30 s and needs
the `quantum` and `viz` extras:

```bash
python examples/milestone3_qae.py
```

```bash
python examples/milestone3_qae.py --quick
```

It prints the state-preparation check, a single budget-matched QAE vs Monte
Carlo run (estimates, CIs, oracle calls, circuit depths, timings) and an
error-scaling table over 30 fixed seeds (0–29). It also writes
`build/plots/qae_error_vs_queries_light.html`.

Reproduce the confidence-interval calibration study, which takes about 8
minutes. Add `--quick` for a short preliminary run:

```bash
python examples/milestone3_ci_calibration.py
```

## Usage

```python
from qsr.models.uncertainty import DiscreteEtaModel
from qsr.simulation.monte_carlo import estimate_failure_probability

# Time unit: hours
model = DiscreteEtaModel(beta=2.0, etas=[1e4, 2e4, 4e4], weights=[0.2, 0.5, 0.3])
exact = model.failure_probability(5e3)                       # 0.079184...
mc = estimate_failure_probability(model, mission_time=5e3, n_samples=100_000, seed=2026)
print(exact, mc.estimate, (mc.ci_low, mc.ci_high))
```

### Figures (Milestone 2)

Each builder takes the same `DiscreteEtaModel` and returns a
`plotly.graph_objects.Figure` without showing or saving it. The same object
can be passed to `fig.show()`, `fig.write_html(...)`, or later to
`streamlit.plotly_chart(fig)`.

```python
from qsr.visualization import (
    DARK, plot_failure_and_reliability, plot_component_curves,
    plot_mc_vs_exact, run_mc_convergence, mc_results_table,
)

fig1 = plot_failure_and_reliability(model, time_unit="h", mission_time=5e3)
fig2 = plot_component_curves(model, time_unit="h", mission_time=5e3)   # F(t|β,ηᵢ) + mixture
results = run_mc_convergence(model, 5e3, [1_000, 10_000, 100_000], seed=2026)
fig3 = plot_mc_vs_exact(model, results, time_unit="h", theme=DARK)    # p̂ ± Wilson CI vs exact
rows = mc_results_table(model, results)                              # same numbers as a table
```

| Figure | Shows |
|---|---|
| `plot_failure_and_reliability` | Mixture F(t) and R(t) = 1 − F(t) on one probability axis |
| `plot_component_curves` | Each F(t \| β, ηᵢ) plus the weighted mixture Σ wᵢ F(t \| β, ηᵢ) |
| `plot_mc_vs_exact` | Monte Carlo p̂ with asymmetric Wilson CIs vs sample size (log scale), exact p as a reference line |

`time_unit` only labels the axes; values are never converted. The colour
palette uses a fixed order that was checked for colour-vision deficiency.
Lines also carry a legend and direct labels, so identity never relies on
colour alone. Models with more than 8 components are drawn as one muted group.

### Quantum Amplitude Estimation (Milestone 3)

The circuit prepares

    A|0⟩ = Σᵢ √wᵢ |i⟩ ⊗ [ √(1 − Fᵢ) |0⟩ + √Fᵢ |1⟩ ],   Fᵢ = F(t | β, ηᵢ)

using the model's own weights and `component_failure_probabilities(t)`, so
P(failure ancilla = 1) = Σᵢ wᵢ Fᵢ, the exact mixture probability that classical
Monte Carlo also estimates. It is estimated with Maximum-Likelihood Amplitude
Estimation (Suzuki et al. 2020), implemented on core Qiskit with a statevector
sampler. Each circuit gets its own seed spawned from one user seed, so runs are
reproducible and counts are independent across Grover powers.

**Query convention:** one oracle query is one application of `A` or `A†` in one
shot. A circuit `Q^m A` uses `2m + 1` of them, so the total is
`shots × Σ(2m_k + 1)`; for example, 100 × 68 = 6,800 for m = 0, 1, 2, 4, 8, 16.
Classical Monte Carlo gets exactly that many samples. Grover iterates and
phase-oracle calls (100 × 31 = 3,100) are reported separately and not added in.

```python
from qsr.quantum import compare_with_monte_carlo, error_scaling_study, exponential_schedule
from qsr.visualization import plot_error_vs_queries

c = compare_with_monte_carlo(model, 5e3, seed=2026, evaluation_schedule=exponential_schedule(6), shots=100)
print(c.exact, c.qae.estimate, (c.qae.ci_low, c.qae.ci_high), c.qae.state_preparation_calls)
print(c.mc.estimate, c.mc.n_samples)              # classical MC with the same budget and seed
rows = error_scaling_study(model, 5e3, seeds=range(30))
fig = plot_error_vs_queries(rows)                  # RMSE vs oracle calls, log-log
```

**How to read the results, and limitations:**

- **Query complexity is not runtime.** "Oracle calls" counts uses of `A` or
  `A†` (QAE) and samples (MC). The dashed curves are theory: the asymptotic
  Cramér–Rao bound (QAE, ~1/calls) and the binomial standard error
  (MC, ~1/√calls). The measured RMSE follows both trends.
- **Wall-clock times are simulator times.** QAE runs on a classical
  statevector simulator and was roughly 600× slower per run than classical MC
  here (≈ 0.5 s vs ≈ 0.001 s at 13,300 calls). Timings vary between runs and
  machines. **No practical quantum speed-up is shown or claimed.**
- The Fᵢ are computed classically and loaded as rotation angles. Any
  advantage would apply only to estimating the weighted sum, not to computing Fᵢ.
- Circuits are noiseless, with no hardware topology or error correction.
  Reported depths use the basis {rz, sx, x, cx} with no connectivity
  constraints, and they grow linearly with the Grover power (921 at m = 16).
  Such depths would be far beyond current noisy hardware.
- The confidence intervals are **approximate** 95 % likelihood-ratio intervals,
  **not** guaranteed 95 %.
  - Over 60,000 calibration trials on fresh seeds, pooled coverage was 0.943
    (Wilson [0.941, 0.945]). Individual cells ranged from 0.900 to 0.994.
  - The worst cell re-run with 20,000 trials gave 0.9286 [0.925, 0.932].
  - Most failures are χ² calibration error at 100 shots. Wrong likelihood peaks
    add more where the likelihood is multimodal. No numerical or optimiser errors
    were found in 2,739 brute-force checks.
  - When `n_ci_regions > 1`, the reported interval is the hull of separate
    regions: conservative, and wider than the set the data support.
  - Full counts are in `docs/qae_design.md` §8, from
    `python examples/milestone3_ci_calibration.py` and
    `python examples/milestone3_ci_diagnostics.py`.
- The finite-sample MLE has small measurable bias. Its RMSE was 0.83–1.48× the
  Cramér–Rao bound depending on the schedule, so treat the bound as an
  asymptotic reference, not a guarantee.

## Project layout

```
src/qsr/validation.py              shared input validation
src/qsr/models/tddb_weibull.py     Weibull CDF, reliability, inverse CDF
src/qsr/models/uncertainty.py      DiscreteEtaModel (shared by MC and QAE)
src/qsr/simulation/monte_carlo.py  classical Monte Carlo and Wilson interval
src/qsr/visualization/plots.py     Plotly figure builders and their data functions
src/qsr/visualization/theme.py     light/dark colour themes and shared layout
examples/milestone1_demo.py        Milestone 1 demonstration script
src/qsr/visualization/scaling.py   error-vs-oracle-calls figure (no Qiskit import)
src/qsr/quantum/state_preparation.py  circuit A: weight loading + uniformly controlled RY
src/qsr/quantum/amplitude_estimation.py  MLAE estimator, query accounting, depth, timing
src/qsr/quantum/comparison.py      budget-matched QAE vs Monte Carlo and scaling study
examples/milestone2_plots.py       Milestone 2 figures → build/plots/*.html
examples/milestone3_qae.py         Milestone 3 QAE demonstration
examples/milestone3_ci_calibration.py  CI coverage study (60,000 + 1,500 trials)
examples/milestone3_ci_diagnostics.py  CI failure diagnostics, sampler comparison, bias study
docs/qae_design.md                 QAE design, implementation decisions and results
tests/                             pytest suite
```

## Secrets

Never commit API keys or tokens. Future credentials (for example an IBM Quantum
token) belong in a local `.env` file or in `.streamlit/secrets.toml`. Both are
already listed in `.gitignore`.