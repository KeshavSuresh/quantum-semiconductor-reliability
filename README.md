# quantum-semiconductor-reliability

Quantum Semiconductor Reliability Explorer: time-dependent dielectric breakdown
(TDDB) reliability modelling that compares **classical Monte Carlo** with
**Quantum Amplitude Estimation (QAE)** on the **same** discrete uncertainty model.

## Status

| Milestone | Scope | State |
|---|---|---|
| 1 | Weibull model, discrete η-uncertainty model, exact reference, classical Monte Carlo, Wilson CI, tests | Done |
| 2 | Plotly visualizations of the shared model (curves, Monte Carlo vs exact) | Done |
| 3 | Qiskit QAE on the shared model ([design](docs/qae_design.md)) | Planned |
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
need only NumPy and SciPy. Plotly is an optional `viz` extra for the figures.
Qiskit and Streamlit are **not** needed yet.

`requirements-dev.txt` installs everything used in development
(`-e .[dev,viz]`). To install only what you need:

```bash
pip install -e .            # model + Monte Carlo only
pip install -e ".[viz]"     # + Plotly figures
```

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

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

## Project layout

```
src/qsr/validation.py              shared input validation
src/qsr/models/tddb_weibull.py     Weibull CDF, reliability, inverse CDF
src/qsr/models/uncertainty.py      DiscreteEtaModel (shared by MC and QAE)
src/qsr/simulation/monte_carlo.py  classical Monte Carlo and Wilson interval
src/qsr/visualization/plots.py     Plotly figure builders and their data functions
src/qsr/visualization/theme.py     light/dark colour themes and shared layout
examples/milestone1_demo.py        Milestone 1 demonstration script
examples/milestone2_plots.py       Milestone 2 figures → build/plots/*.html
docs/qae_design.md                 Milestone 3 QAE mathematical design
tests/                             pytest suite
```

## Secrets

Never commit API keys or tokens. Future credentials (for example an IBM Quantum
token) belong in a local `.env` file or in `.streamlit/secrets.toml`. Both are
already listed in `.gitignore`.