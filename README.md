# quantum-semiconductor-reliability

Quantum Semiconductor Reliability Explorer: time-dependent dielectric breakdown
(TDDB) reliability modelling that compares **classical Monte Carlo** with
**Quantum Amplitude Estimation (QAE)** on the **same** discrete uncertainty model.

## Status

| Milestone | Scope | State |
|---|---|---|
| 1 | Weibull model, discrete η-uncertainty model, exact reference, classical Monte Carlo, Wilson CI, tests | Done |
| 2 | Plotly visualizations | Planned |
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

Requires Python ≥ 3.10 (developed on 3.13). Milestone 1 depends only on NumPy
and SciPy; Qiskit, Streamlit and Plotly are **not** needed yet.

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

## Project layout

```
src/qsr/validation.py              shared input validation
src/qsr/models/tddb_weibull.py     Weibull CDF, reliability, inverse CDF
src/qsr/models/uncertainty.py      DiscreteEtaModel (shared by MC and QAE)
src/qsr/simulation/monte_carlo.py  classical Monte Carlo and Wilson interval
examples/milestone1_demo.py        demonstration script
docs/qae_design.md                 Milestone 3 QAE mathematical design
tests/                             pytest suite
```

## Secrets

Never commit API keys or tokens. Future credentials (for example an IBM Quantum
token) belong in a local `.env` file or in `.streamlit/secrets.toml`. Both are
already listed in `.gitignore`.