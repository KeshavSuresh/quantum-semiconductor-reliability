# Live demo guide (5–7 minutes)

A step-by-step script for demonstrating the dashboard with its **default
settings**. Every expected number below was checked in a real browser against
a fresh install, and the defaults are guarded by an automated test
(`test_default_settings_reproduce_the_documented_demo`). Wall-clock times vary
by machine; all other numbers are deterministic.

## Before the demo (once, about 5 minutes)

You need Python 3.11 or newer. From the repository root:

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

(On Windows: `.venv\Scripts\activate`.)

```bash
pip install -e ".[app,quantum,dev]"
```

Check that everything works (about a minute):

```bash
pytest -q
```

Start the dashboard and open http://localhost:8501:

```bash
streamlit run app/streamlit_app.py
```

**Warm-up:** click through every page once and press each Run button before
the audience arrives. A cold start took about 2.5 s for the first page and
about 1.4 s for the first quantum run on the test machine. Then press **Reset
to example model** in the sidebar, or restart the dashboard, so the defaults
are back.

**No internet, accounts or credentials are needed.** Everything runs locally.

## The demo, step by step

| Time | Page and action | What the audience sees | What to say |
|---|---|---|---|
| 0:00–0:45 | **Overview** | The TDDB explanation, the two formulas, and **Exact model probability 0.079184** | "A gate oxide slowly degrades until it breaks. We ask what fraction of devices fails by 5,000 hours. Breakdown times follow a Weibull law, but the characteristic life η is uncertain, so we average over three possible values." |
| 0:45–1:15 | **Sidebar** | The η table (10,000 / 20,000 / 40,000 h) with raw and normalised weights 0.2 / 0.5 / 0.3, and each component's F(t) (0.2212, 0.0606, 0.0155) | "These are the three possible lifetimes and how likely each is. The exact answer is their weighted average, 0.079184. That is our ground truth." |
| 1:15–2:00 | **Classical** → **Run Monte Carlo** (100,000 samples) | **0.080190**, Wilson interval **[0.078523, 0.081889]**, 8,019 failures, and a convergence plot | "Monte Carlo simulates devices at random. It is close, the interval contains the exact value, and the convergence plot shows the error shrinking like 1 over √n." |
| 2:00–3:15 | **Quantum** → **Run QAE on the simulator** (defaults: schedule 6, 100 shots) | Powers m = 0, 1, 2, 4, 8, 16 and **6,800 oracle queries**; estimate **0.078668**, interval **[0.076968, 0.080449]**, the calibrated-interval caption, a green circuit check, 3 qubits, and depths up to 921 | "The model is loaded into 3 qubits so that the chance of measuring 1 is exactly p. The green check confirms this to 10⁻¹⁶. Grover steps amplify that chance; we run six circuits, 100 shots each, and estimate p from the counts. Note the interval is labelled approximate, and the circuits are deep, up to 921 layers." |
| 3:15–3:45 | *(optional)* Move **Shots per circuit** to 30 | A notice that the settings are outside the calibration study, and the old result marked **out of date** | "The dashboard knows where its interval was tested and warns us outside that range. It also flags results computed with different settings." Move shots back to 100: both messages disappear. |
| 3:45–4:45 | **Comparison** → **Run matched comparison** | Table: exact 0.079184, MC **0.077206**, QAE **0.078668**, with 6,800 model evaluations each; QAE error 0.000516 vs MC 0.001978; runtime block, where MC is far faster | "Same model, same seed, same budget: 6,800 evaluations each. QAE is about 4× closer here and its interval about 3.6× narrower. But look at the clock: the simulator is hundreds of times *slower* than Monte Carlo." |
| 4:45–5:45 | **Comparison** → **Run scaling study** (defaults: up to m = 16, 20 seeds, about 13 s) | A log-log plot of error vs oracle queries, with measured points and dashed theory lines | "Averaged over 20 seeds, Monte Carlo follows 1/√queries and QAE follows the faster 1/queries, about 7× lower error at 6,800 queries. One point, m = 8, is worse than theory: the likelihood has two peaks there, and we show that honestly." |
| 5:45–6:45 | **Limitations** | The red "No quantum speed-up is shown" box and the list of limitations | "This is an ideal, noiseless simulation of 3 qubits. The quantum advantage is in query counts, not runtime. Real hardware noise and circuit depth are the open problems, and the intervals are approximate. Next steps would be shallow-circuit tests on real hardware." |

## Expected numbers (defaults)

* **Exact:** p = 0.079184, R = 0.920816.
* **Monte Carlo, 100,000 samples:** 0.080190, interval [0.078523, 0.081889].
* **QAE** (m = 0, 1, 2, 4, 8, 16; 100 shots; seed 2026): 0.078668, interval
  [0.076968, 0.080449], one region, inside the calibrated settings. 6,800
  oracle queries and 3,100 Grover steps.
* **Matched Monte Carlo, 6,800 samples:** 0.077206, interval
  [0.071098, 0.083791].
* **Scaling study** (seeds 2026–2045): QAE RMSE goes from 0.0221 to 0.00041,
  and MC RMSE from 0.0307 to 0.00288, as queries go from 100 to 6,800.
* **Simulator timing (machine-dependent):** QAE execution about 0.25 s, MC
  about 0.0006 s, so a ratio of a few hundred times.

## If something goes wrong

* **A number differs:** check the sidebar wasn't changed. Press **Reset to
  example model**, or restart the dashboard.
* **A typed value lands in the wrong cell of the η table:** after switching
  pages, the first click only focuses the table, as the caption under it says.
  Click the cell again.
* **The dashboard won't start:** run the same story from the terminal.

  ```bash
  python examples/milestone1_demo.py
  ```

  ```bash
  python examples/milestone3_qae.py --quick
  ```

  This prints the exact value, Monte Carlo, QAE, the matched comparison and a
  short scaling table, and writes `build/plots/qae_error_vs_queries_light.html`.
* **Qiskit isn't installed:** the Overview, Classical and Limitations pages
  still work. The Quantum and Comparison pages show the install command.

For background, answers to likely questions and the full results, see
[`docs/presentation.md`](presentation.md).
