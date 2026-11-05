# Dashboard UX redesign: audit, decisions and results

This records the design review of the Streamlit dashboard: the problems found
in the original interface, the visual direction chosen, and what changed.
The scientific code was not touched. Every number shown is produced by the
same functions as before, and the tests check that.

## 1. Audit of the original interface

The audit was made from screenshots of every page at 1400 px and 800 px
width, with every Run button pressed (default settings).

| # | Problem | Where | Why it matters |
|---|---|---|---|
| A1 | Emoji icons on every page link and alert (🏠 🎲 ⚛️ ⚖️ ⚠️ ℹ️ ✅ 🚫) | Navigation, alerts | Reads as a template; the icons carry no information |
| A2 | Streamlit's default red accent on buttons and sliders | All pages | Red normally signals an error; the tool had no visual identity |
| A3 | Very large page titles, about 40 px | All pages | Titles outweigh the results they introduce |
| A4 | The same bordered card of four large numbers at the top of every page | All pages | Repetitive; mixes the key results (p, R) with trivia (K / β) |
| A5 | At 800 px those cards truncate their values to "0.07…", "0.92…", "500…" | All pages | The most important numbers become unreadable |
| A6 | Every result shown as a large `st.metric`: reference and estimates look identical; classical and quantum are not distinguished | Classical, Quantum, Comparison | No visual semantics; intervals were pushed to secondary text |
| A7 | Metric labels truncated ("QAE execution (build + simulate + es…", "Oracle queries (…") | Quantum, Comparison | Units and meaning lost |
| A8 | The comparison's centrepiece was a raw table with "None" cells and a 0.000000 error for the exact value; no picture of estimates against the reference | Comparison | The main result of the project was visually buried |
| A9 | Accuracy, query budget and wall-clock time given equal visual weight, with up to six stacked grey captions | Comparison | Hard to tell which numbers answer which question |
| A10 | Inconsistent colours: Monte Carlo blue in one figure and orange in another; η components drawn in the same blue and orange | Charts | Colour could not be trusted to mean "classical" or "quantum" |
| A11 | The same "no speed-up" caption repeated on four pages, plus long explanatory paragraphs | Overview, Quantum, Comparison | Important caveats drowned in repetition |
| A12 | Scientific result and circuit diagnostics interleaved; diagnostics table with long headers that overflow at 800 px | Quantum | Hard to separate what was estimated from what it cost |
| A13 | Sidebar is long: the η values appear twice; seed and label-only time unit sit among the physics inputs; mission time shown as "5000.00" | Sidebar | Weak hierarchy among inputs |
| A14 | Limitations led by a red error box with an emoji | Limitations | Alarmist styling for what should be calm, factual statements |

## 2. Visual direction

The goal is engineering-software restraint: the look of a measurement
instrument or a lab notebook figure, not a product dashboard.

* **Colour carries meaning, and only three colours do:**
  - **reference** (exact value): near-black ink, dashed in figures;
  - **classical** Monte Carlo: orange;
  - **quantum** MLAE (simulated): blue.

  These are the first two slots of the project's colour-blind-validated
  palette. Every figure, table badge and legend uses them consistently.
  Everything else is neutral grey. η components are drawn in ink with
  distinct dash patterns and direct labels, so they never borrow the
  classical or quantum colours.
* **Neutral interface accent.** Buttons and sliders use a dark slate instead of
  red, which keeps red-like warm hues free for real warnings.
* **Typography.** Smaller, weight-led headings. Numbers use tabular figures
  so columns of values align. The bundled sans-serif font needs no network,
  so the demo works offline.
* **Results as readouts, not cards.** Compact tables with a coloured badge
  per row (Reference / Classical / Quantum · simulated), the estimate, the
  full interval with its width, the error against the exact value, and the
  budget. Nothing is truncated; tables wrap on narrow screens.
* **One statement of each caveat, placed where it applies.** Warnings are
  kept for genuine conditions (stale results, untested calibration settings,
  ambiguous intervals). They are never used for decoration.
* **Light theme only.** One fixed, high-contrast light theme, so figure
  colours stay identical on screens and projectors. Streamlit 1.49, the
  minimum supported version, cannot define separate light and dark themes.

## 3. Changes

### Shared foundation

| Change | Fixes | Where |
|---|---|---|
| Theme: light base, slate accent `#33414c`, warm off-white surfaces, thin borders, small radii, 15 px base font, smaller 600-weight headings | A2, A3 | `.streamlit/config.toml` |
| Static stylesheet: readable measure (1120 px), tabular numerals, thin-rule tables without zebra stripes, small uppercase section labels ("1 · Setup", "2 · Result") | A3, A6 | `app/components/style.py` |
| Readout tables: one row per quantity with a Reference / Classical / Quantum · simulated badge, estimate, full interval, width, error vs exact and budget with its unit (samples or oracle queries) | A4–A7 | `src/qsr/dashboard/readout.py` (pure, tested) |
| Fact tables for costs and times, with a non-breaking space between number and unit | A7, A9 | `readout.facts_table`, `readout.seconds` |
| Semantic figure colours: reference = ink, classical = orange, quantum = blue, the same in every figure; η components in ink with distinct dash patterns | A10 | `src/qsr/visualization/theme.py`, `plots.py`, `scaling.py` |
| Legends anchored to the bottom of the figure, so a legend that wraps on a narrow screen grows into reserved space instead of over the title | found in testing | `theme.apply_layout` |
| Text-only navigation with method names as page titles; no emoji on any link or alert | A1, A14 | `app/streamlit_app.py`, all views |
| One compact context line replaces the four-number card at the top of each page | A4, A5 | `components/state.model_summary` |
| Sidebar: grouped inputs with units in labels (`Mission time t [h]`, `η [h]`), run settings (time unit, seed) in a collapsed expander, a fixed-width "Weights in use" table that fits without truncation | A13 | `components/model_sidebar.py` |

### Pages

* **Overview.** The research question as a quotation, the model next to the
  current configuration (a reference readout of p and R), and a three-row
  table explaining how each estimate is obtained and what uncertainty it
  reports.
* **Classical Monte Carlo.** Four numbered steps: the exact reference curves,
  the sampling experiment (what one sample does and why error falls as
  1/√N), the result readout with the exact value directly above the
  estimate, and the convergence figure.
* **Quantum amplitude estimation.** Setup (schedule, shots, the oracle-query
  arithmetic), then the result readout with its calibration notice and the
  circuit check, then two clearly labelled *diagnostics* sections: circuit
  cost and simulator wall-clock time. The per-circuit table moved into an
  expander. (A12)
* **Comparison.** The new centrepiece is an interval figure: both estimates
  with their intervals against a dashed vertical line at the exact value, on
  a linear axis that always contains every interval. A three-row readout
  follows; the raw data table is in an expander. Query budget and wall-clock
  time are side-by-side tables in a separate *Cost* section, so accuracy,
  budget and runtime are never read as one score. (A8, A9)
* **Limitations.** A calm scope statement and a *shows / does not show* table
  replace the red error box. The detailed sections are unchanged. (A14)

### Deliberately kept

* **The no-speed-up caption** still appears once at the foot of the
  Overview and of each page that shows a quantum result, and on Limitations. A reader may open any
  page first, and the tests require it next to quantum results. The long
  repeated paragraphs around it were removed, so A11 is reduced, not
  eliminated.
* **Button labels, widget keys and default settings** are unchanged, because
  the demo guide, the tests and the documented numbers depend on them.
* **Warnings** remain for the three genuine conditions: stale results,
  settings outside the calibration study, and a multi-region likelihood
  interval.

## 4. Before and after

Screenshots of every page were taken in Chromium at 1400 px and 800 px
width, before the redesign and after it, with every Run button pressed at
the default settings. They were used for review and are not stored in the
repository. To reproduce them, run `streamlit run app/streamlit_app.py` and
open each page at both widths.

What the after screenshots show, and how it was checked:

* **Numbers unchanged.** Exact 0.079184; Monte Carlo 0.080190
  [0.078523, 0.081889]; QAE 0.078668 [0.076968, 0.080449]; matched Monte
  Carlo 0.077206 [0.071098, 0.083791]; 6,800 evaluations each. These are
  asserted by `tests/test_dashboard_app.py`, which now reads the readout
  tables instead of metric widgets, and by a browser click-through of the
  documented demo.
* **No truncation at 800 px.** Readouts wrap instead of cutting values; the
  sidebar weights table fits; times never split from their unit.
* **Interval figure.** Its precondition, low ≤ estimate ≤ high, was checked
  for 432 real dashboard estimates (3 mission times, giving p from near 0 to
  near 1; 4 schedule lengths; 3 shot counts; 6 seeds), with no violations.
  The page still falls back to the tables if it ever fails.
* **Defects found and fixed while testing:**
  - legends wrapping over figure titles at 800 px;
  - interval-figure row labels clipped at the left edge;
  - the run button truncated in a three-column row;
  - a cramped three-column runtime table;
  - "0.0007 s" splitting across two lines.
