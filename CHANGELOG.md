# Changelog

This project was built in phases. Early work (phase1-12, in `analysis_notebooks/`
and the `phase*` outputs) established six baseline models over hand-run
notebooks. The entries below cover the second pass, which made the repo
actually reproducible from a clone and re-evaluated the baseline-vs-V2 claims
honestly.

## Unreleased

- **Customer 360 combined view.** `analysis/customer_360.py` joins churn,
  uplift, segment and recommendations into one recommended action per
  customer. Uplift and recommendations are recomputed for the full customer
  population (not read from a sample), giving real signal overlap instead of
  near-zero. See [docs/customer_360.md](docs/customer_360.md).
- **Build overwrite guard.** `src/data/build.py` and `src/data/generate.py`
  refuse to overwrite an existing `data/raw` unless `--force` is passed,
  after an earlier `--scale sample` run silently clobbered a full dataset.
- Fixed a Windows console crash in `src/models/train_uplift.py` (a print
  statement used characters cp1252 can't encode).
- Added `.pre-commit-config.yaml` (ruff lint + format) and a coverage
  threshold on `src/features`/`src/models` in CI.
- Added `make eval`, an offline harness that reruns the phase runners against
  the built marts and checks the regenerated metrics against the committed
  `outputs/*_metrics.json`/`*_model_comparison.csv` within tolerance.

## PR #27 — Deterministic synthetic data generator and one-command build

`data/` was gitignored with no generator, so a clone could not reproduce
anything. Added `src/data/generate.py` (seeded, scale presets
`sample`/`default`/`full`) and `src/data/build.py` (`generate` then materialise
marts in one command). CI now builds the sample scale and runs the full test
suite against it on every push.

## PR #26 — Reproducibility + honest evaluation

- Committed the `src/data` package (mart loaders, schemas, validators, phase
  runners) that tests and notebooks already imported but that did not exist
  in the repo — `pytest` could not even collect before this.
- Added CI (`ruff`, `pytest --cov`, a clean-install import check).
- **Churn:** dropped `customer_segment_seed` (a data-generating seed leaking
  into features), ran a like-for-like baseline-vs-V2 comparison on the
  identical split, and found logistic regression beats the LightGBM V2 model
  (ROC-AUC 0.844 vs 0.812, paired-bootstrap confirmed). Added isotonic
  calibration — raw LightGBM was predicting mean churn probability 0.44
  against a 0.15 actual rate.
- **Uplift:** filled in the missing top-3-decile figure and stated the
  targeting decision (target deciles 1-3, ~62 incremental conversions per
  1,000 treated vs ~44 untargeted).
- **Forecast:** added naive-seasonal and naive-last-week benchmarks
  (`analysis/forecast_naive_baseline.py`) alongside the RF baseline, and
  stated plainly that the RF-vs-V2 delta is modest (~1.4% MAE) even though
  V2 beats the naive benchmarks by ~23%.
- **Anomaly:** corrected doc wording — `returns_hidden_labels.csv` is a
  synthetic generator seed, not manually-labelled ground truth, and flagged
  that the resulting AP is optimistic.
- Added `lightgbm`/`shap` to `requirements.txt` (four modules imported
  `lightgbm` without it being declared).

## Earlier work (phase1-12)

The original baseline build: six workstreams (churn, uplift, forecast,
segmentation, recommendations, returns-anomaly) developed in
`analysis_notebooks/`, with a `phase12_pipeline_registry.csv` deployment
design. Superseded by the V2 layer in `src/` and the re-evaluations above;
kept for history but not the source of truth for headline numbers.
