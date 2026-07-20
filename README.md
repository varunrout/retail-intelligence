# Retail Growth Intelligence System

End-to-end retail analytics and ML over one shared DuckDB mart layer, covering six
decisions for an omnichannel retailer: churn, uplift/retention targeting, demand
forecasting, segmentation, recommendations and returns-anomaly detection. The data
is **synthetic** (generated for the project); every metric below is measured on it.

![ci](https://github.com/varunrout/retail-intelligence/actions/workflows/ci.yml/badge.svg)

## Headline results (read this first)

The project has a baseline layer (phases 6-11) and a second "V2" iteration
(LightGBM/X-learner/etc.). The honest position after a like-for-like re-evaluation:

- **Churn — a logistic regression beats the LightGBM V2 model.** On the identical
  time-ordered split, same features, seed leakage removed, logistic regression
  scores ROC-AUC 0.844 vs LightGBM's 0.812; a paired bootstrap puts the lift at
  -0.033 [-0.038, -0.027], i.e. LightGBM is *worse* and the gap is real. The
  earlier "V2 vs baseline" file compared models across three different regimes at
  once and is not a valid comparison. See
  [docs/churn_incremental_lift_and_reconciliation.md](docs/churn_incremental_lift_and_reconciliation.md).
- **Churn probabilities are miscalibrated.** Raw LightGBM predicts a mean churn
  probability of 0.44 against a 0.15 actual rate, so the 0.50/0.70 offer thresholds
  are not meaningful on raw scores. Isotonic calibration cuts Brier from 0.197 to
  0.141. Calibrate before thresholding.
- **Uplift — target the top 3 deciles.** X-learner V2 ties the baseline on overall
  ATE but concentrates uplift better (Qini 745 vs 502, top-3 uplift 0.0625 vs
  0.0398). Targeting the top 3 deciles yields ~62 incremental conversions per 1,000
  treated vs ~44 untargeted. Past decile 3 the marginal uplift falls below the
  population ATE. See [docs/uplift_targeting_decision.md](docs/uplift_targeting_decision.md).

A logistic regression winning and a boosted model failing a fair test is reported
deliberately: a rigorous negative result is more useful than a comparison rigged
across moving parts.

## Business questions

1. Which customers are likely to churn soon?
2. Which customers should receive a retention offer for incremental impact?
3. What products should be recommended to each customer?
4. What will future demand look like at product, store and week level?
5. Which orders, stores or SKUs show unusual (return-risk) behaviour?
6. What customer segments exist, and what should the business do about them?

## Stack

- **Data store:** DuckDB over synthetic raw CSVs, built into six marts via `sql/mart_*.sql`.
- **Python:** 3.11+. pandas, numpy, scikit-learn, scipy, statsmodels, **lightgbm**, **shap**.
- **Models (V2):** LightGBM churn, X-learner uplift, LightGBM-Tweedie forecast,
  nested k-means segmentation, hybrid SVD recsys, supervised returns-anomaly.
- **CI:** ruff lint + format, pytest with coverage, and a clean-install import check
  (`.github/workflows/ci.yml`).

## Reproduce

Data (raw CSVs and built marts) is synthetic and gitignored. A committed
generator rebuilds it from scratch, so a fresh clone runs end to end:

```bash
pip install -r requirements.txt
export PYTHONPATH=.

# Build reproducible data from nothing: generate raw tables -> DuckDB marts.
make build            # full default scale  (or: python -m src.data.build)
make sample           # tiny dataset, seconds, used by CI

# Re-run a model end to end (marts -> features -> model -> outputs/)
python -m src.data.run_phase_churn_v2
python -m src.data.run_phase_uplift_v2

# The honest re-evaluations (each also has a --smoke mode needing no data)
python -m analysis.churn_incremental_lift        # matched churn baseline vs V2 + CI
python -m analysis.churn_calibration             # reliability + calibrated Brier
python -m analysis.uplift_targeting_decision     # who gets an offer
python -m analysis.forecast_naive_baseline       # naive benchmarks

make test             # pytest (data tests run once marts are built)
make lint             # ruff check + format check
make eval             # retrain all six V2 models, check metrics vs committed outputs/

# Optional: catch lint/format drift before it reaches CI
pip install pre-commit
pre-commit install
```

The generator (`src/data/generate.py`) is deterministic and scalable
(`--scale sample|default|full`, `--seed`). It plants real signal: engagement
drives churn recency, a latent persuadability drives heterogeneous uplift, weekly
demand has seasonality and trend, and a small fraction of returns are abuse
(~0.7%, written to `returns_hidden_labels.csv`). CI builds the sample dataset and
runs the full suite on every push, so reproducibility is enforced, not just
claimed.

## Results to files

| Result | File |
|---|---|
| Churn matched baseline comparison | `outputs/churn_incremental_lift_metrics.csv` |
| Churn paired-bootstrap lift + verdict | `outputs/churn_incremental_lift_paired_bootstrap.csv` |
| Churn calibration (Brier, reliability) | `outputs/churn_calibration_brier.csv`, `outputs/churn_calibration_reliability.png` |
| Uplift decile table + Qini | `outputs/phase_uplift_v2_decile_summary.csv`, `outputs/phase_uplift_v2_vs_baseline.csv` |
| Uplift targeting decision | `outputs/uplift_targeting_decision.csv` |
| Forecast vs RF baseline + naive benchmarks | `outputs/phase_forecast_v2_vs_baseline.csv`, `outputs/forecast_naive_baseline.csv` |

## Combined view

`analysis/customer_360.py` joins churn, uplift, segment and recommendations
into one recommended action per customer — see
[docs/customer_360.md](docs/customer_360.md) for coverage caveats and how to run it.

## Combined view

`analysis/customer_360.py` joins churn, uplift, segment and recommendations
into one recommended action per customer — see
[docs/customer_360.md](docs/customer_360.md) for coverage caveats and how to run it.

## Repository layout

- `sql/` — raw-to-mart SQL, one script per mart plus shared staging.
- `src/data/` — mart loaders, schema contracts, validators, and per-phase runners.
- `src/features/`, `src/models/` — feature builders and trainers per workstream.
- `analysis/` — the honest re-evaluation scripts (matched comparisons, calibration, decisions).
- `docs/` — design docs and the reconciliation write-ups.
- `tests/` — schema/validator unit tests (always run) and mart tests (skip without data).
- `outputs/` — generated metrics, tables and charts.

See [CHANGELOG.md](CHANGELOG.md) for the phase-by-phase build history.

## Limitations

- Data is synthetic; supervised metrics (notably returns-anomaly AP) are optimistic
  because labels and features share a generative process.
- The generated data reproduces the pipeline and realistic metrics, but is not the
  exact dataset the committed `outputs/` were first produced from, so regenerated
  numbers will differ slightly from the committed CSVs.
- The baseline phase6-11 notebooks are superseded by the V2 layer and the analysis
  re-evaluations; treat the `analysis/` results and the docs above as authoritative.
