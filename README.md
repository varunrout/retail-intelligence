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

Data (raw CSVs ~1GB and the built marts) is gitignored. With the marts present
under `data/processed/`:

```bash
pip install -r requirements.txt
export PYTHONPATH=.

# Re-run a model end to end (marts -> features -> model -> outputs/)
python -m src.data.run_phase_churn_v2
python -m src.data.run_phase_uplift_v2

# The honest re-evaluations (each has a --smoke mode needing no data)
python -m analysis.churn_incremental_lift        # matched churn baseline vs V2 + CI
python -m analysis.churn_calibration             # reliability + calibrated Brier
python -m analysis.uplift_targeting_decision     # who gets an offer

pytest            # data tests skip if marts absent; unit tests always run
ruff check . && ruff format --check .
```

`--smoke` runs the analysis scripts on synthetic in-memory data, so they work on a
clean clone with no marts.

> Reproducibility gap (known): a committed raw-data generator is not yet in the
> repo, so a fresh clone cannot rebuild `data/processed` from scratch. Tracked in
> FIXES.md (RETA-02). The analysis scripts' `--smoke` mode and the unit tests are
> the parts that run with no data at all.

## Results to files

| Result | File |
|---|---|
| Churn matched baseline comparison | `outputs/churn_incremental_lift_metrics.csv` |
| Churn paired-bootstrap lift + verdict | `outputs/churn_incremental_lift_paired_bootstrap.csv` |
| Churn calibration (Brier, reliability) | `outputs/churn_calibration_brier.csv`, `outputs/churn_calibration_reliability.png` |
| Uplift decile table + Qini | `outputs/phase_uplift_v2_decile_summary.csv`, `outputs/phase_uplift_v2_vs_baseline.csv` |
| Uplift targeting decision | `outputs/uplift_targeting_decision.csv` |

## Repository layout

- `sql/` — raw-to-mart SQL, one script per mart plus shared staging.
- `src/data/` — mart loaders, schema contracts, validators, and per-phase runners.
- `src/features/`, `src/models/` — feature builders and trainers per workstream.
- `analysis/` — the honest re-evaluation scripts (matched comparisons, calibration, decisions).
- `docs/` — design docs and the reconciliation write-ups.
- `tests/` — schema/validator unit tests (always run) and mart tests (skip without data).
- `outputs/` — generated metrics, tables and charts.

## Limitations

- Data is synthetic; supervised metrics (notably returns-anomaly AP) are optimistic
  because labels and features share a generative process.
- No committed raw-data generator yet (see above).
- The baseline phase6-11 notebooks are superseded by the V2 layer and the analysis
  re-evaluations; treat the `analysis/` results and the docs above as authoritative.
