# Forecast V2 — Model Design and Implementation

## Overview

LightGBM Tweedie regression for weekly product-store demand forecasting. Addresses
the Phase 8 Random Forest baseline's systematic under-prediction bias and limited
feature set with enriched lag features, series-level baselines, and cyclical calendar
encoding.

---

## Source Files

| File | Role |
|---|---|
| `src/features/features_forecast.py` | Lag engineering, series baselines, short-series filter |
| `src/models/train_forecast.py` | LightGBM Tweedie training, per-category evaluation |
| `src/data/run_phase_forecast_v2.py` | End-to-end pipeline runner |

---

## Population and Split

- **Short-series filter:** Series with < 8 weeks of history are excluded before training.
  This removes 12.7% of series but substantially reduces noise from under-specified lag
  features.
- **Split strategy:** Time-ordered 80/20 by `week_start_date`.
  - Test rows: 150,688

---

## Feature Set

Key features by importance gain:

| Rank | Feature | Gain | Notes |
|---|---|---|---|
| 1 | `product_mean_demand` | 203,693 | Series baseline demand |
| 2 | `store_mean_demand` | 127,573 | Store-level baseline |
| 3 | `category` | 50,334 | Product category OHE |
| 4 | `roll_13w_avg` | 36,917 | 13-week lagged rolling average |
| 5 | `sin_woy` | 32,902 | Cyclical week-of-year |
| 6 | `avg_starting_inventory` | 26,599 | Inventory signal |
| 7 | `lag_13w` | 26,394 | Quarterly lag |
| 8 | `cos_woy` | 22,742 | Cyclical week-of-year (pair) |
| 9 | `subcategory` | 12,911 | Subcategory OHE |

**New vs Phase 8:**
- `product_mean_demand` and `store_mean_demand` — series-level intercepts (top 2 features)
- `roll_8w_avg`, `roll_13w_avg` — lagged rolling means (avoid leakage)
- `sin_woy` / `cos_woy` — cyclical encoding (no week-52 to week-1 discontinuity)
- `lag_52w` — YoY anchor (available for ~50% of test rows, with null flag)
- `avg_starting_inventory` — re-enabled with null handling

---

## Model Architecture

**Algorithm:** LightGBM with Tweedie objective

The Tweedie distribution (variance power = 1.5) is appropriate for count-like right-skewed
demand data with non-negative target values. It reduces the bias problem of MSE objectives
on zero-heavy or skewed targets.

| Hyperparameter | Value |
|---|---|
| objective | tweedie |
| tweedie_variance_power | 1.5 |
| learning_rate | 0.05 |
| num_leaves | 127 |
| min_data_in_leaf | 50 |
| feature_fraction | 0.8 |
| lambda_l1 | 0.1 |
| lambda_l2 | 0.1 |
| best_iteration | 182 |

Higher `num_leaves = 127` (vs 63 used in churn/uplift) to capture the multi-dimensional
product × store × time interaction structure.

---

## Results

| Metric | LightGBM Tweedie V2 | Phase 8 RF Baseline | Delta | % Improvement |
|---|---|---|---|---|
| MAE | **1.234** | 1.251 | −0.018 | +1.4% |
| RMSE | **1.827** | 1.850 | −0.023 | +1.2% |
| sMAPE | **0.456** | 0.468 | −0.011 | +2.4% |
| Mean Bias | **−0.471** | ~−2.8/wk aggregate | Substantially reduced |

**Mean bias reduction** is the most operationally significant improvement. Phase 8 RF
systematically under-predicted by ≈ 2,800 units per week in aggregate. V2 Tweedie
reduces this to a mean bias of −0.47 units per product-store-week.

---

## Per-Category Performance

V2 is evaluated per category to identify where forecast quality is highest and lowest:

| Category | sMAPE (V2) | Phase 8 RF | Notes |
|---|---|---|---|
| electronics | ~0.45 | ~0.47 | Stable, predictable demand |
| fashion | ~0.47 | ~0.50 | Seasonal; sin/cos encoding helps |
| beauty | ~0.62 | ~0.67 | High volatility; hardest category |
| grocery_light | ~0.65 | ~0.67 | Impulse demand; weakest autocorrelation |

---

## Limitations

- Absolute improvement over Phase 8 baseline is modest (1–2%) because the Phase 8 RF
  was already a reasonable model on the same features. The Tweedie objective and richer
  features primarily address the systematic under-prediction bias.
- Beauty and grocery_light categories remain hard to forecast accurately due to
  inherently low autocorrelation. Per-category models or separate hyperparameter tuning
  could be explored in a future iteration.
- `lag_52w` (YoY seasonality) is only available for series ≥ 52 weeks old (≈ 50% of
  test rows), limiting its contribution to newer products.
