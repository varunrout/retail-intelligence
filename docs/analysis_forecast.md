# Forecast Analysis — Evidence Audit Before V2

## Purpose

Evidence-first analysis of `mart_product_demand` to identify limitations of the Phase 8
forecasting baseline and design a V2 grounded in data properties.

---

## §1 Data Profile

| Metric | Value |
|---|---|
| Active series | 12,729 (product × store combinations) |
| Median series length | 47 weeks |
| Short series (<8 weeks) | 12.7% of series |
| Target distribution | Right-skewed count (mean=2.24, max=26) |
| Zero-demand weeks | Absent — filtered out in mart construction |

**Finding 1.A:** 12.7% of series are very short (<8 weeks). These lack sufficient
history for lag features to be meaningful and inflate evaluation error metrics. They
require either exclusion or a separate short-series treatment.

**Finding 1.B:** The target (`target_units`) is right-skewed count data. Mean of 2.24
units per product-store-week with a maximum of 26. Absolute-error metrics (MAE, RMSE)
will be dominated by high-volume product-store pairs.

---

## §2 Leakage Audit

**Finding 2.A — Confirmed leakage in mart:** `rolling_4w_avg_units` is computed over
the *current* week — it is a 100% information leak. The Phase 8 baseline correctly
excluded it. `rolling_4w_revenue` is also leakage by the same mechanism.

**Finding 2.B:** The largest **safe** lagged signals are:
- `lag_1w` (r = 0.214)
- `lag_4w` (r = 0.207)

Both are significant but modest — demand is genuinely noisy. Strong autocorrelation is
not present; the forecasting problem is hard by nature.

---

## §3 Baseline Limitations

**Finding 3.A:** Random Forest beats the naive seasonal-lag blend by 16.1% MAE. A feature-
based model adds value but sMAPE = 0.47 leaves substantial room for improvement.

**Finding 3.B:** Weekly aggregate forecast shows RF systematically **under-predicts** —
residuals are consistently negative (mean undercount ≈ 2,800 units/week). This is a
systematic bias, not random noise. LightGBM with proper leaf regularisation should
reduce this bias.

---

## §4 Autocorrelation and Seasonality

**Finding 4.A:** Lag-1 is the strongest autocorrelation (r ≈ 0.21). Lags 4, 8, 13, 52
are all positive but weaker. The YoY lag-52 is available and statistically significant —
it provides a seasonality anchor.

**Finding 4.B:** Week-of-year shows a seasonal pattern with a 1.30 unit peak-to-trough
swing. Cyclical encoding (sin/cos of week_of_year) is preferred over ordinal integers to
avoid creating a discontinuity between week 52 and week 1.

**Finding 4.C:** Inventory features have r = 0.168 — meaningful signal. However 25% of
inventory columns are null (partial store coverage). LightGBM handles missing values
natively; a null-indicator flag is not required.

---

## §5 Category and Series-Length Heterogeneity

**Finding 5.A:** `beauty` and `grocery_light` have sMAPE ≈ 0.67 vs `fashion`, `home`,
`electronics` at ≈ 0.47–0.50. High-volatility categories may need separate hyperparameter
tuning or category-specific models in a future iteration.

**Finding 5.B:** Short-series (Q1 by length) have materially higher sMAPE. Series with
<8 weeks of history should be filtered before training — this removes 12.7% of series
but substantially reduces training noise.

---

## §6 New Features for V2

**Finding 6.A:** `product_mean_demand` and `store_mean_demand` are strong predictors
(r ≈ 0.3–0.4). Encoding the series-level baseline demand as a feature gives the model
an explicit intercept for each product-store combination.

**Finding 6.B:** `roll_8w_avg` and `roll_13w_avg` are moderately predictive (r ≈ 0.20–0.25).
Lagged rolling means avoid current-week leakage and add smoothed signal beyond lag_1w alone.

**Finding 6.C:** `lag_52w` (YoY) is available for ≈ 50% of test rows (series ≥ 52 weeks).
Include with a null indicator flag for shorter series.

---

## §7 V2 Design Decisions

| Dimension | Phase 8 Baseline | V2 Decision |
|---|---|---|
| Model | Random Forest | LightGBM with Optuna tuning |
| Series filter | None | Exclude series with <8 weeks history |
| Features | lag_1w, lag_4w, week seasonality | + lag_52w, product_mean, store_mean, roll_8/13w_avg, sin/cos week |
| Seasonality encoding | Ordinal week-of-year | Cyclical sin/cos encoding |
| Category handling | Pooled | Category as feature; per-category evaluation |
| Bias correction | None | Monitor train/test residual mean; add mean-shift correction if needed |
