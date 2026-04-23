# Phase 8 Forecasting Baseline Report

## Objective

Build a leakage-safe demand forecasting baseline from mart_product_demand and compare naive time-series baselines against a feature-based model for weekly product-store demand.

## Scope And Method

- Target: target_units (derived from units_sold).
- Split: time-ordered 80/20 by week_start_date to mimic forward forecasting.
- Leakage controls:
  - Dropped same-week realized outcomes and target mirrors:
    - units_sold
    - target_units
    - order_line_count
    - net_revenue
    - weekly_demand_band
    - rolling_4w_avg_units
    - rolling_4w_revenue
  - Exported leakage audit and approved feature list.
- Baselines:
  - naive_lag_1w
  - naive_lag_4w
  - naive_blend_70_30
- Feature model:
  - Random Forest Regressor with preprocessing pipeline.
  - Numeric: median imputation.
  - Categorical: most_frequent imputation + one-hot encoding.

## Data Snapshot

- Train rows: 488,985
- Test rows: 150,688
- Split week anchor: 2025-08-04
- Approved feature count: 23
- RF training mode: quality (full train set)
- RF training runtime: 9.9 minutes (592 seconds)

## Model Results

### Random Forest Baseline (best by RMSE)

- MAE: 1.2512
- RMSE: 1.8496
- sMAPE: 0.4677

### Naive Blend 70/30

- MAE: 1.4906
- RMSE: 2.1717
- sMAPE: 0.5142

### Naive Lag 4-Week

- MAE: 1.5983
- RMSE: 2.4045
- sMAPE: 0.5392

### Naive Lag 1-Week

- MAE: 1.6369
- RMSE: 2.4473
- sMAPE: 0.5443

## Interpretation

- The feature-based Random Forest outperforms all naive baselines across MAE, RMSE, and sMAPE.
- Relative to the strongest naive baseline (naive_blend_70_30), the Random Forest achieves:
  - RMSE improvement: about 14.8%
  - MAE improvement: about 16.1%
  - sMAPE improvement: about 9.1%
- Weekly diagnostics show systematic underprediction during peak-demand periods (Nov-Dec), which is common in baseline tree models without explicit holiday/event effects.
- Baseline is suitable for ranking demand intensity and planning inventory at weekly horizon, with further gains expected from richer temporal/event features.

## Diagnostics Produced

- Three-panel forecasting diagnostic chart with:
  - Actual vs RF predicted weekly demand and residual uncertainty band.
  - Weekly residual bars with rolling 4-week MAE.
  - Actual vs RF vs naive blend comparison.

## Caveats

- Baseline does not explicitly encode holidays/promo calendars as exogenous regressors in the final model.
- Evaluation is aggregate-friendly but still based on point forecasts; no quantile/interval forecasting model was trained.
- Some late-year peak weeks are underpredicted, so high-spike replenishment should include safety buffers.
- Feature importance and error stratification by product class/store type should be added before operational deployment.

## Artifacts

- outputs/phase8_forecast_leakage_audit.csv
- outputs/phase8_forecast_approved_features.csv
- outputs/phase8_forecast_model_comparison.csv
- outputs/phase8_forecast_scored_sample_top500.csv
- outputs/phase8_forecast_weekly_actual_vs_pred.csv
- outputs/phase8_forecast_vs_actual.png
