# Phase 12 registry — reconciliation with the shipping models

`outputs/phase12_pipeline_registry.csv` and `outputs/phase12_alert_thresholds.csv`
were written against the **phase6-11 V1 models**, which used stratified-random
splits. The baselines and alert thresholds in those files are therefore taken from
random-split holdouts and do not match the models that should actually ship, which
are evaluated on **time-ordered** splits and score lower. Left unchanged, several
alert thresholds would fire on day one.

## Where the registry is wrong

| Pipeline | Registry says | Reality on the time-ordered holdout | Consequence |
|---|---|---|---|
| P01 Churn | `churn_rf_v1.pkl`, AUC 0.896, alert if AUC < 0.85 | Best model is logistic regression at ROC-AUC **0.844** (LightGBM 0.812); see the churn reconciliation | The 0.85 alert **fires immediately** |
| P01 Churn | Precision@0.3 = 0.432 | Raw scores are miscalibrated (mean pred 0.44 vs 0.15 actual); thresholds need calibrated probabilities | Precision alerts are measured on the wrong scale |
| P02 Uplift | Top-3-decile uplift = 0.180 | Observed top-3-decile uplift is **0.0625** | The 0.10 alert **fires immediately** |
| P03 Forecast | `forecast_rf_v1.pkl`, RMSE 1.85 | V2 RMSE 1.827, but the honest framing is "RF is the ceiling, V2 marginal" | Artifact and framing both stale |
| P06 Anomaly | AUC 0.885, Precision 0.101 | Supervised AP ~0.58 on synthetic seed labels (optimistic); unsupervised baselines much lower | Metric provenance overstated |

## Corrected reference numbers (time-ordered holdouts)

Use these when the registry is regenerated. Thresholds are set relative to the
actual holdout, not the superseded random-split number.

| Pipeline | Model to pin | Baseline metric | Suggested alert |
|---|---|---|---|
| P01 Churn | logistic regression (ranking) + isotonic calibration (probabilities) | ROC-AUC 0.844, calibrated Brier 0.141 | AUC-ROC < 0.80; recalibrate if Brier > 0.17 |
| P02 Uplift | X-learner V2 | top-3-decile uplift 0.0625, Qini 745 | top-3-decile uplift < 0.045 (the population ATE) |
| P03 Forecast | RF baseline (V2 is marginal) | MAE 1.251, beats naive 1.60 | MAE > 1.45 (naive floor is ~1.60) |
| P06 Anomaly | supervised LightGBM (flag optimism) | AP ~0.58 on synthetic labels | daily flag-rate band as before |

Regenerating `phase12_pipeline_registry.csv` and `phase12_alert_thresholds.csv`
from these time-ordered numbers, and swapping the `*_v1.pkl` artifacts for the
shipping models, is tracked in FUNCTIONAL.md (RETA-F07). This file records the
correct targets in the meantime so nobody trusts the stale thresholds.
