# Phase 6 Churn Baseline Report

## Objective

Build a leakage-safe churn baseline from `mart_customer_features` and establish threshold-ready retention targeting metrics.

## Scope And Method

- Label: `churn_flag_90d`.
- Split: stratified random 80/20.
- Leakage controls:
  - Rule blacklist for known leakage and proxy-label mirrors.
  - Automated single-feature AUC screening with audit export.
- Models:
  - Logistic Regression (balanced class weights).
  - Random Forest (balanced subsample class weights).

## Data Snapshot

- Train rows: 40,000
- Test rows: 10,000
- Churn prevalence (train): 17.1225%
- Churn prevalence (test): 17.12%
- Approved feature count: 34

## Model Results

### Random Forest Baseline (best by ROC/PR)

- ROC AUC: 0.8656
- PR AUC: 0.5965
- Log loss: 0.3299
- Brier score: 0.1026
- Accuracy at 0.50: 0.8552
- Balanced accuracy at 0.50: 0.7171
- MCC at 0.50: 0.4616

### Logistic Regression Baseline

- ROC AUC: 0.8477
- PR AUC: 0.5553
- Log loss: 0.4826
- Brier score: 0.1610
- Accuracy at 0.50: 0.7637
- Balanced accuracy at 0.50: 0.7685
- MCC at 0.50: 0.4299

## Threshold Policy Outputs

Using the Random Forest model:

- Max-F1 operating point:
  - Threshold: 0.35
  - Precision: 0.4957
  - Recall: 0.6741
  - F1: 0.5713
  - Specificity: 0.8583

- Precision-floor policy (>= 0.70):
  - Threshold: 0.70
  - Precision: 0.7456
  - Recall: 0.2465
  - F1: 0.3705
  - Specificity: 0.9826

## Interpretation

- Leakage-safe metrics are now realistic and suitable as baseline references.
- The model is useful for ranking churn risk, with configurable trade-off between coverage and precision through threshold policy.
- Recommended default for balanced retention campaigns: threshold 0.35.
- Recommended default for constrained high-confidence outreach: threshold 0.70.

## Caveats

- Churn remains a proxy label based on inactivity policy.
- Model importance is predictive, not causal.
- Threshold should be aligned to campaign budget and false-positive tolerance.

## Artifacts

- `outputs/phase6_churn_leakage_audit.csv`
- `outputs/phase6_churn_approved_features.csv`
- `outputs/phase6_churn_model_comparison.csv`
- `outputs/phase6_churn_threshold_diagnostics.csv`
- `outputs/phase6_churn_threshold_selection.csv`
- `outputs/phase6_churn_feature_importance_top30.csv`
- `outputs/phase6_churn_scored_sample_top500.csv`
- `outputs/phase6_churn_roc_curve.png`
- `outputs/phase6_churn_pr_curve.png`
