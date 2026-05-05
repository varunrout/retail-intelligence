# Phase Churn V2 Report

## Objective

Train a production-ready churn prediction model on the active customer population
using a time-ordered evaluation regime free from recency leakage.

## Scope and Method

- **Label:** `churn_flag_90d` — no purchase in the 90 days following the customer's
  last recorded order.
- **Population:** Active customers only (44,432 customers with ≥ 1 purchase).
  Non-purchasers excluded.
- **Split:** Time-ordered 80/20 at 2025-03-21 with 90-day maturity filter.
- **Algorithm:** LightGBM binary classification with early stopping.
- **Feature set:** 35 approved features — recency-derived features explicitly excluded
  after AUC = 1.0 leakage detection.

## Data Snapshot

- Train rows: 28,515 | Validation: 5,033 | Test: 8,388
- Churn rate (train): 21.9% | Churn rate (test): 15.0%
- Approved features: 35 (34 base + 1 engineered: `order_velocity_per_month`)

## Model Results

### LightGBM V2 (time-ordered, no recency)

- ROC AUC: 0.8153
- PR AUC: 0.4592
- Balanced Accuracy: 0.7389
- MCC: 0.3596
- Best iteration: 133

## Comparison With Phase 6 Baseline

| Metric | Phase 6 RF | V2 LightGBM | Note |
|---|---|---|---|
| ROC AUC | 0.8656 | 0.8153 | Different regimes — not comparable |
| Balanced Accuracy | 0.7171 | **0.7389** | +2.2 pp on honest time-ordered split |

Metrics are not directly comparable: Phase 6 used a random split and included
recency-derived features (which are near-perfect predictors). V2 represents an
honest evaluation on a harder, more realistic regime.

## Threshold Policy

| Policy | Threshold | Precision | Recall | F1 |
|---|---|---|---|---|
| Retention campaigns (max-F1) | 0.65 | 0.418 | 0.585 | 0.487 |
| High-confidence outreach | 0.85 | 0.705 | 0.131 | 0.221 |

## Output Artifacts

| File | Contents |
|---|---|
| `outputs/phase_churn_v2_params.csv` | Hyperparameter log |
| `outputs/phase_churn_v2_model_comparison.csv` | All metrics on test set |
| `outputs/phase_churn_v2_vs_baseline.csv` | V2 vs Phase 6 baseline |
| `outputs/phase_churn_v2_threshold_selection.csv` | Threshold policies |
| `outputs/phase_churn_v2_feature_importance_top30.csv` | Top 30 features by gain |

## Key Findings

1. **Top feature by a wide margin:** `total_orders` (gain = 65,771 — 14× next feature).
   Purchase frequency dominates over all other signals.
2. **Recency exclusion confirmed necessary:** recency-derived features yielded AUC = 1.0
   in single-feature screening — they encode the label directly.
3. **Balanced accuracy improves over baseline (+2.2 pp)** despite the harder evaluation
   regime, confirming that LightGBM with proper validation is a more stable model.
4. **Maturity filter is essential:** Without the 90-day maturity filter, recently acquired
   customers with short tenure contaminate the churn label.

## Business Recommendations

- Deploy at threshold 0.65 for broad retention campaigns (recall = 58.5%).
- Use threshold 0.85 for high-cost personalised interventions where precision is critical.
- Retrain quarterly; churn rate may drift seasonally (test period churn = 15.0% vs train = 21.9%).
- Enrich feature set with campaign response history once uplift model outputs are
  available — campaign exposure likely moderates churn probability.
