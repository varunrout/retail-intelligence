# Churn V2 — Model Design and Implementation

## Overview

LightGBM binary classifier trained on the active customer population (purchased at
least once) using a time-ordered train/test split. Addresses the two core Phase 6
flaws: non-purchaser contamination and random-split evaluation.

---

## Source Files

| File | Role |
|---|---|
| `src/features/features_churn.py` | Feature engineering, leakage audit, maturity filter |
| `src/models/train_churn.py` | LightGBM training, threshold selection, evaluation |
| `src/data/run_phase_churn_v2.py` | End-to-end pipeline runner |

---

## Population and Split

- **Population:** Active customers only — customers with at least one purchase
  (`total_orders > 0`). Non-purchasers (11.1% of mart) are excluded; their
  `churn_flag_90d = 0` is structural, not predictive.
- **Maturity filter:** Customers must have ≥ 90 days of tenure before the split date
  to give the churn label time to resolve. Customers signed up too recently cannot
  reliably be labelled churned or not.
- **Split strategy:** Time-ordered by `signup_date` at 2025-03-21 (80/20).
  - Train: 28,515 rows | Validation: 5,033 rows | Test: 8,388 rows
  - Churn rate (train): 21.9% | Churn rate (test): 15.0%

---

## Feature Set

35 approved features after leakage blacklist. Key additions over Phase 6:

| Feature | Type | Notes |
|---|---|---|
| `total_orders` | numeric | Highest gain feature (65,771) |
| `avg_item_discount_pct` | numeric | 2nd by split count (603) |
| `order_velocity_per_month` | engineered | total_orders / tenure_months |
| `purchase_session_rate` | engineered | sessions_with_purchase / total_sessions |
| `cart_to_purchase_rate` | engineered | captures browse-to-buy conversion |
| `is_loyalty_null` | binary flag | explicit null indicator for unenrolled customers |
| `is_campaign_null` | binary flag | explicit null indicator for never-targeted customers |

Excluded: `recency_days` and all derivatives (AUROC = 1.0 → label leakage).

---

## Model Architecture

**Algorithm:** LightGBM (gradient-boosted trees, binary classification)

| Hyperparameter | Value |
|---|---|
| objective | binary |
| learning_rate | 0.05 |
| num_leaves | 63 |
| min_data_in_leaf | 100 |
| feature_fraction | 0.9 |
| bagging_fraction | 0.9 |
| bagging_freq | 5 |
| best_iteration | 133 |

Early stopping on validation set (binary log-loss). Balanced class weights via
`is_unbalance=True`.

---

## Results

| Metric | LightGBM V2 | Phase 6 RF Baseline | Note |
|---|---|---|---|
| ROC AUC | 0.8153 | 0.8656 | Different regimes — not directly comparable |
| PR AUC | 0.4592 | 0.5965 | Different regimes |
| Balanced Accuracy | **0.7389** | 0.7171 | +2.2 pp on comparable metric |
| MCC | 0.3596 | 0.4616 | Different regimes |

**Important caveat:** V2 and Phase 6 metrics are not directly comparable. V2 uses a
time-ordered split with the maturity filter and no recency-derived features. Phase 6
used a random split and included features with implicit recency signal. The V2 regime
is more conservative (harder split) and more honest.

---

## Threshold Policy

| Policy | Threshold | Precision | Recall | F1 | Specificity |
|---|---|---|---|---|---|
| Max-F1 | 0.65 | 0.418 | 0.585 | 0.487 | 0.856 |
| Precision-floor ≥ 0.70 | 0.85 | 0.705 | 0.131 | 0.221 | 0.990 |

Recommended defaults:
- **Retention campaigns (balanced):** threshold = 0.65
- **High-confidence targeted outreach:** threshold = 0.85
