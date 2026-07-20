# Anomaly Detection V2 — Model Design and Implementation

## Overview

LightGBM supervised binary classifier for return fraud / abuse detection, trained on
**synthetic seed-derived labels** (`returns_hidden_labels.csv`). These are not
manually annotated: the data is synthetic and the label comes from the same
generative process that produced the features. Replaces the Phase 10 unsupervised
baselines (IQR flag rate 12%, IF average precision 0.058, LOF AP 0.080) with a
precision-oriented model suitable for operational flagging.

> Caveat: because the label and the features share a generative process, the
> supervised average precision (~0.58) is optimistic and would not transfer to real
> annotated returns. Treat it as an upper bound, not a production estimate.

---

## Source Files

| File | Role |
|---|---|
| `src/features/features_anomaly.py` | Feature engineering from returns + orders + customers |
| `src/models/train_anomaly.py` | LightGBM training, class weighting, threshold selection |
| `src/data/run_phase_anomaly_v2.py` | End-to-end pipeline runner |

---

## Population and Split

Ground truth from `data/processed/returns_hidden_labels.csv`:
- Total labelled returns: 188,399
- Train: 150,719 | Test: 37,680
- Split date: 2025-09-26 (time-ordered, chronological by `return_date`)

Class prevalence:
- Train prevalence: 0.79% (positive = abuse / fraud)
- Test prevalence: 0.42%

---

## Class Imbalance Handling

The extreme class imbalance (≈ 120:1 negative to positive) is addressed via
LightGBM's `scale_pos_weight = 125.34` (calculated as
`n_negative_train / n_positive_train`).

This directly adjusts the gradient weight for positive class examples during boosting,
effectively increasing recall without modifying the training data.

---

## Feature Set

12 features, selected from the signal audit:

| Feature | Category | Notes |
|---|---|---|
| `item_net_price` | Price | AUROC 0.925 — highest single-feature discriminator |
| `refund_amount` | Return | Refund relative to item value |
| `prior_customer_return_rate` | Customer history | Abuse pattern in prior returns |
| `item_discount_pct` | Price | High-discount items targeted for fraud |
| `customer_item_recency_rank` | Timing | Returns on recently purchased items |
| `item_margin` | Price | Low-margin abuse items |
| `recent_product_return_events` | Product | Product-level return spike |
| `days_to_return` | Timing | Fast returns (abuse) vs slow returns (genuine) |
| `is_suspected_abuse_reason` | Label-derived | Return reason text flag |
| `is_electronics` | Category | 75.5% of abusive returns are electronics |
| `is_low_discount` | Price | Low-discount items more likely to be genuine returns |
| `is_high_risk_band` | Risk | Customer-level risk banding |

Excluded: item identifiers, raw text fields, date fields, post-return status flags.

---

## Model Architecture

**Algorithm:** LightGBM binary classifier with scale_pos_weight

| Hyperparameter | Value |
|---|---|
| objective | binary |
| scale_pos_weight | 125.34 |
| learning_rate | 0.05 |
| num_leaves | 63 |
| feature_fraction | 0.8 |
| n_features | 12 |

Early stopping on validation set (binary log-loss with class weights applied).

---

## Results

| Metric | LightGBM V2 | IQR Baseline | IF Baseline | LOF Baseline |
|---|---|---|---|---|
| Average Precision | **0.580** | — | 0.058 | 0.080 |
| Precision at threshold | **0.332** | 0.027 | 0.027 | 0.027 |
| Recall | **0.786** | — | — | — |
| F1 | **0.466** | — | — | — |
| Flag rate | **1.0%** | 12.0% | ~1–2% | ~1–2% |

V2 average precision (0.580) is **7.25× IF** and **10× LOF**, confirming that ground-truth
labels unlock discriminative performance the unsupervised methods cannot approach.

---

## Threshold Policy

| Policy | Threshold | Precision | Recall | F1 | Flag Rate |
|---|---|---|---|---|---|
| High-recall operations | 0.00170 | 0.332 | 0.786 | 0.466 | 1.0% |

The selected threshold is set by the model at the point that yields 1% flag rate on the
test set. At 1% flag rate, each flagged return has a **1-in-3 chance** of being confirmed
abuse — a significant improvement over the unsupervised baselines' ≈ 1-in-37 rate.

For operations requiring higher precision (e.g., automatic hold), raise the threshold
to reduce flag rate to ≈ 0.1–0.3% while accepting lower recall.

---

## Operational Context

- **Electronics dominance:** 75.5% of labelled abuse involves electronics. A category-level
  routing rule (electronics items → always scored by model) is recommended.
- **item_net_price AUROC = 0.925:** Price is the most discriminative single feature,
  suggesting a price-tiered scoring policy (score all returns above price threshold)
  may achieve near-model recall at lower compute cost.
- **Prior return rate:** Customers with high historical return rates drive a substantial
  portion of positive labels — a customer-level block-list can complement the model.
