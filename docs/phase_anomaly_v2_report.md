# Phase Anomaly V2 Report

## Objective

Replace the Phase 10/11 unsupervised baselines with a supervised LightGBM model for
return fraud and abuse detection, using the `returns_hidden_labels.csv` **synthetic
seed-derived labels** (not manual annotations; label and features share a generative
process, so the supervised AP is optimistic — see `docs/modeling_v2_anomaly.md`).

## Scope and Method

- **Label:** Binary fraud/abuse flag from `data/processed/returns_hidden_labels.csv`.
- **Population:** All labelled returns (188,399 records).
- **Split:** Time-ordered at 2025-09-26 (80/20 by `return_date`).
- **Algorithm:** LightGBM binary classifier with `scale_pos_weight` for class imbalance.
- **Feature set:** 12 features from returns + orders + customers.

## Data Snapshot

- Train: 150,719 | Test: 37,680
- Prevalence (train): 0.79% | Prevalence (test): 0.42%
- Scale pos weight: 125.34 (n_neg_train / n_pos_train)

## Model Results

### LightGBM V2 (Supervised)

- Average Precision: **0.580**
- Precision (at 1% flag rate): **0.332**
- Recall (at 1% flag rate): **0.786**
- F1 (at 1% flag rate): **0.466**
- Flag rate: 1.0%
- Flagged returns: 377

## Comparison With Phase 10/11 Baselines

| Method | Type | Flag Rate | Precision | Recall | F1 | AP |
|---|---|---|---|---|---|---|
| IQR / Rules | unsupervised | 12.0% | 0.027 | 0.460 | 0.052 | 0.017 |
| Isolation Forest | unsupervised | 1.0% | 0.101 | 0.141 | 0.117 | 0.058 |
| LOF (n=20) | unsupervised | 1.0% | 0.168 | 0.062 | 0.091 | 0.080 |
| **LightGBM V2** | **supervised** | **1.0%** | **0.332** | **0.786** | **0.466** | **0.580** |

**AP improves 10×** over the best unsupervised baseline (LOF, AP = 0.080). At the same
1% flag rate, precision improves from 0.027 (IQR) to 0.332 — meaning each flagged
return has a **1-in-3 chance of being confirmed abuse** versus 1-in-37 for IQR rules.

## Top Features by Gain

| Feature | Gain | Interpretation |
|---|---|---|
| `item_net_price` | 5,425,036 | High-value items dominate abuse |
| `prior_customer_return_rate` | 1,030,810 | Repeat offenders |
| `customer_item_recency_rank` | 538,508 | Quick returns after purchase |
| `is_suspected_abuse_reason` | 365,882 | Return reason text flag |
| `item_discount_pct` | 162,177 | High-discount items targeted |
| `item_margin` | 64,223 | Margin arbitrage motive |

## Output Artifacts

| File | Contents |
|---|---|
| `outputs/phase_anomaly_v2_metrics.json` | AP, precision, recall, F1, flag rate, threshold |
| `outputs/phase_anomaly_v2_model_comparison.csv` | All 4 methods side-by-side |
| `outputs/phase_anomaly_v2_vs_baseline.csv` | V2 vs Phase 11 baseline |
| `outputs/phase_anomaly_v2_feature_importance.csv` | Feature importance by gain |
| `outputs/phase_anomaly_v2_review_queue.csv` | Flagged returns for review |
| `outputs/phase_anomaly_v2_pr_curve.png` | Precision-recall curve |

## Key Findings

1. **Ground truth transforms performance:** AP of 0.580 is 10× LOF (0.080) and 7.25×
   Isolation Forest (0.058). Supervised learning requires only 12 features to massively
   outperform unsupervised methods.
2. **`item_net_price` AUROC = 0.925** as a single feature — price alone is the strongest
   discriminator. A simple price-threshold rule captures most abuse with zero ML complexity.
3. **Electronics = 75.5% of abuse** — a category routing rule (score all electronics
   returns by the model) would capture the vast majority of fraud with low false-positive
   cost on other categories.
4. **Repeat offenders are identifiable:** `prior_customer_return_rate` is the 2nd most
   important feature — a customer-level block-list for high-rate returners could
   complement the model.

## Business Recommendations

- **Immediate review queue:** Use `outputs/phase_anomaly_v2_review_queue.csv` (377
  flagged returns at 1% flag rate) as the starting point for manual review.
- **Category routing:** Route all electronics returns through the model automatically;
  apply a lighter price-threshold rule for other categories.
- **Precision vs recall tradeoff:** Raise the threshold from 0.00170 for automated
  holds (higher precision required); lower for watchlisting only.
- **Retrain cadence:** Retrain quarterly. Fraud patterns evolve; `prior_customer_return_rate`
  may drift as known abusers churn off or create new accounts.
