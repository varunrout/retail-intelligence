# Segmentation V2 — Model Design and Implementation

## Overview

KMeans segmentation on the buyer population with an explicit Non-Purchaser segment
added post-hoc. Addresses Phase 9 contamination of non-purchasers and feature
collinearity by introducing log1p transforms and an adjusted PCA dimension.

---

## Source Files

| File | Role |
|---|---|
| `src/features/features_segmentation.py` | Feature selection, log1p transforms, collinearity audit |
| `src/models/train_segmentation.py` | KMeans training, PCA, cluster profiling |
| `src/data/run_phase_segmentation_v2.py` | End-to-end pipeline runner |

---

## Population Design

### Problem with Phase 9

Phase 9 included all 50,000 customers in the clustering, including 5,568 non-purchasers
(`total_orders = 0`). Non-purchasers share many feature values (all revenue/order features
= 0) and form a structural cluster that dominates cluster assignments. The spread of
churn rate across Phase 9 segments was **0 pp** because non-purchasers do not have a
meaningful churn label.

### V2 Approach

1. **Buyer population only:** Cluster k=3 over the 44,432 customers with `total_orders > 0`.
2. **Non-purchaser segment:** After clustering, re-add the 5,568 non-purchasers as a
   dedicated 4th segment (cluster_id = 3, name = "Non-Purchasers").
3. **Total segments:** 4 — three meaningful buyer segments + one structural non-buyer segment.

---

## Feature Engineering

### Log1p Transforms

18 collinear feature pairs were identified in Phase 9 (Spearman r > 0.90). V2 applies
`log1p(x)` to all right-skewed numeric features before PCA:
- Revenue features: `total_revenue`, `avg_order_value`, `avg_item_price`
- Frequency: `total_orders`, `total_units`, `total_sessions`
- Returns: `total_refund_amount`

Log1p transforms reduce collinearity by spreading the heavy tail (high-value / high-frequency
customers) and improve silhouette from 0.244 → 0.346 before PCA.

### Null Handling

- `loyalty_tier` null encoded as `unknown` level
- `preferred_channel` null encoded as `unknown` level
- `is_loyalty_null` binary flag added for customers never enrolled

---

## Dimensionality Reduction

PCA applied on the buyer population to reduce the feature space before KMeans.

| Setting | Phase 9 | V2 |
|---|---|---|
| n_components | 6 | 5 |
| Population | All 50,000 | 44,432 buyers |
| Variance explained | 78% | ~75% |

PCA-5 was selected by silhouette on the buyer subset. Using 5 components on a population
without the structural zero-spike of non-purchasers yields cleaner separability.

---

## Cluster Optimisation

Silhouette score on buyer population:

| k | Silhouette (V2) |
|---|---|
| 2 | 0.551 |
| 3 | 0.293 ← selected |
| 4 | 0.241 |
| 5 | 0.214 |

k=2 has highest silhouette but insufficient business granularity (only high vs low value).
k=3 was selected as the best balance of cluster quality and actionable segment count.

---

## Results

### Segment Profiles

| Segment | n | % of Total | Churn Rate | Median AOV | Avg Revenue | Median Recency (days) |
|---|---|---|---|---|---|---|
| **Champions** | 24,352 | 48.7% | 8.0% | £130.92 | £2,320.5 | 19 |
| **Mid-Tier Active** | 5,187 | 10.4% | 1.2% | £111.53 | £505.8 | 14 |
| **Dormant Buyers** | 14,893 | 29.8% | 43.9% | £88.49 | £440.5 | 71 |
| **Non-Purchasers** | 5,568 | 11.1% | — | — | — | 474 |

### Quality Metrics

| Metric | Phase 9 | V2 | Delta |
|---|---|---|---|
| Silhouette (buyers) | 0.252 | 0.293 | +0.040 |
| Davies-Bouldin (buyers) | — | 1.592 | — |
| Churn spread across segments | 0 pp | **43.9 pp** | +43.9 pp |

**Churn spread** — the difference in churn rate between the highest and lowest churn
segment — improves from 0 pp (Phase 9, where non-purchasers' 0% masked all variation)
to 43.9 pp (Champions 8.0% vs Dormant Buyers 43.9%).

---

## Actionability

| Segment | Priority Action |
|---|---|
| Champions | Loyalty programme deepening, cross-sell high-margin categories |
| Mid-Tier Active | Upsell to higher AOV; most frequent purchasers (order_rate = 1.02/month) |
| Dormant Buyers | Targeted win-back campaign; 43.9% churn rate indicates urgent re-engagement |
| Non-Purchasers | On-boarding flow review; 474-day median recency indicates cold acquisition |
