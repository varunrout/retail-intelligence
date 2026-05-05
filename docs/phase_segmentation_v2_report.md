# Phase Segmentation V2 Report

## Objective

Produce actionable customer segments with meaningful churn differentiation, correcting
the Phase 9 non-purchaser contamination that suppressed all churn variation across
segments.

## Scope and Method

- **Population:** All 50,000 customers from `mart_customer_features`.
- **Approach:** KMeans (k=3) on 44,432 buyers + Non-Purchasers segment added post-hoc.
- **Feature transforms:** log1p applied to all right-skewed numeric features before PCA.
- **Dimensionality reduction:** PCA with 5 components (buyer subset only).
- **Selection criterion:** Silhouette score on buyer population.

## Data Snapshot

- Total customers: 50,000
- Buyer population (clustered): 44,432
- Non-purchaser population (assigned post-hoc): 5,568
- Features before PCA: ~18 (after collinear pair reduction)
- PCA components: 5

## Segment Profiles

| Segment | n | % of Total | Churn Rate | Median AOV | Avg Revenue | Median Recency |
|---|---|---|---|---|---|---|
| **Champions** | 24,352 | 48.7% | 8.0% | £130.92 | £2,320.5 | 19 days |
| **Mid-Tier Active** | 5,187 | 10.4% | 1.2% | £111.53 | £505.8 | 14 days |
| **Dormant Buyers** | 14,893 | 29.8% | **43.9%** | £88.49 | £440.5 | 71 days |
| **Non-Purchasers** | 5,568 | 11.1% | — | — | — | 474 days |

## Comparison With Phase 9 Baseline

| Metric | Phase 9 | V2 | Delta |
|---|---|---|---|
| Total segments | 4 | 4 | — |
| Silhouette (buyers) | 0.252 | **0.293** | +0.040 |
| Davies-Bouldin (buyers) | — | 1.592 | — |
| Churn spread across segments | **0 pp** | **43.9 pp** | **+43.9 pp** |

Phase 9 had zero churn spread across segments because non-purchasers (churn = 0 by
definition) contaminated the clustering, making segments structurally indistinguishable
on churn rate.

## Output Artifacts

| File | Contents |
|---|---|
| `outputs/phase_segmentation_v2_metrics.json` | Silhouette, DB, churn spread, cluster sizes |
| `outputs/phase_segmentation_v2_cluster_profiles.csv` | Per-segment descriptive stats |
| `outputs/phase_segmentation_v2_vs_baseline.csv` | V2 vs Phase 9 baseline comparison |

## Key Findings

1. **Churn spread 0 pp → 43.9 pp:** The V2 segmentation now fully distinguishes
   high-risk (Dormant Buyers, 43.9% churn) from low-risk (Mid-Tier Active, 1.2% churn).
2. **Champions are the largest and highest-value segment** (48.7% of customers,
   £2,320 average revenue) — but also have an 8% churn rate worth monitoring.
3. **Mid-Tier Active have the highest purchase frequency** (order_rate = 1.02/month)
   despite moderate revenue — strong candidates for upsell.
4. **Non-purchasers are structurally distinct:** Median recency = 474 days indicates
   they signed up but never converted — an acquisition funnel problem, not a retention one.

## Business Recommendations

| Segment | Priority Action |
|---|---|
| **Champions** | Loyalty deepening; cross-sell high-margin categories; monitor 8% churn risk |
| **Mid-Tier Active** | Upsell to higher AOV; activate on high-frequency purchase cadence |
| **Dormant Buyers** | Win-back campaign; 43.9% churn = highest urgency reactivation target |
| **Non-Purchasers** | Review on-boarding flow; first-purchase incentive experiment |

Segment assignments should be refreshed monthly as customer behaviour evolves.
Use churn model scores (V2) alongside segment membership for campaign prioritisation.
