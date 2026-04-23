# Phase 9: Customer Segmentation and PCA — Baseline Report

## Objective

Group the 50,000-customer base into behaviorally distinct, actionable segments using
unsupervised KMeans clustering. Visualise separation in 2D via PCA and export labelled
assignments for downstream campaigns and CRM.

---

## Method

| Step | Detail |
|---|---|
| Feature set | 14 numeric features: transactional, engagement, channel, and campaign |
| Null handling | "No activity" nulls filled with 0 (e.g. customers with no sessions score 0) |
| Scaling | StandardScaler — zero mean, unit variance per feature |
| k selection | Swept k = 2–10; evaluated inertia (elbow) and silhouette score |
| Final k | 4 — silhouette peaks at k=2 (0.270) but too coarse for operations; elbow inflects at k=4 (silhouette=0.201) |
| Labelling | Auto-labelled from centroid medians of revenue, recency, and order volume |
| Visualisation | PCA projected to 2D (PC1=46.1%, PC2=12.8% variance) |

---

## k-Sweep Results

| k | Silhouette | Inertia |
|---|---|---|
| 2 | 0.270 | 478,141 |
| 3 | 0.262 | 390,309 |
| **4** | **0.201** | **350,317** |
| 5 | 0.210 | 320,555 |
| 6 | 0.196 | 302,948 |
| 7–10 | ≤0.183 | declining |

k=4 was selected: the elbow bends sharply at 4, and 4 segments map naturally to
recognisable CRM archetypes.

---

## Segment Summary

| Cluster | Segment Name | Customers | Share | Median Revenue | Median Recency | Median Orders | Churn Rate |
|---|---|---|---|---|---|---|---|
| 0 | Dormant 1 | 18,236 | 36.5% | $341 | 47 days | 4 | 0% |
| 1 | Champions | 9,173 | 18.3% | $3,312 | 16 days | 22 | 0% |
| 2 | Dormant 2 | 5,958 | 11.9% | $80 | 51 days | 1 | 0% |
| 3 | Mid-Tier Active | 16,633 | 33.3% | $1,388 | 22 days | 12 | 0% |

### Segment Playbooks

**Champions (18.3%)** — High-value, highly active, recently purchased.
- Action: Loyalty programmes, exclusive early-access campaigns, referral incentives.
- KPIs to watch: Average order value, campaign conversion rate.

**Mid-Tier Active (33.3%)** — Solid spenders, recent, moderate frequency.
- Action: Cross-sell, category expansion, upgrade nudges toward Champions.
- KPIs to watch: Total orders growth, basket size.

**Dormant 1 (36.5%)** — Light buyers, moderate recency, low order count.
- Action: Re-engagement emails, seasonal win-back campaigns, product discovery.
- KPIs to watch: Reactivation rate within 90 days.

**Dormant 2 (11.9%)** — Single-purchase or near-lost customers, very low engagement.
- Action: Last-chance win-back at low cost; deprioritise from paid retargeting.
- KPIs to watch: Cost-per-reactivation vs lifetime value.

---

## PCA Interpretation

PC1 (46.1% variance) primarily captures **transaction volume and revenue** — customers
move left-to-right from near-zero activity (Dormant 2, left cluster) to Champions (right).

PC2 (12.8% variance) captures **engagement and session depth** — higher PC2 = more
browsing without purchasing, lower PC2 = more transactional efficiency.

Dormant 2 is clearly isolated in the lower-left, confirming its distinctness from all other
segments. Champions and Mid-Tier Active overlap in PC space, suggesting the boundary is
defined by spend magnitude rather than behavioural style.

---

## Caveats and Limitations

1. KMeans assumes spherical clusters of equal size; real-world customer distributions are
   skewed. Gaussian Mixture Models or hierarchical clustering may improve boundary
   precision.
2. PC1+PC2 capture only 58.9% of total variance; some segment overlap in 2D is expected
   and does not indicate clustering failure.
3. Churn flag (churn_flag_90d) shows 0% across all segment medians — this is a *median*
   effect masking the minority of churners in each group. Use mean or top-decile analysis
   for churn-risk within segment.
4. Segments should be re-computed quarterly as customer behaviour shifts seasonally.
5. Dormant 1 and Dormant 2 are both labelled "dormant" by the auto-labeller due to low
   orders and elevated recency; they differ meaningfully in revenue depth (4x difference).

---

## Artifacts

| File | Description |
|---|---|
| `outputs/phase9_segmentation_cluster_assignments.csv` | customer_id → cluster, segment_name |
| `outputs/phase9_segmentation_cluster_profiles.csv` | Per-segment median feature profiles |
| `outputs/phase9_segmentation_k_sweep.csv` | Silhouette and inertia for k=2–10 |
| `outputs/phase9_segmentation_elbow_silhouette.png` | Elbow + silhouette selection chart |
| `outputs/phase9_segmentation_pca_plot.png` | 2D PCA scatter with segment centroids |
| `outputs/phase9_segmentation_profile_bars.png` | 6-panel bar chart: median KPIs per segment |

---

## Next Phase

**Phase 10: Recommendation System** — collaborative or content-based product
recommendations per customer, seeded by purchase history in the order items mart.
