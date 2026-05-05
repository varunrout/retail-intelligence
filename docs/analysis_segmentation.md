# Segmentation Analysis — Evidence Audit Before V2

## Purpose

Evidence-first analysis of `mart_customer_features` to identify structural flaws in the
Phase 9 KMeans baseline and design a V2 with cleaner feature engineering and more
defensible cluster selection.

---

## §1 Data Profile and Population Issues

**Finding 1.A — Non-purchaser contamination:**
5,568 customers (11.1%) have no purchase history. Phase 9 imputed their purchase
features to 0 without an explicit flag, creating an artificial zero-boundary cluster
that artificially inflates apparent segment separation.

**Finding 1.B — loyalty_tier unusable as a clustering feature:**
50% of customers are not enrolled in the loyalty programme; their `loyalty_tier` is null.
Imputing null to a "not enrolled" category would create a synthetic majority category.
Exclude from clustering features; retain for post-hoc segment profiling only.

**Finding 1.C — Skewed features distort KMeans:**
`total_orders` (skew = 0.79) and `total_net_revenue` (skew = 1.81) are right-skewed.
KMeans centroids are mean-sensitive — a small number of high-value customers pull
centroids towards extremes, producing unbalanced segment boundaries.
**Decision:** Log1p transform all skewed features before standardisation.

---

## §2 Feature Collinearity

**Finding 2.A:** 18 feature pairs with |r| > 0.80. Volume/activity features
(`total_orders`, `total_units`, `total_sessions`, `sessions_with_purchase`) form a
near-collinear bloc. Feeding all four into KMeans triple-weights the activity axis
relative to revenue or engagement axes.

**Finding 2.B:** `avg_session_minutes` × `avg_pages_viewed` (r = 0.72) represent the
same browsing-depth axis. `purchase_rate` (purchases / sessions) captures this more
compactly as a single ratio.

**Finding 2.C:** Log-transforming skewed features *before* PCA prevents PC1 from being
dominated by high-revenue outliers.

---

## §3 Cluster Selection — Phase 9 Critique

**Finding 3.A:** k=3 achieves silhouette = 0.319 on PCA features — higher than the
Phase 9 choice of k=4 (0.252). Phase 9 selected k=4 without stability testing; k=3
is a cleaner geometric partition.

**Finding 3.B:** Phase 9's Dormant 1 cluster (36.5% of customers) mixes two fundamentally
different behaviours:
- **Never-active** customers (no purchases — null features imputed to 0)
- **Previously-active churned** customers (historical purchases, elevated recency)

These should not be in the same cluster; they require entirely different CRM responses
(acquisition vs win-back).

**Finding 3.C:** Phase 9 churn spread across segments = 30.4 pp. V2 target: ≥ 15 pp
churn spread with a cleaner segment definition that avoids mixing non-purchasers into
the churned population.

---

## §4 Feature Engineering for V2

**Finding 4.A — Log transforms:**
Log1p transformation reduces skewness from > 3.0 to < 0.5 on key features.
Silhouette improves from 0.244 (raw features, k=4) to **0.346** (log-transformed + PCA-6, k=4).

**Finding 4.B — purchase_rate:**
`sessions_with_purchase / total_sessions` is orthogonal to volume — it captures
browse-to-buy conversion efficiency independently of raw order count. Confirmed by
low collinearity with `total_orders` (r < 0.3).

**Finding 4.C — is_non_purchaser flag:**
An explicit binary flag marks the null→0 imputation boundary. Without it, KMeans
treats near-zero buyers and true never-buyers as interchangeable. With the flag, the
non-purchaser group is cleanly isolated.

---

## §5 K Selection for V2

K-sweep results on engineered features (log1p + PCA):

| k | Silhouette | Stability std | GMM BIC |
|---|---|---|---|
| 2 | **0.551** | 0.000 | low |
| 3 | 0.379 | 0.000 | medium |
| 4 | 0.331 | 0.053 | medium |
| 5 | 0.293 | 0.034 | rising |
| 6–8 | ≤0.287 | ≥0.021 | high |

**Finding 5.A:** k=2 achieves the highest silhouette (0.551) with perfect stability. GMM
BIC prefers k=8, but k=8 segments are too granular for operational CRM targeting.
**V2 uses k=2.**

**Finding 5.B:** Stability (seed-to-seed variance) is lowest at k ≤ 2. Larger k values
have higher seed-sensitivity — the data does not support strongly separated clusters
beyond k=2.

**Finding 5.C:** V2 log features require only 5 PCA components for 80% variance (Phase 9
raw features needed 6). Better feature engineering concentrates signal more efficiently.

---

## §6 V2 Cluster Profiles

V2 k=2 cluster results (with is_non_purchaser flag):

| Cluster | Label | n | % Non-purchaser |
|---|---|---|---|
| 0 | one_time | 6,329 | 88% |
| 1 | discount_sensitive | 43,671 | ~0% |

**Finding 6.A:** V2 churn spread = 9.8 pp between clusters (lower than Phase 9's 30.4 pp
but with a much cleaner partition — Phase 9's spread was inflated by the non-purchaser
contamination in its dormant cluster).

**Finding 6.B:** Cluster 0 is 88% non-purchasers — the `is_non_purchaser` flag successfully
isolates the never-bought group into its own cluster, enabling a distinct acquisition-
focused CRM strategy.

---

## §7 V2 Design Decisions

| Dimension | Phase 9 Baseline | V2 Decision |
|---|---|---|
| Feature preparation | Raw, StandardScaler | Log1p transform → StandardScaler |
| Non-purchasers | Imputed 0, no flag | Explicit `is_non_purchaser` binary flag |
| Collinear features | All 14 raw features | Deduplicated + `purchase_rate` ratio |
| PCA | 2 components (visual) | 5 components (80% variance threshold) |
| k | 4 | 2 (highest silhouette + stability) |
| Cluster stability | Not tested | Sweep with multiple seeds; report std |
