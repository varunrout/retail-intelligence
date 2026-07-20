# Anomaly Detection Analysis — Evidence Audit Before V2

## Purpose

Evidence-first analysis of the `mart_returns_risk` data before designing Anomaly
Detection V2. The goal is to understand the abuse signal, confirm label availability,
and establish why the Phase 10/11 unsupervised approach underperformed.

---

## §1 Data Profile

| Metric | Value |
|---|---|
| Total returns | 188,399 |
| Abuse positives | 1,352 |
| Prevalence | **0.72%** — severe class imbalance |
| `loyalty_tier` null rate | 29.5% |

**Finding 1.A:** 1,352 abuse positives from 188,399 returns (0.72% prevalence). Severe
class imbalance: approximately 138 negatives per positive. `scale_pos_weight ≈ 125–138`
is required for LightGBM.

**Finding 1.B:** `loyalty_tier` is 29.5% null in the returns mart. Impute to explicit
`"unknown"` tier rather than dropping — null-tier customers may have a distinct
abuse profile.

**Finding 1.C — Label availability confirmed:** `returns_hidden_labels.csv` is available
and contains synthetic seed-derived abuse labels (not manual annotations). Phase 10/11 used unsupervised methods only
(Isolation Forest, LOF) despite these labels being accessible. The supervised path is
valid and unlocks a materially higher performance ceiling.

---

## §2 Feature Signals

**Finding 2.A — Top numeric predictors (single-feature AUROC):**

| Feature | AUROC |
|---|---|
| item_net_price | 0.925 |
| refund_amount | 0.924 |
| item_margin | 0.913 |

High-value, high-margin items are the dominant abuse pattern — consistent with
"buy-use-return" / wardrobing behaviour on premium products.

**Finding 2.B — Category concentration:**
Electronics = 75.5% of all abuse returns vs only 19% of non-abuse returns.
Low-discount band = 59.6% of abuse returns. The archetypal abuse pattern is:
*high-price electronics at full/near-full price, returned as "suspected_abuse".*

**Finding 2.C — return_reason signal:**
`return_reason = suspected_abuse` shows an abuse rate of 5.7% vs 0.5% baseline (+11×).
This is a legitimate feature (not circular): the reason is recorded independently of
the hidden `is_abuse` label by a different operational process.

---

## §3 Baseline Failure Analysis

**Finding 3.A:** Phase 10/11 used unsupervised-only methods (IF AP=0.058, LOF AP=0.080)
despite `returns_hidden_labels.csv` being available. This was the primary source of
underperformance — unsupervised methods cannot specialise to the specific abuse patterns
in the data.

**Finding 3.B:** IQR rules flag 12.04% of all returns (22,683 transactions) at only
2.74% precision. A fraud team acting on this queue would spend 97% of its time reviewing
legitimate returns. Operationally unworkable.

**Finding 3.C:** Isolation Forest AP = 0.058 on the full feature set. LOF is limited to
a 50,000-row subsample due to O(n²) complexity. Neither baseline uses the strongest
categorical signals (`category`, `return_reason`) because they are string-typed.

---

## §4 Supervised Validation

Cross-validated LightGBM results on available labels:

| Model | CV AP | Notes |
|---|---|---|
| Isolation Forest | 0.058 | Unsupervised baseline |
| Logistic Regression | 0.258 | 4× IF; linear on 12 features |
| **LightGBM** | **0.791 ± 0.017** | **14× IF; non-linear + categorical features** |

**Finding 4.A:** LightGBM CV AP = 0.791 ± 0.017 — 14× improvement over Isolation Forest.
Even a linear logistic regression (AP = 0.258) is 4× better than unsupervised approaches,
confirming that the label, not the algorithm complexity, is the key bottleneck.

**Finding 4.B:** At 1% flag rate: precision = 33.4%, recall = 79.2%, F1 = 47.0%.
Compared to Phase 11 IQR rules at 12% flag rate / 2.7% precision, the supervised model
flags 12× fewer returns while catching 1.7× more abuse cases.

**Finding 4.C:** Time-ordered split is required. Features `prior_customer_return_rate`
and `cust_abuse_rate` encode temporal history — future returns must not influence past
feature values. A random split would leak these aggregates.

---

## §5 Repeat Abuser Pattern

**Finding 5.A:** `cust_abuse_rate` (customer historical abuse density) shows AUROC = 0.996.
There are 147 customers with highly elevated abuse rates — some have 37 out of 68 returns
flagged as abuse. Repeat abusers are a discrete sub-population with near-perfect model
separability.

**Finding 5.B:** The `high_value_low_discount` interaction (electronics at full price
returned) is the archetypal abuse pattern. Electronics category + `is_suspected_abuse_reason`
+ high net price together account for the majority of the high-confidence abuse cases.

---

## §6 V2 Design Decisions

| Dimension | Phase 11 Baseline | V2 Decision |
|---|---|---|
| Supervision | Unsupervised (IF, LOF) | Supervised LightGBM with `is_abuse` label |
| Split | N/A (unsupervised) | Time-ordered 80/20 by `order_date` |
| Class weight | N/A | `scale_pos_weight = 125` |
| Features | Numeric only | 12 features: numeric + binary categoricals (is_electronics, is_suspected_abuse_reason, is_high_risk_band, is_low_discount) |
| Threshold | 12% flag rate (IQR) | 1–2% flag rate via score percentile |
| Eval metric | Precision @ flag rate | Average Precision (AP), PR curve |

**V2 feature set (12 features):**
`item_net_price`, `refund_amount`, `prior_customer_return_rate`, `item_discount_pct`,
`customer_item_recency_rank`, `item_margin`, `recent_product_return_events`,
`days_to_return`, `is_suspected_abuse_reason`, `is_electronics`, `is_low_discount`,
`is_high_risk_band`
