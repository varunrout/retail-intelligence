# Phase 10: Return Fraud and Anomaly Detection — Baseline Report

## Objective

Identify anomalous and potentially fraudulent return transactions using unsupervised
methods — no labels, no assumptions about which specific patterns constitute fraud.
Establish a labelled benchmark to enable supervised improvement in V2.

---

## Scope and Method

| Step | Detail |
|---|---|
| Target | Return transactions flagged as potential abuse (`returns` table) |
| Evaluation label | `is_abuse` (binary, derived from `return_reason = suspected_abuse`) |
| Split | Ordered by `return_date`; last 20% as test |
| Train rows | 150,719 |
| Test rows | 37,680 |
| Train prevalence | 0.79% |
| Test prevalence | 0.42% |
| Evaluation metric | Average Precision (AP), Precision, Recall, F1 at 1% flag rate |

All three baselines are **unsupervised** — they receive no abuse labels during fitting.

---

## Baseline Methods Compared

### 1. IQR / Rules (heuristic)

A simple rule-based filter using interquartile range outlier detection on
`refund_amount` and `item_discount_pct`, combined with a `suspected_abuse` reason flag.

- Flag rate: **12.04%** (casts a very wide net — 22,683 of 188,399 returns flagged)
- Precision: 0.0274 — only 2.7% of flagged returns are actual abuse
- Recall: 0.4601
- F1: 0.0518
- AUC-PR: 0.0165

**Problem:** Near-useless precision. A fraud team acting on this queue would waste 97%
of their reviews on legitimate returns. The high recall is achieved by flagging 12% of all
returns, not through genuine signal.

### 2. Isolation Forest

Unsupervised anomaly isolation on numerical features. No labels used.

- Flag rate: **1.0%**
- Precision: 0.1008
- Recall: 0.1405
- F1: 0.1174
- AUC-PR: 0.0580

Better precision than IQR rules, but recall is low — most abusive returns are not
isolated as outliers because fraudsters blend into the statistical norm of the return
distribution.

### 3. Local Outlier Factor (LOF, n=20)

Density-based local outlier detection evaluated on a 50,000-row subsample.

- Flag rate: **1.0%**
- Precision: 0.1680
- Recall: 0.0621
- F1: 0.0907
- AUC-PR: 0.0799

LOF achieves the highest precision of the three baselines at a 1% flag rate, but recall
is very low. LOF runs on a subsample due to O(n²) memory and time complexity.

---

## Phase 10 Summary

| Method | Flag Rate | Precision | Recall | F1 | AUC-PR |
|---|---|---|---|---|---|
| IQR / Rules | 12.04% | 0.0274 | 0.4601 | 0.0518 | 0.0165 |
| Isolation Forest | 1.0% | 0.1008 | 0.1405 | 0.1174 | 0.0580 |
| LOF (n=20) | 1.0% | 0.1680 | 0.0621 | 0.0907 | 0.0799 |

**Key finding:** None of the unsupervised baselines achieve actionable precision.
The best unsupervised AP is 0.0799 (LOF). A supervised approach with the abuse label
as ground truth is the clear next step.

---

## Interpretation

- Fraudulent returns are not statistical outliers in feature space — they follow plausible
  return patterns, which is by design. Isolation Forest and LOF therefore have limited
  power.
- The IQR rule's 46% recall at 12% flag rate is essentially noise — the base rate is
  0.42%, so random flagging at 12% would already capture ≈29% of abuse cases by chance.
- The abuse label (`return_reason = suspected_abuse`) is available in the raw data and
  can be used as a supervised target in V2 without data leakage, provided it is not used
  as a direct feature.

---

## Caveats

- `is_abuse` is derived from self-reported `return_reason`. Ground truth is imperfect:
  some genuine fraud may be reported as other reasons; some `suspected_abuse` may be
  legitimate returns.
- LOF results are on a 50,000-row subsample; full-dataset LOF would be prohibitively slow.
- Unsupervised models have no mechanism to learn the specific patterns (electronics +
  high refund + short days_to_return + repeat customer) that define abuse in this dataset.

---

## Artifacts

| File | Description |
|---|---|
| `outputs/phase11_anomaly_eval.csv` | Precision / recall / F1 / AUC-PR for all three baselines |
| `outputs/phase11_anomaly_comparison.png` | PR curve comparison chart |
| `outputs/phase11_if_score_distribution.png` | Isolation Forest score distribution |
| `outputs/phase11_pr_curves.png` | Precision-recall curves |
| `outputs/phase11_review_queue.csv` | Top-flagged return transactions from Isolation Forest |

---

## Next Phase

**Phase 10 → V2 (Supervised LightGBM):** Use `is_abuse` as the supervised target with
a time-ordered train/test split. Engineer 12 features around pricing, customer history,
return timing, and categorical risk signals. Target 1% flag rate with materially higher
precision and recall than all unsupervised baselines.
