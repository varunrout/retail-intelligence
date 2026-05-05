# Churn Analysis — Evidence Audit Before V2

## Purpose

Audit the Phase 6 churn baseline and underlying `mart_customer_features` data to
expose structural and feature-level limitations, and to design a V2 grounded in evidence.

---

## §1 Population Structure

**Dataset:** 50,000 customers × 40 columns. Churn rate: 17.12%. Signup dates 2022-07-10 → 2025-12-21.

**Key finding — non-purchasers:**
5,568 customers (11.1% of mart) have never placed an order. Their `churn_flag_90d` is
always 0 because the label is defined as 90 days without an *order* — you cannot churn
if you never purchased. Including non-purchasers in the training population trains the
model on a structural impossibility (label is always 0 regardless of features).

| Population | n | Churn rate |
|---|---|---|
| Full mart | 50,000 | 17.12% |
| Active (purchased at least once) | 44,432 | **19.27%** |
| Non-purchasers | 5,568 | 0% (by construction) |

**Decision:** Model on the active population only (44,432 customers). Non-purchasers
are a separate CRM problem (acquisition / first-purchase conversion).

---

## §2 Feature Signal Audit

Approved features available: 34 (after Phase 6 leakage blacklist).

**Top 5 features by churn spread (max-quintile minus min-quintile churn rate):**

| Feature | Type | Churn Spread |
|---|---|---|
| customer_segment_seed | categorical | 76.78 pp |
| total_orders | numeric | 37.79 pp |
| total_units | numeric | 36.99 pp |
| total_discount_amount | numeric | 36.83 pp |
| total_net_revenue | numeric | 36.58 pp |

**Bottom 5 (weakest signal):**

| Feature | Type | Churn Spread |
|---|---|---|
| preferred_channel | categorical | 0.68 pp |
| city_tier | categorical | 0.96 pp |
| region | categorical | 1.21 pp |
| loyalty_tier | categorical | 1.80 pp |
| is_marketing_opt_in | categorical | 1.91 pp |

**Recency leakage confirmation:** `recency_days` single-feature ROC AUC = **1.0000**.
This is not a useful feature — it *is* the label (a customer with recent_days > 90 is
by definition churned). Phase 6 baseline already excluded it; confirmed here.

---

## §3 Null Patterns (Active Population)

| Column | Meaning When Null | % Null | Churn (null) | Churn (present) | Delta |
|---|---|---|---|---|---|
| loyalty_tier | Not enrolled in loyalty programme | 45.6% | 23.7% | 15.5% | +8.2 pp |
| campaigns_targeted | Never targeted by a campaign | 17.4% | 24.7% | 18.1% | +6.6 pp |
| total_sessions | No web session data | 0.1% | 52.6% | 19.2% | +33.4 pp |

**Finding:** Nulls are informative — unenrolled customers and never-targeted customers
churn at materially higher rates. Null indicators should be explicit features, not just
imputed to zero.

---

## §4 Derived Features

Three engineered features pass the leakage audit and show strong signal:

| Feature | Formula | Churn Spread |
|---|---|---|
| order_velocity_per_month | total_orders / tenure_months | **52.7 pp** |
| purchase_session_rate | sessions_with_purchase / total_sessions | 30.0 pp |
| cart_to_purchase_rate | purchases / cart_adds | 26.8 pp |

`recency_to_tenure_ratio` was **excluded** despite passing the null check — it
encodes recency implicitly (AUROC 0.921) and approaches label leakage.

---

## §5 Baseline Limitations

- **Phase 6 used a stratified random split** — no time-ordering. A customer's future
  behaviour can influence feature values computed from all history, creating implicit
  lookahead. V2 must use a time-ordered split.
- **Split point for V2:** `signup_date = 2025-03-21`
  - Train: customers signed up before 2025-03-21 (n ≈ 35,545)
  - Test: customers signed up on/after 2025-03-21 (n ≈ 8,887)
- **Non-purchaser contamination:** Baseline trained on 50K customers including 5,568
  non-purchasers, diluting the signal in the active churn-at-risk population.

---

## §6 V2 Design Decisions

| Dimension | Phase 6 Baseline | V2 Decision |
|---|---|---|
| Population | Full 50K (incl. non-purchasers) | Active 44,432 only |
| Split | Stratified random 80/20 | Time-ordered at 2025-03-21 |
| Features | 34 approved | 34 approved + 3 engineered + null indicators |
| Model | Logistic Regression, Random Forest | LightGBM with Optuna tuning |
| Threshold | Max-F1 / precision-floor | Precision-floor ≥ 0.70 priority |
