# Uplift V2 — Model Design and Implementation

## Overview

X-learner (two-stage meta-learner) using LightGBM at both stages, with enriched
pre-treatment customer features and campaign identity features. Addresses the Phase 7
failure of T-learner having no decile ranking ability.

---

## Source Files

| File | Role |
|---|---|
| `src/features/features_uplift.py` | Feature engineering, leakage audit, label selection |
| `src/models/train_uplift.py` | X-learner training, Qini evaluation, decile analysis |
| `src/data/run_phase_uplift_v2.py` | End-to-end pipeline runner |

---

## Model Architecture — X-Learner

The X-learner improves over the T-learner by using pseudo-treatment-effects to recalibrate
individual uplift estimates, particularly in settings where treatment and control group
sizes are imbalanced.

**Stage 1 — Response models:**
- `μ₁(x)`: LightGBM on treated customers → predicts P(response | treated)
- `μ₀(x)`: LightGBM on control customers → predicts P(response | control)

**Stage 2 — Imputed effects:**
- For treated customers: `D₁ = Y₁ − μ₀(x)` (actual outcome minus control model prediction)
- For control customers: `D₀ = μ₁(x) − Y₀` (treatment model prediction minus actual outcome)
- Fit regressors `τ₁(x)` on D₁ (treated) and `τ₀(x)` on D₀ (control)

**Final uplift score:**
`τ(x) = e(x) · τ₁(x) + (1 − e(x)) · τ₀(x)`
where `e(x)` is the treatment propensity score.

---

## Hyperparameters

**Stage 1 (binary classification):**

| Parameter | Value |
|---|---|
| objective | binary |
| learning_rate | 0.05 |
| num_leaves | 63 |
| min_data_in_leaf | 50 |
| feature_fraction | 0.8 |
| bagging_fraction | 0.8 |
| best_iteration (τ₁) | 36 |
| best_iteration (τ₀) | 1 |

**Stage 2 (regression on imputed effects):**
Same architecture as Stage 1 with `objective = regression (rmse)`.

---

## Feature Set

30 approved features. Key additions over Phase 7:

| Feature Group | Features |
|---|---|
| Pre-treatment behaviour | `pre_90d_orders`, `pre_90d_revenue`, `pre_90d_aov`, `recency_days` |
| Engagement | `total_sessions`, `online_order_share`, `return_rate_per_unit` |
| Campaign identity | `campaign_type`, `channel`, `offer_type`, `targeting_source` |
| Customer profile | `tenure_days`, `avg_basket_size`, `avg_item_discount_pct` |

Excluded (post-treatment / label): `email_opens`, `email_clicks`, `campaign_revenue`,
`conversion_within_30d`, `response_bucket`, `response_rank`.

---

## Split Strategy

Time-ordered 80/20 by `assignment_datetime` (same as Phase 7 baseline).
Train: 61,559 rows | Test: 20,519 rows.
Stratified by campaign to ensure each campaign has representation in both train and test.

---

## Results

| Metric | X-Learner V2 | T-Learner RF Baseline | Delta |
|---|---|---|---|
| Overall ATE | 0.0444 | 0.0424 | +0.0020 |
| Top-5 Decile Observed Uplift | 0.0450 | 0.0537 | −0.009 |
| **Qini Area** | **744.8** | **502.5** | **+242.3 (+48%)** |
| Spearman Rank Corr | **0.406** | ~0 | significant ranking ability |

The primary gain is **ranking quality** — Qini area improves 48% and the model now
produces meaningful customer-level uplift scores (Spearman rank correlation = 0.406 vs
≈ 0 for the T-learner baseline).

**Per-campaign ATE (test set):**

| Campaign | Treatment n | Control n | Response (T) | Response (C) | ATE |
|---|---|---|---|---|---|
| CMP011 | 1,768 | 751 | 21.0% | 14.8% | +6.2 pp |
| CMP012 | 15,263 | 2,737 | 20.7% | 16.7% | +4.0 pp |

---

## Practical Targeting Guidance

The V2 model produces customer-level uplift scores. Recommended deployment:
- **Top 10% by uplift score:** Priority for personalised outreach — highest predicted
  incremental response.
- **Bottom 10% (or negative uplift):** Suppress from campaign — these customers may
  respond negatively to contact.
- **Per-campaign routing:** Use `campaign_type` + `channel` as features at score time
  to predict uplift for a specific upcoming campaign.
