# Uplift Analysis — Evidence Audit Before V2

## Purpose

Interrogate the Phase 7 T-learner baseline and the `mart_campaign_response` mart for
leakage, covariate balance, label fitness, and treatment-effect heterogeneity. Every V2
design decision must trace back to a finding here.

---

## §1 Population Structure

11 campaigns with both treatment and control arms — all are usable.

| Campaign | Type | Channel | Total | Treated | Control | T-rate | T/C ratio |
|---|---|---|---|---|---|---|---|
| CMP011 | retention | email | 18,000 | 12,610 | 5,390 | 19.1% | 2.3× |
| CMP012 | discount | email | 18,000 | 15,263 | 2,737 | 20.1% | 5.6× |
| CMP005 | loyalty_bonus | email | 15,245 | 12,491 | 2,754 | 19.7% | 4.5× |
| CMP008 | seasonal_promo | app_push | 10,246 | 9,436 | 810 | 20.1% | 11.7× |
| CMP007 | retention | email | 8,322 | 6,848 | 1,474 | 18.4% | 4.6× |
| CMP004 | reactivation | email | 1,246 | 1,153 | 93 | 14.5% | 12.4× |

**Finding 1.A:** Every campaign has both treatment and control arms — no campaign is excluded.

**Finding 1.B:** T/C ratio varies from ~2.3× (CMP011) to ~12× (CMP004, CMP008). Heavy
imbalance in smaller campaigns will make per-campaign uplift estimates noisy.

**Finding 1.C:** ~22% of rows come from rule-based targeting (not RCT). Rule-targeted rows
are not randomly assigned; T-learner models trained on these rows will conflate targeting
rules with customer response propensity. Use `targeting_source` as a covariate.

---

## §2 Label Audit

Five candidate outcome columns were evaluated.

**Finding 2.A:** `response_flag_30d` ≡ `conversion_within_30d` — 100% agreement. Drop
the duplicate; use `response_flag_30d` as the training target.

**Finding 2.B:** 30-day and 7-day response windows give near-identical overall ATE, but
the 7-day window has tighter z-stats (lower noise). 30d is retained as primary label
to capture delayed conversions.

**Finding 2.C:** Per-campaign ATE varies from < 0 pp (some campaigns *hurt* response rate)
to > 8 pp. A pooled model that ignores campaign identity will average out this
heterogeneity and produce misleading decile rankings.

**Finding 2.D:** Revenue lift = £4.47/customer (30d). A revenue-uplift V2 variant is
feasible; blocked for now because ATE stability is needed first.

---

## §3 Leakage Audit

**Hard leakage (label or perfect derivatives) — excluded:**
- `conversion_within_30d` (= the label)
- `conversion_within_7d` (derived from same event)
- `response_bucket` / `response_rank` (label-derived bucketing)

**Post-treatment outcomes — excluded (realised after assignment):**
- `campaign_revenue`, `realized_uplift`, `email_opens`, `email_clicks`, `delivered`

**Finding 3.C:** Engagement signals (delivered/open/click/unsubscribe) sit at AUC ≈ 0.50 —
they carry no pre-treatment information and must be excluded from features.

**Finding 3.D:** Phase 7 baseline already excluded all of the above. No regression risk;
feature list is confirmed clean.

---

## §4 Feature Fitness

**Finding 4.A:** Customer-level pre-treatment features (`pre_90d_orders`, `pre_90d_revenue`,
`pre_90d_aov`) are the strongest predictors of response — customers who recently purchased
are more likely to respond. These are safe and should anchor V2 feature engineering.

**Finding 4.B:** Campaign-level features (`campaign_type`, `channel`, `offer_type`,
`targeting_source`) are important for learning cross-campaign response patterns, but
a pooled T-learner trained without campaign identity cannot distinguish between a
discount-channel effect and a high-propensity customer effect.

**Finding 4.C:** A naive pooled T-learner confuses campaign-mix effects with customer
effects. V2 must include campaign identity features (type, channel) explicitly.

---

## §5 Baseline Failure Analysis

**Finding 5.A:** Both baseline T-learner models (Logistic Regression, Random Forest) have
**no ranking ability**. The decile chart is essentially flat — the top decile is not
meaningfully more responsive than the bottom decile.

**Finding 5.B:** Top-vs-bottom decile spread ≈ 4 pp (logistic regression) and ≈ 0 pp
(random forest). A working uplift model should produce spreads of 10–20 pp or more.

**Finding 5.C:** Cumulative Qini curve hugs the random-targeting diagonal. The Qini area
is near zero — the model provides no targeting value over random selection.

**Finding 5.D:** Response AUC proxy ≈ 0.50–0.52. The underlying response models in the
T-learner cannot distinguish responders from non-responders — the base prediction is
no better than chance.

**Finding 5.E:** Root cause — T-learner with 14 thin features (mostly campaign metadata)
lacks the customer behavioural depth to estimate heterogeneous treatment effects.
Pre-treatment purchase velocity, recency, and engagement features are needed.

---

## §6 V2 Design Decisions

| Dimension | Phase 7 Baseline | V2 Decision |
|---|---|---|
| Model family | T-learner Logistic/RF | T-learner LightGBM |
| Feature depth | 14 features (mostly campaign metadata) | 30+ features incl. pre-treatment behaviour |
| Campaign identity | Not included | campaign_type, channel, offer_type as features |
| Split | Time-ordered (already correct) | Retained; stratify by campaign |
| Evaluation | Decile table, Qini area | Qini AUC, per-campaign ATE, decile spread |
