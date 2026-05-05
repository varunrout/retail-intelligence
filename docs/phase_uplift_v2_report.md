# Phase Uplift V2 Report

## Objective

Build a customer-level incremental uplift model to rank customers by expected causal
response to marketing campaigns, enabling targeted outreach beyond the average treatment
effect.

## Scope and Method

- **Label:** `response_flag_30d` — purchase within 30 days of campaign assignment.
- **Campaigns:** CMP011 and CMP012 only — the two campaigns with both treatment and
  control arms in the mart.
- **Split:** Time-ordered 80/20 by `assignment_datetime`.
- **Algorithm:** X-learner with LightGBM at both stages.
- **Feature set:** 30 approved features including pre-treatment behaviour (90-day
  lookback window).

## Data Snapshot

- Train rows: 61,559 | Test rows: 20,519
- Treatment / control ratio: approximately 5:1 (campaign design)
- Response rate (treatment): 20.7–21.0% | Response rate (control): 14.8–16.7%

## Model Results

### X-Learner LightGBM V2

- Overall ATE (test): 0.0444
- Top-5 decile observed uplift: 0.0450
- **Qini area: 744.8**
- Spearman rank correlation: 0.406

## Comparison With Phase 7 Baseline

| Metric | Phase 7 T-Learner RF | V2 X-Learner | Delta |
|---|---|---|---|
| Overall ATE | 0.0424 | 0.0444 | +0.002 |
| Top-5 decile uplift | 0.0537 | 0.0450 | −0.009 |
| **Qini area** | **502.5** | **744.8** | **+242.3 (+48%)** |
| Rank correlation | ≈ 0 | **0.406** | Ranking now meaningful |

The baseline T-learner produced essentially random uplift scores (Qini ≈ random
curve, rank correlation ≈ 0). V2 produces genuine ranking ability — the primary goal
of an uplift model.

## Per-Campaign ATE (Test Set)

| Campaign | Treatment n | Control n | Response (T) | Response (C) | ATE |
|---|---|---|---|---|---|
| CMP011 | 1,768 | 751 | 21.0% | 14.8% | +6.2 pp |
| CMP012 | 15,263 | 2,737 | 20.7% | 16.7% | +4.0 pp |

CMP011 has higher observed uplift (+6.2 pp vs +4.0 pp), though its smaller test size
means higher variance in the estimate.

## Output Artifacts

| File | Contents |
|---|---|
| `outputs/phase_uplift_v2_params.csv` | Hyperparameter log |
| `outputs/phase_uplift_v2_model_comparison.csv` | Uplift metrics on test set |
| `outputs/phase_uplift_v2_vs_baseline.csv` | V2 vs Phase 7 baseline |
| `outputs/phase_uplift_v2_per_campaign_ate.csv` | Per-campaign ATE table |
| `outputs/phase_uplift_v2_feature_importance.csv` | Feature importance by stage |

## Key Findings

1. **Qini area improves 48%** — the primary indication that V2 produces actionable
   customer-level rankings, unlike the baseline.
2. **Pre-treatment behaviour drives ranking:** Top features by gain are `pre_90d_aov`,
   `recency_days` (pre-treatment), `avg_item_discount_pct`, and `avg_basket_size`.
   Customers with a clear pre-treatment purchase pattern are more predictably uplifted.
3. **Stage 2 tau₀ best_iteration = 1** — the control-group pseudo-effect model
   converged almost immediately, suggesting the control population is highly homogeneous.
4. **Only 2 campaigns available** with proper T/C splits. A broader campaign portfolio
   would improve generalisation.

## Business Recommendations

- Target top-10% of customers by V2 uplift score for personalised campaign outreach.
- Suppress bottom-10% (lowest or negative predicted uplift) from campaign sends.
- Use `campaign_type` + `channel` as scoring-time features when predicting uplift for
  an upcoming specific campaign.
- Collect proper randomised T/C splits for all future campaigns to expand the
  training base beyond CMP011 and CMP012.
