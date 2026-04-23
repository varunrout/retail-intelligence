# Phase 7 Uplift Baseline Report

## Objective

Estimate incremental campaign impact and produce decile-level targeting guidance using treatment and control assignments from `mart_campaign_response`.

## Scope And Method

- Response target: `response_flag_30d`.
- Treatment flag: `treatment_flag`.
- Split: time-ordered 80/20 by `assignment_datetime`.
- Leakage controls:
  - Removed post-treatment behavior fields (opens, clicks, conversions, realized campaign revenue, response buckets/ranks).
  - Exported leakage audit and approved feature list.
- Modeling approach: T-learner baseline.
  - Fit one response model on treated customers.
  - Fit one response model on control customers.
  - Score uplift as `p(response|treated) - p(response|control)`.
- Models compared:
  - T-learner Logistic Regression.
  - T-learner Random Forest.

## Data Snapshot

- Total rows: 102,593
- Train rows: 82,074
- Test rows: 20,519
- Approved feature count: 14
- Treatment share:
  - Train: 83.80%
  - Test: 83.00%

## Global Campaign Effect

- Control response rate: 15.7371% (2,641 / 16,782)
- Treatment response rate: 19.9753% (17,141 / 85,811)
- Overall uplift (ATE): +4.2382 percentage points
- Two-proportion z-test: z = 12.7278, p-value ~ 0

Conclusion: campaign effect is statistically significant and practically meaningful.

## Uplift Model Comparison (Test)

### T-learner Logistic Regression

- Top-3 decile observed uplift: 0.0524
- Top-5 decile observed uplift: 0.0432
- Total incremental responses estimate: 891.50
- Qini-like area: 462.43
- Response AUC proxy: 0.5231
- Response PR AUC proxy: 0.2106

### T-learner Random Forest

- Top-3 decile observed uplift: 0.0398
- Top-5 decile observed uplift: 0.0537
- Total incremental responses estimate: 927.08
- Qini-like area: 502.48
- Response AUC proxy: 0.5052
- Response PR AUC proxy: 0.2019

## Interpretation

- Both uplift baselines identify positive incremental impact on average.
- Logistic model is stronger in top-3 decile concentration.
- Random Forest is stronger on cumulative incremental-response metrics.
- Decile uplift curve is positive but not monotonic, so the current baseline should be used as a ranking starter, not final policy.

## Decision Guidance

- If strategy prioritizes highest-confidence top bucket: use T-learner Logistic Regression.
- If strategy prioritizes broader cumulative gain: consider T-learner Random Forest.
- In either case, require campaign-level monitoring and periodic recalibration.

## How To Use Deciles In Campaign Operations

- Decile 1 represents the top 10% of customers ranked by predicted uplift score, meaning the model believes this group is most likely to respond because of treatment.
- Decile 10 represents the bottom 10% of customers by predicted uplift score, meaning treatment is expected to add the least incremental value there.
- The primary operational check is `observed_uplift` in each decile, defined as treatment response rate minus control response rate within that decile.

### How To Read The Charts

- `phase7_uplift_by_decile.png`: shows whether the model is concentrating incremental lift near the top of the ranked list.
- A strong ranking model usually has higher observed uplift in earlier deciles and lower uplift in later deciles.
- In this baseline, all deciles remain positive, which confirms that treatment is broadly helpful, but the curve is not monotonic, which means ranking quality is useful but still noisy.
- `phase7_uplift_cumulative_incremental.png`: shows the running total of estimated incremental responses as additional deciles are included.
- A steeper early rise indicates that the model is concentrating more incremental value in top-ranked customers.

### How To Translate Deciles Into Action

- If campaign budget is tight, begin with deciles 1 to 3 and validate realized uplift against a holdout or control group.
- If campaign budget is broader, extend to deciles 4 to 5 only if incremental gain remains above the business threshold for profitability.
- Use the top-3 uplift metric when selecting a model for narrow, high-confidence targeting.
- Use cumulative incremental response metrics when selecting a model for broader campaign deployment.
- Keep a live control policy in production so uplift estimates remain measurable after deployment.

### Practical Reading Of Current Results

- Logistic T-learner is the better choice if the business wants the strongest concentration in the top few deciles.
- Random Forest T-learner is the better choice if the business wants stronger cumulative gain across a wider targeted group.
- Because the decile curve is not monotonic, this phase should be treated as a baseline prioritization framework rather than a fully optimized targeting policy.

## Caveats

- Treatment/control allocation is imbalanced; ranking quality should be monitored per campaign.
- Uplift estimates are observational within randomized assignment framework but still baseline-level.
- Recommended next refinement: campaign-wise normalization or monotonic decile smoothing.

## Artifacts

- `outputs/phase7_uplift_leakage_audit.csv`
- `outputs/phase7_uplift_approved_features.csv`
- `outputs/phase7_uplift_overall_summary.csv`
- `outputs/phase7_uplift_model_comparison.csv`
- `outputs/phase7_uplift_decile_summary.csv`
- `outputs/phase7_uplift_scored_sample_top500.csv`
- `outputs/phase7_uplift_by_decile.png`
- `outputs/phase7_uplift_cumulative_incremental.png`
