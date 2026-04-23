# Phase 5 Decision Memo

## Executive Summary

The statistical decision layer indicates that retention, campaign optimization, and demand planning actions are justified and should proceed with controlled thresholds. Results show meaningful campaign uplift, stable uncertainty bounds for key metrics, and strong temporal dependence in demand.

## Key Measured Results

1. Customer economics
- Average order value: 126.87
- 95% CI: [126.24, 127.50]

2. Campaign outcomes
- 30-day response rate: 19.28%
- 95% CI: [19.04%, 19.52%]
- Treatment vs control uplift: +4.24 percentage points
- Two-proportion z-test: statistically significant (p < 0.001)

3. Returns profile
- Item-level return rate: 15.11%
- 95% CI: [15.05%, 15.18%]
- Deep vs low discount item net price difference: statistically significant (Welch t-test p < 0.001)

4. Demand behavior
- Mean weekly units sold at mart grain: 2.239
- 95% CI: [2.235, 2.243]
- Demand autocorrelation remains high at lag-1 (0.667) and lag-4 (0.647), supporting lag-based forecasting design.

5. Bayesian conversion view
- Treatment conversion posterior mean: 19.98%
- 95% credible interval: [19.71%, 20.24%]

## Business Recommendations

1. Retention actions
- Prioritize customers with higher recency and high value exposure for retention interventions.
- Use churn risk alongside value band and engagement features to avoid over-targeting low-value churners.

2. Campaign targeting
- Continue treatment strategy with tighter segment-level prioritization based on measured uplift.
- Move next-phase uplift modeling to decile-level targeting recommendations.

3. Discount and return governance
- Add discount-band monitoring to return-risk workflows.
- Require anomaly review for high-discount cohorts with rising return signals.

4. Forecasting operations
- Use lag-aware models as baseline in forecasting phase and monitor against naive benchmark.

## Risks And Caveats

- Campaign treatment/control imbalance remains material and must be controlled in Phase 7 uplift modeling.
- Churn analyses rely on proxy churn definition from recency threshold and should be revisited for policy alignment.
- Anonymous session behavior is only partially represented in customer-linked analyses.

## Immediate Next Step

Proceed to Phase 6 baseline churn modeling using the validated feature set and thresholds established in Phase 5.
