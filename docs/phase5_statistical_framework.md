# Phase 5 Statistical Framework

## Objective

Establish the statistical decision layer for the Retail Growth Intelligence System before moving into model-heavy phases.

## Hypotheses And Decision Questions

1. Churn and value
- H1: High recency and weaker engagement are associated with higher churn probability.
- Decision use: prioritize retention interventions by risk and expected value.

2. Campaign incrementality
- H1: Treatment response rate exceeds control response rate in 30-day window.
- Decision use: target customers likely to be persuadable, not just likely buyers.

3. Returns behavior
- H1: Deep discounts are associated with lower item net price and altered return behavior profile.
- Decision use: set discount guardrails and return-risk screening.

4. Demand behavior
- H1: Weekly demand exhibits autocorrelation and can benefit from lag-based forecasting features.
- Decision use: use lag-aware demand planning and error tracking.

## Guardrails And Practical Significance

- Campaign uplift practical threshold: >= 2.0 percentage points.
- Return-rate drift alert threshold: >= 1.0 percentage point change versus recent baseline.
- Forecast operating guardrail: monitor naive baseline gap and trigger model review if deterioration persists for 3 consecutive cycles.
- Churn prioritization rule: use risk + value views jointly; do not action churn scores in isolation.

## Implemented Statistical Components

1. Confidence intervals
- Average order value (mean CI)
- Campaign response rate 30d (proportion CI)
- Item-level return rate (proportion CI)
- Weekly units sold mean (mean CI)

2. Hypothesis tests
- Welch t-test: item net price, deep discount vs low discount groups
- Two-proportion z-test: treatment vs control response rate

3. Time-series diagnostics
- Weekly total demand series with lag-1 naive baseline
- ACF checks at lag-1 and lag-4

4. Bayesian update
- Beta-Binomial posterior for treatment conversion probability with 95% credible interval

## Artifact Outputs

- outputs/phase5_stat_summary.csv
- outputs/phase5_weekly_demand_diagnostics.csv
- outputs/phase5_chart_registry.csv
- outputs/phase5_chart_churn_distribution.png
- outputs/phase5_chart_demand_trend_by_category.png
- outputs/phase5_chart_return_rate_by_discount_band.png
- outputs/phase5_chart_campaign_response_by_segment.png
- outputs/phase5_chart_customer_segment_scatter.png
- outputs/phase5_chart_feature_importance_churn.png
- outputs/phase5_chart_forecast_vs_actual.png

## Readiness For Next Phases

Phase 5 provides the statistical and business framing needed for:
- Phase 6 churn modeling
- Phase 7 uplift modeling
- Phase 8 demand forecasting
- Phase 11 anomaly threshold tuning
