# Data Quality Audit

## Audit Basis

- Table and date profile: outputs/phase2_table_profile_summary.csv.
- Key checks: outputs/phase2_key_checks_summary.csv.
- Join checks: outputs/phase2_join_validation_summary.csv.

## High-Level Results

1. Source completeness
- 16 raw source tables are present.
- Candidate key null rate is 0.0 across all checked tables.

2. Key uniqueness
- All declared keys are unique except campaign_events on campaign_id + customer_id.
- Duplicate rows on campaign_events key: 205.

3. Referential integrity
- All defined orphan checks passed with failed_rows = 0.
- This supports safe baseline mart construction in Phase 3.

## Distribution And Bias Checks

1. Channel distribution in orders
- Online: 392,826 rows (73.94%).
- Store: 138,418 rows (26.06%).
- Implication: channel imbalance should be modeled explicitly.

2. Loyalty representation in customers
- None: 25,004 (50.01%).
- Bronze: 11,782 (23.56%).
- Silver: 7,791 (15.58%).
- Gold: 4,173 (8.35%).
- Platinum: 1,250 (2.50%).
- Implication: segmentation and churn models should avoid overfitting to non-loyalty majority.

3. Campaign treatment/control balance
- Treatment rows: 85,811.
- Control rows: 16,782.
- Treatment-to-control ratio: 5.113.
- Implication: uplift evaluation should use weighting/stratification and decile diagnostics.

4. Known identity missingness in digital data
- web_sessions missing customer_id_nullable: 70,376 rows (12.49%).
- session_events missing customer_id_nullable: 210,909 rows (9.17%).
- Implication: anonymous traffic should be handled as separate cohorts.

## Value Sanity Checks

- Negative refunds: 0.
- Zero refunds: 0.
- Negative item_net_price: 0.
- Non-positive quantity: 0.
- Invalid discount_pct outside [0,1]: 0.
- Observed discount_pct range: 0.0000 to 0.5344.

## Time Coverage Checks

- Main calendar window: 2024-01-01 to 2025-12-31.
- Returns and reviews extend into 2026 due to lag effects.
- Policy: forecasting marts should align on calendar window while post-purchase risk modules can include lag spillover.

## Data Treatment Policy For Phase 3+

1. Deduplicate campaign_events to one row per campaign_id + customer_id before feature generation.
2. Preserve anonymous digital behavior as separate aggregate features instead of dropping rows.
3. Keep channel-specific logic in marts and model splits.
4. Enforce date-window consistency per module to prevent leakage.
5. Apply explicit handling for post-period returns/reviews when training churn or demand models.
