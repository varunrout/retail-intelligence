# Phase 3 SQL Validation

## Execution Summary

- Runner: src/data/run_phase3_marts.py
- SQL scripts executed: staging_shared.sql + 6 mart scripts
- Build status: success
- DuckDB output database: data/processed/retail_intelligence.duckdb

## Mart Row Counts

Source: data/processed/phase3_mart_row_counts.csv

- mart_campaign_response: 102,593
- mart_customer_features: 50,000
- mart_product_demand: 639,673
- mart_recommendation_interactions: 2,468,038
- mart_returns_risk: 1,246,512
- mart_store_week_performance: 5,512

## Grain And Key Validation Checks

Source: data/processed/phase3_mart_quality_checks.csv

1. mart_customer_features_duplicate_customer_id: PASS
2. mart_product_demand_duplicate_grain: PASS
3. mart_campaign_response_duplicate_grain: PASS
4. mart_returns_risk_duplicate_order_item: PASS
5. mart_reco_duplicate_customer_product: PASS
6. mart_store_week_duplicate_grain: PASS
7. mart_customer_features_null_customer_id: PASS
8. mart_campaign_response_null_campaign_or_customer: PASS

## Additional Notes

- Campaign events are deduplicated in staging via stg_campaign_events_dedup before campaign mart assembly.
- Channel-aware logic is applied for store and online records in demand and store-week marts.
- Window-based lag and rolling fields are included for forecasting and ranking-oriented downstream modules.

## Phase 3 Exit Decision

Phase 3 is complete and ready for Phase 4 because:

1. All required marts were built and exported to data/processed.
2. Core mart grain and key quality checks passed.
3. Mart contracts and validation documentation are in place.
