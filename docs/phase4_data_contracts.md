# Phase 4 Data Contracts

## Purpose

Define the Python-side contracts that govern how mart data is loaded and validated before analysis and modeling.

## Contract Components

Each mart contract specifies:

1. Required columns
- Columns that must exist for the mart to be considered valid.

2. Key columns
- Columns that define the expected uniqueness grain.

3. Date columns
- Columns parsed as date/datetime and checked for null date rows.

4. Non-negative columns
- Metrics that must never be negative.

5. Zero-to-one bounded columns
- Percentage or rate fields expected in [0, 1].

## Implemented Contract Files

- src/data/mart_schemas.py
- src/data/mart_loaders.py
- src/data/mart_validators.py
- src/data/feature_dictionary.py
- src/data/run_phase4_validation.py

## Mart-Level Contract Summary

1. mart_customer_features
- Key: customer_id
- Date columns: signup_date, first_order_date, last_order_date
- Focus checks: churn features, recency, revenue and behavioral aggregates

2. mart_product_demand
- Key: week_start_date + product_id + store_id_or_online
- Date columns: week_start_date
- Focus checks: demand units, revenue, pricing and inventory bands

3. mart_campaign_response
- Key: campaign_id + customer_id
- Date columns: assignment_datetime
- Focus checks: treatment/control outcomes and pre-period baseline fields

4. mart_returns_risk
- Key: order_item_id
- Date columns: order_date
- Focus checks: item-level return labels and risk covariates

5. mart_recommendation_interactions
- Key: customer_id + product_id
- Date columns: first_interaction_date, last_interaction_date
- Focus checks: interaction score and ranking fields

6. mart_store_week_performance
- Key: week_start_date + store_id_or_online
- Date columns: week_start_date
- Focus checks: store-week revenue, unit, and operational indicators

## Runtime Outputs

Phase 4 runner writes:

- outputs/phase4_validation_details.csv
- outputs/phase4_validation_summary.csv
- outputs/phase4_feature_dictionary.csv

## Intended Usage

1. Run Phase 3 marts first.
2. Run python -m src.data.run_phase4_validation.
3. Inspect summary file before starting Phase 5 or model training notebooks.
4. Treat FAIL statuses as blockers unless explicitly accepted and documented.
