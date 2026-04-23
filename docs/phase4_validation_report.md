# Phase 4 Validation Report

## Status

Completed. All mart-level contracts pass in the latest run.

## Latest Execution

- Runner: src/data/run_phase4_validation.py
- Timestamp context: generated after latest phase 3 mart refresh
- Output artifacts:
	- outputs/phase4_validation_details.csv
	- outputs/phase4_validation_summary.csv
	- outputs/phase4_feature_dictionary.csv

## Mart Validation Summary

- mart_campaign_response: PASS (13 checks, 0 failed)
- mart_customer_features: PASS (17 checks, 0 failed)
- mart_product_demand: PASS (15 checks, 0 failed)
- mart_recommendation_interactions: PASS (13 checks, 0 failed)
- mart_returns_risk: PASS (16 checks, 0 failed)
- mart_store_week_performance: PASS (16 checks, 0 failed)

## Notes On Rules Applied

1. Key uniqueness and null-in-key checks run for every mart.
2. Non-negative constraints are applied to commercial and volume metrics.
3. 0-1 range checks are applied to rate-like fields.
4. Date null checks are applied only to required date columns for each mart.
5. Feature dictionary export is generated as a governance artifact for downstream analysis.

## How To Reproduce

1. Run phase 3 mart build:
- python -m src.data.run_phase3_marts

2. Run phase 4 validation:
- python -m src.data.run_phase4_validation

3. Review generated outputs:
- outputs/phase4_validation_details.csv
- outputs/phase4_validation_summary.csv
- outputs/phase4_feature_dictionary.csv

## Interpretation Guide

- status = PASS means all checks for that mart passed.
- status = FAIL means one or more checks failed and should be investigated before modeling.
- failed_checks indicates number of failed checks at mart level.

## Governance Note

This validation report is the required checkpoint before Phase 5 analysis notebooks or any Phase 6+ modeling scripts consume mart data.
