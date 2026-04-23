# Phase 2 Decisions Log

## Objective

Capture assumptions and decisions made during schema audit so downstream work is consistent and interview-defensible.

## Final Decisions

1. SQL execution layer
- Decision: use DuckDB views over raw CSVs for audit and baseline mart development.
- Reason: fast local execution with large flat files and minimal setup overhead.

2. Canonical business date policy
- Decision: use order_date or order_datetime-derived date as the canonical transaction date for customer and sales features.
- Reason: prevents ambiguity between fulfillment and return timelines.

3. Weekly aggregation policy
- Decision: derive week keys from calendar table fields and apply one week definition across marts.
- Reason: ensures consistency between forecasting and performance marts.

4. Campaign event deduplication policy
- Decision: aggregate/deduplicate campaign_events to campaign_id + customer_id before uplift feature engineering.
- Reason: 205 duplicate records on this key would otherwise bias treatment response rates.

5. Identity missingness policy for digital behavior
- Decision: keep records with customer_id_nullable missing and aggregate separately for anonymous traffic signals.
- Reason: dropping anonymous sessions would hide acquisition and browse behavior patterns.

6. Channel conditioning policy
- Decision: preserve explicit channel logic in feature engineering and evaluation.
- Reason: orders are 73.94% online and 26.06% store, so mixed modeling without channel controls can bias decisions.

7. Explainability policy under current runtime
- Decision: use permutation importance as baseline; SHAP is optional if runtime compatibility is solved later.
- Reason: current environment does not reliably install SHAP.

## Open Follow-Ups

1. Confirm whether campaign_events should be deduplicated by latest status, max engagement flag, or summed interaction counts.
2. Define the exact churn observation and prediction windows before Phase 6 model training.
3. Decide whether returns/reviews beyond 2025-12-31 are included as lag features or excluded for strict period alignment.
