# Source Table Dictionary

## Purpose

This document defines grain, key strategy, date fields, and business meaning for each raw source table.

## Table Definitions

### calendar
- Grain: one row per calendar date.
- Candidate key: date.
- Core date fields: date.
- Business role: time dimension for all trend, seasonality, and weekly rollup work.

### customers
- Grain: one row per customer.
- Candidate key: customer_id.
- Core date fields: signup_date, last_purchase_date.
- Business role: customer profile, loyalty, and channel preference context.

### orders
- Grain: one row per order.
- Candidate key: order_id.
- Core date fields: order_datetime, order_date.
- Business role: transaction header for revenue, channel, payment, and customer ordering behavior.

### order_items
- Grain: one row per order line item.
- Candidate key: order_item_id.
- Core date fields: item_fulfillment_date.
- Business role: SKU-level demand, margin, discount, and return-eligible unit details.

### products
- Grain: one row per product.
- Candidate key: product_id.
- Core date fields: launch_date, discontinued_date.
- Business role: core product taxonomy and commercial attributes.

### product_attributes
- Grain: one row per product.
- Candidate key: product_id.
- Core date fields: none.
- Business role: feature metadata for recommendation and segmentation inputs.

### stores
- Grain: one row per store.
- Candidate key: store_id.
- Core date fields: opening_date.
- Business role: physical retail footprint and store descriptors.

### daily_inventory
- Grain: one row per date, product, and location (store_id_or_online).
- Candidate key: date + product_id + store_id_or_online.
- Core date fields: date.
- Business role: inventory trajectory, stockout dynamics, and supply indicators.

### daily_prices
- Grain: one row per date, product, and location (store_id_or_online).
- Candidate key: date + product_id + store_id_or_online.
- Core date fields: date.
- Business role: price and discount history used for demand and returns analysis.

### campaigns
- Grain: one row per campaign.
- Candidate key: campaign_id.
- Core date fields: start_date, end_date.
- Business role: campaign metadata and treatment context.

### campaign_targets
- Grain: one row per campaign and customer assignment.
- Candidate key: campaign_id + customer_id.
- Core date fields: assignment_datetime.
- Business role: treatment/control assignment and eligibility logic.

### campaign_events
- Grain: one row per campaign and customer event summary.
- Candidate key used for checks: campaign_id + customer_id.
- Core date fields: event_date not present in source header; event timing is encoded via flags and conversion windows.
- Business role: delivery, engagement, and conversion outcomes.
- Data note: 205 duplicate rows exist on campaign_id + customer_id and should be deduplicated or aggregated in marts.

### returns
- Grain: one row per return transaction.
- Candidate key: return_id.
- Core date fields: return_request_date, return_processed_date.
- Business role: post-purchase risk, return reason, and refund behavior.

### reviews
- Grain: one row per review.
- Candidate key: review_id.
- Core date fields: review_date.
- Business role: sentiment and voice-of-customer signals.

### web_sessions
- Grain: one row per session.
- Candidate key: session_id.
- Core date fields: session_start, session_end.
- Business role: top-level digital journey and pre-purchase engagement context.

### session_events
- Grain: one row per session event.
- Candidate key: event_id.
- Core date fields: event_time.
- Business role: clickstream-level behavior and product/category interaction signals.

## Observed Data Window

- Earliest observed calendar date: 2024-01-01.
- Latest observed calendar date: 2025-12-31.
- Returns and reviews extend beyond calendar into early 2026 due to post-purchase lag.
