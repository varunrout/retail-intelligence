# Phase 3 Mart Contracts

## Purpose

Define grain, keys, refresh logic, and downstream consumers for each mart created in Phase 3.

## 1) mart_customer_features
- Grain: one row per customer_id.
- Primary key: customer_id.
- Refresh logic: rebuild from full historical data each run.
- Inputs: customers, completed orders, order_items, returns, web_sessions, campaign targets/events.
- Consumers: churn modeling, segmentation, retention prioritization.

## 2) mart_product_demand
- Grain: one row per week_start_date, product_id, store_id_or_online.
- Primary key: week_start_date + product_id + store_id_or_online.
- Refresh logic: rebuild full history; compatible with weekly incremental extension later.
- Inputs: completed orders + order_items, daily_prices, daily_inventory, products.
- Consumers: forecasting, assortment and pricing analytics.

## 3) mart_campaign_response
- Grain: one row per campaign_id, customer_id assignment.
- Primary key: campaign_id + customer_id.
- Refresh logic: rebuild from full campaign assignment/event history.
- Inputs: campaigns, campaign_targets, deduplicated campaign_events, pre-assignment order history.
- Consumers: uplift modeling, campaign effectiveness diagnostics.

## 4) mart_returns_risk
- Grain: one row per order_item_id.
- Primary key: order_item_id.
- Refresh logic: rebuild from completed orders and returns.
- Inputs: order_items, completed orders, returns, products, customers.
- Consumers: returns risk prediction, anomaly detection.

## 5) mart_recommendation_interactions
- Grain: one row per customer_id, product_id.
- Primary key: customer_id + product_id.
- Refresh logic: rebuild from purchases, session events, and reviews.
- Inputs: completed order interactions, session_events, reviews, products, product_attributes.
- Consumers: popularity/content/collaborative recommendation layers.

## 6) mart_store_week_performance
- Grain: one row per week_start_date, store_id_or_online.
- Primary key: week_start_date + store_id_or_online.
- Refresh logic: rebuild from transactional, inventory, and pricing weekly aggregates.
- Inputs: completed orders, order_items, daily_inventory, daily_prices, stores.
- Consumers: store performance tracking, forecasting support, anomaly monitoring.

## SQL Design Standards Applied

- CTEs to stage transformation logic.
- CASE WHEN bands and operational flags.
- Window functions for lag, rolling, rank, and running metrics.
- Deduplication policy for campaign events.
- Channel-aware handling for online versus physical store rows.
