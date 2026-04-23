# Business Question To Data Map

## Purpose

This map links each business question to source tables, required joins, and intended mart outputs.

## Question Mapping

1. Which customers are likely to churn soon?
- Sources: customers, orders, order_items, web_sessions, session_events, reviews, returns, calendar.
- Core joins: customer transaction path plus customer behavior path.
- Target mart: customer feature mart.

2. Which customers should receive a retention offer for incremental impact?
- Sources: campaigns, campaign_targets, campaign_events, customers, orders.
- Core joins: campaign path plus customer transaction aggregates.
- Target mart: campaign response mart and customer feature mart.

3. What products should be recommended to each customer?
- Sources: order_items, orders, products, product_attributes, web_sessions, session_events, reviews.
- Core joins: product interaction path from transactions and behavior.
- Target mart: recommendation interaction mart.

4. What will future demand look like at product, store, and week level?
- Sources: order_items, orders, products, stores, daily_inventory, daily_prices, calendar.
- Core joins: transaction + pricing + inventory + calendar week mapping.
- Target mart: product demand mart and store-week performance mart.

5. Which entities show unusual behavior?
- Sources: returns, order_items, orders, daily_inventory, daily_prices, campaign_events, session_events.
- Core joins: returns path and operational trend path.
- Target mart: returns risk mart and store-week performance mart.

6. What customer segments exist, and how should business act on them?
- Sources: customers, orders, order_items, web_sessions, session_events, returns, reviews.
- Core joins: customer transaction and behavior path.
- Target mart: customer feature mart.

## Cross-Cutting Keys

- Customer key: customer_id.
- Order key: order_id.
- Order line key: order_item_id.
- Product key: product_id.
- Campaign assignment key: campaign_id + customer_id.
- Date/period key: canonical date + derived week key from calendar.

## Phase 3 Dependency

Phase 3 marts must preserve these mappings so every model output can be traced to explicit source data paths.