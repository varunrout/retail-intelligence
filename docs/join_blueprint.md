# Join Blueprint

## Purpose

This document defines approved join paths, cardinality expectations, and guardrails to prevent row inflation and leakage.

## Core Join Paths

1. Customer transaction path
- Path: customers -> orders -> order_items -> products.
- Join keys: customers.customer_id = orders.customer_id; orders.order_id = order_items.order_id; order_items.product_id = products.product_id.
- Cardinality: 1:M, 1:M, M:1.
- Validation: all joins passed orphan checks with zero failed rows.

2. Returns path
- Path: order_items -> returns.
- Join key: order_items.order_item_id = returns.order_item_id.
- Cardinality: 1:0..M depending on operational return processing.
- Validation: returns without matching order_item_id = 0.

3. Campaign path
- Path: campaigns -> campaign_targets -> campaign_events.
- Join keys: campaign_id and campaign_id + customer_id.
- Cardinality: campaigns to targets is 1:M; targets to events should be 1:0..1 after dedup policy.
- Validation: events without target = 0; note duplicate events on campaign_id + customer_id = 205.

4. Store and channel path
- Path: orders -> stores.
- Join key: orders.store_id_nullable = stores.store_id for store channel orders.
- Cardinality: M:1.
- Validation: store channel orders with null or unresolved store_id = 0.

5. Inventory and pricing path
- Path: daily_inventory -> products and daily_prices -> products.
- Join key: product_id.
- Cardinality: M:1.
- Validation: all rows resolved to products.

6. Session behavior path
- Path: web_sessions -> session_events and optionally to customers using customer_id_nullable.
- Join keys: session_id and customer_id_nullable.
- Cardinality: web_sessions to session_events is 1:M.
- Guardrail: customer_id_nullable is missing for a subset of sessions/events and must not be used as a hard-required key.

## Join Guardrails

1. Never join order-level and item-level tables without defining final grain first.
2. For campaign events, aggregate or deduplicate to campaign_id + customer_id before uplift features.
3. Preserve channel logic when joining orders to stores; only store channel orders should enforce store_id.
4. For session data, separate known-customer behavior from anonymous behavior.
5. When creating weekly marts, derive a single week key from calendar and apply consistently across modules.

## Validation Evidence

Join checks are exported in outputs/phase2_join_validation_summary.csv and currently all checks pass.