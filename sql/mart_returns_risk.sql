CREATE OR REPLACE TABLE mart_returns_risk AS
WITH item_base AS (
    SELECT
        oi.order_item_id,
        oi.order_id,
        oi.product_id,
        oi.quantity,
        oi.item_list_price,
        oi.item_discount_pct,
        oi.item_net_price,
        oi.item_cost,
        oi.item_margin,
        oi.fulfillment_type,
        o.customer_id,
        o.order_date,
        o.channel,
        o.payment_type,
        o.net_amount AS order_net_amount,
        p.category,
        p.subcategory,
        p.price_tier,
        p.return_risk_seed,
        c.region,
        c.loyalty_tier,
        r.return_id,
        r.return_request_date,
        r.return_processed_date,
        r.return_reason,
        r.refund_amount,
        r.return_status,
        r.return_channel
    FROM stg_order_items oi
    JOIN stg_completed_orders o
        ON oi.order_id = o.order_id
    LEFT JOIN stg_products p
        ON oi.product_id = p.product_id
    LEFT JOIN stg_customers c
        ON o.customer_id = c.customer_id
    LEFT JOIN stg_returns r
        ON oi.order_item_id = r.order_item_id
),
item_with_flags AS (
    SELECT
        *,
        CASE WHEN return_id IS NOT NULL THEN 1 ELSE 0 END AS return_flag,
        CASE
            WHEN item_discount_pct >= 0.35 THEN 'deep_discount'
            WHEN item_discount_pct >= 0.15 THEN 'medium_discount'
            ELSE 'low_discount'
        END AS discount_band,
        DATE_DIFF('day', order_date, return_request_date) AS days_to_return
    FROM item_base
),
item_with_history AS (
    SELECT
        *,
        AVG(return_flag) OVER (
            PARTITION BY customer_id
            ORDER BY order_date, order_item_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS prior_customer_return_rate,
        SUM(return_flag) OVER (
            PARTITION BY product_id
            ORDER BY order_date, order_item_id
            ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
        ) AS recent_product_return_events
    FROM item_with_flags
)
SELECT
    order_item_id,
    order_id,
    customer_id,
    product_id,
    order_date,
    channel,
    payment_type,
    category,
    subcategory,
    price_tier,
    region,
    loyalty_tier,
    quantity,
    item_list_price,
    item_discount_pct,
    item_net_price,
    item_cost,
    item_margin,
    fulfillment_type,
    return_flag,
    return_id,
    return_reason,
    return_status,
    return_channel,
    refund_amount,
    days_to_return,
    discount_band,
    return_risk_seed,
    COALESCE(prior_customer_return_rate, 0.0) AS prior_customer_return_rate,
    COALESCE(recent_product_return_events, 0) AS recent_product_return_events,
    CASE
        WHEN COALESCE(prior_customer_return_rate, 0.0) >= 0.35 OR item_discount_pct >= 0.35 THEN 'high'
        WHEN COALESCE(prior_customer_return_rate, 0.0) >= 0.15 OR item_discount_pct >= 0.15 THEN 'medium'
        ELSE 'low'
    END AS return_risk_band,
    ROW_NUMBER() OVER (
        PARTITION BY customer_id
        ORDER BY order_date DESC, order_item_id DESC
    ) AS customer_item_recency_rank
FROM item_with_history;
