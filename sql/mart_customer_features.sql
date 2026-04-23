CREATE OR REPLACE TABLE mart_customer_features AS
WITH anchor AS (
    SELECT MAX(order_date) AS anchor_date
    FROM stg_completed_orders
),
order_base AS (
    SELECT
        customer_id,
        order_id,
        order_date,
        channel,
        net_amount,
        discount_amount,
        basket_size,
        ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC, order_id DESC) AS order_recency_rank
    FROM stg_completed_orders
),
order_agg AS (
    SELECT
        customer_id,
        COUNT(DISTINCT order_id) AS total_orders,
        SUM(net_amount) AS total_net_revenue,
        AVG(net_amount) AS avg_order_value,
        SUM(discount_amount) AS total_discount_amount,
        AVG(basket_size) AS avg_basket_size,
        MIN(order_date) AS first_order_date,
        MAX(order_date) AS last_order_date,
        SUM(CASE WHEN LOWER(channel) = 'online' THEN 1 ELSE 0 END) AS online_orders,
        SUM(CASE WHEN LOWER(channel) = 'store' THEN 1 ELSE 0 END) AS store_orders
    FROM order_base
    GROUP BY customer_id
),
item_agg AS (
    SELECT
        o.customer_id,
        SUM(oi.quantity) AS total_units,
        AVG(oi.item_discount_pct) AS avg_item_discount_pct,
        AVG(oi.item_margin) AS avg_item_margin
    FROM stg_completed_orders o
    JOIN stg_order_items oi
        ON o.order_id = oi.order_id
    GROUP BY o.customer_id
),
return_agg AS (
    SELECT
        o.customer_id,
        COUNT(r.return_id) AS total_returns,
        SUM(r.refund_amount) AS total_refund_amount,
        AVG(DATE_DIFF('day', o.order_date, r.return_request_date)) AS avg_days_to_return
    FROM stg_completed_orders o
    JOIN stg_order_items oi
        ON o.order_id = oi.order_id
    LEFT JOIN stg_returns r
        ON oi.order_item_id = r.order_item_id
    GROUP BY o.customer_id
),
session_agg AS (
    SELECT
        customer_id_nullable AS customer_id,
        COUNT(*) AS total_sessions,
        AVG(DATE_DIFF('minute', session_start, session_end)) AS avg_session_minutes,
        AVG(pages_viewed) AS avg_pages_viewed,
        SUM(CASE WHEN add_to_cart_flag THEN 1 ELSE 0 END) AS sessions_add_to_cart,
        SUM(CASE WHEN purchase_flag THEN 1 ELSE 0 END) AS sessions_with_purchase
    FROM stg_web_sessions
    WHERE customer_id_nullable IS NOT NULL
    GROUP BY customer_id_nullable
),
campaign_agg AS (
    SELECT
        t.customer_id,
        COUNT(*) AS campaigns_targeted,
        SUM(CASE WHEN t.treatment_flag THEN 1 ELSE 0 END) AS campaigns_treatment,
        SUM(CASE WHEN e.conversion_within_30d = 1 THEN 1 ELSE 0 END) AS campaigns_converted_30d,
        SUM(COALESCE(e.revenue_within_30d, 0.0)) AS campaign_revenue_30d
    FROM stg_campaign_targets t
    LEFT JOIN stg_campaign_events_dedup e
        ON t.campaign_id = e.campaign_id
       AND t.customer_id = e.customer_id
    GROUP BY t.customer_id
)
SELECT
    c.customer_id,
    c.region,
    c.city_tier,
    c.preferred_channel,
    c.loyalty_tier,
    c.income_band,
    c.customer_segment_seed,
    c.is_marketing_opt_in,
    c.signup_date,
    oa.total_orders,
    oa.total_net_revenue,
    oa.avg_order_value,
    oa.total_discount_amount,
    oa.avg_basket_size,
    oa.first_order_date,
    oa.last_order_date,
    DATE_DIFF('day', oa.last_order_date, a.anchor_date) AS recency_days,
    DATE_DIFF('day', c.signup_date, a.anchor_date) AS tenure_days,
    CASE
        WHEN oa.total_orders > 0 THEN oa.total_net_revenue / oa.total_orders
        ELSE NULL
    END AS revenue_per_order,
    CASE
        WHEN oa.total_orders > 0 THEN oa.online_orders * 1.0 / oa.total_orders
        ELSE NULL
    END AS online_order_share,
    CASE
        WHEN oa.total_orders > 0 THEN oa.store_orders * 1.0 / oa.total_orders
        ELSE NULL
    END AS store_order_share,
    ia.total_units,
    ia.avg_item_discount_pct,
    ia.avg_item_margin,
    ra.total_returns,
    ra.total_refund_amount,
    ra.avg_days_to_return,
    CASE
        WHEN ia.total_units > 0 THEN COALESCE(ra.total_returns, 0) * 1.0 / ia.total_units
        ELSE 0.0
    END AS return_rate_per_unit,
    sa.total_sessions,
    sa.avg_session_minutes,
    sa.avg_pages_viewed,
    sa.sessions_add_to_cart,
    sa.sessions_with_purchase,
    ca.campaigns_targeted,
    ca.campaigns_treatment,
    ca.campaigns_converted_30d,
    ca.campaign_revenue_30d,
    CASE
        WHEN DATE_DIFF('day', oa.last_order_date, a.anchor_date) > 90 THEN 1
        ELSE 0
    END AS churn_flag_90d,
    CASE
        WHEN oa.total_net_revenue >= 1500 THEN 'high_value'
        WHEN oa.total_net_revenue >= 600 THEN 'mid_value'
        WHEN oa.total_net_revenue IS NULL THEN 'no_purchase'
        ELSE 'low_value'
    END AS customer_value_band,
    DENSE_RANK() OVER (
        PARTITION BY c.region
        ORDER BY COALESCE(oa.total_net_revenue, 0.0) DESC
    ) AS spend_rank_in_region
FROM stg_customers c
CROSS JOIN anchor a
LEFT JOIN order_agg oa
    ON c.customer_id = oa.customer_id
LEFT JOIN item_agg ia
    ON c.customer_id = ia.customer_id
LEFT JOIN return_agg ra
    ON c.customer_id = ra.customer_id
LEFT JOIN session_agg sa
    ON c.customer_id = sa.customer_id
LEFT JOIN campaign_agg ca
    ON c.customer_id = ca.customer_id;
