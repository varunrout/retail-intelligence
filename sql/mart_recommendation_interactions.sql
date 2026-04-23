CREATE OR REPLACE TABLE mart_recommendation_interactions AS
WITH purchase_interactions AS (
    SELECT
        o.customer_id,
        oi.product_id,
        o.order_date AS interaction_date,
        'purchase' AS interaction_source,
        SUM(oi.quantity) AS interaction_count,
        SUM(oi.item_net_price) AS interaction_value,
        SUM(oi.quantity) * 3.0 AS weighted_score
    FROM stg_completed_orders o
    JOIN stg_order_items oi
        ON o.order_id = oi.order_id
    GROUP BY o.customer_id, oi.product_id, o.order_date
),
event_interactions AS (
    SELECT
        se.customer_id_nullable AS customer_id,
        se.product_id_nullable AS product_id,
        CAST(se.event_time AS DATE) AS interaction_date,
        'session_event' AS interaction_source,
        COUNT(*) AS interaction_count,
        0.0 AS interaction_value,
        SUM(
            CASE
                WHEN LOWER(se.event_type) IN ('add_to_cart', 'begin_checkout') THEN 2.0
                WHEN LOWER(se.event_type) IN ('view_product', 'product_view') THEN 1.0
                WHEN LOWER(se.event_type) IN ('purchase') THEN 3.0
                ELSE 0.5
            END
        ) AS weighted_score
    FROM stg_session_events se
    WHERE se.customer_id_nullable IS NOT NULL
      AND se.product_id_nullable IS NOT NULL
    GROUP BY se.customer_id_nullable, se.product_id_nullable, CAST(se.event_time AS DATE)
),
review_interactions AS (
    SELECT
        customer_id,
        product_id,
        review_date AS interaction_date,
        'review' AS interaction_source,
        COUNT(*) AS interaction_count,
        AVG(rating) AS interaction_value,
        SUM(COALESCE(rating, 0)) AS weighted_score
    FROM stg_reviews
    WHERE customer_id IS NOT NULL
      AND product_id IS NOT NULL
    GROUP BY customer_id, product_id, review_date
),
all_interactions AS (
    SELECT * FROM purchase_interactions
    UNION ALL
    SELECT * FROM event_interactions
    UNION ALL
    SELECT * FROM review_interactions
),
aggregated AS (
    SELECT
        customer_id,
        product_id,
        COUNT(*) AS active_days,
        SUM(interaction_count) AS total_interactions,
        SUM(interaction_value) AS total_interaction_value,
        SUM(weighted_score) AS interaction_score,
        MAX(interaction_date) AS last_interaction_date,
        MIN(interaction_date) AS first_interaction_date
    FROM all_interactions
    GROUP BY customer_id, product_id
),
interaction_anchor AS (
    SELECT MAX(last_interaction_date) AS max_interaction_date
    FROM aggregated
)
SELECT
    a.customer_id,
    a.product_id,
    p.category,
    p.subcategory,
    p.price_tier,
    p.brand_type,
    pa.color,
    pa.material_type,
    pa.recommendation_embedding_group,
    a.active_days,
    a.total_interactions,
    a.total_interaction_value,
    a.interaction_score,
    a.first_interaction_date,
    a.last_interaction_date,
    DATE_DIFF('day', a.last_interaction_date, ia.max_interaction_date) AS recency_days,
    ROW_NUMBER() OVER (
        PARTITION BY a.customer_id
        ORDER BY a.interaction_score DESC, a.last_interaction_date DESC, a.product_id
    ) AS product_rank_for_customer,
    ROW_NUMBER() OVER (
        PARTITION BY a.customer_id, p.category
        ORDER BY a.interaction_score DESC, a.last_interaction_date DESC, a.product_id
    ) AS category_rank_for_customer
FROM aggregated a
CROSS JOIN interaction_anchor ia
LEFT JOIN stg_products p
    ON a.product_id = p.product_id
LEFT JOIN stg_product_attributes pa
    ON a.product_id = pa.product_id;
