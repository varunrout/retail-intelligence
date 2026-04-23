CREATE OR REPLACE TABLE mart_campaign_response AS
WITH pre_assignment_orders AS (
    SELECT
        t.campaign_id,
        t.customer_id,
        t.assignment_datetime,
        COUNT(DISTINCT o.order_id) AS pre_90d_orders,
        SUM(o.net_amount) AS pre_90d_revenue,
        AVG(o.net_amount) AS pre_90d_aov
    FROM stg_campaign_targets t
    LEFT JOIN stg_completed_orders o
        ON t.customer_id = o.customer_id
       AND o.order_datetime >= t.assignment_datetime - INTERVAL '90 day'
       AND o.order_datetime < t.assignment_datetime
    GROUP BY t.campaign_id, t.customer_id, t.assignment_datetime
),
base AS (
    SELECT
        t.campaign_id,
        c.campaign_name,
        c.campaign_type,
        c.channel AS campaign_channel,
        c.offer_type,
        c.offer_strength,
        t.customer_id,
        t.assignment_datetime,
        t.treatment_flag,
        t.control_flag,
        t.predicted_business_segment_at_send,
        t.targeting_rule_source,
        e.delivered_flag,
        e.open_flag,
        e.click_flag,
        e.unsubscribe_flag,
        e.conversion_within_7d,
        e.conversion_within_30d,
        COALESCE(e.revenue_within_7d, 0.0) AS revenue_within_7d,
        COALESCE(e.revenue_within_30d, 0.0) AS revenue_within_30d,
        COALESCE(e.source_event_rows, 0) AS source_event_rows,
        p.pre_90d_orders,
        p.pre_90d_revenue,
        p.pre_90d_aov
    FROM stg_campaign_targets t
    LEFT JOIN stg_campaign_events_dedup e
        ON t.campaign_id = e.campaign_id
       AND t.customer_id = e.customer_id
    LEFT JOIN stg_campaigns c
        ON t.campaign_id = c.campaign_id
    LEFT JOIN pre_assignment_orders p
        ON t.campaign_id = p.campaign_id
       AND t.customer_id = p.customer_id
)
SELECT
    campaign_id,
    campaign_name,
    campaign_type,
    campaign_channel,
    offer_type,
    offer_strength,
    customer_id,
    assignment_datetime,
    treatment_flag,
    control_flag,
    predicted_business_segment_at_send,
    targeting_rule_source,
    delivered_flag,
    open_flag,
    click_flag,
    unsubscribe_flag,
    conversion_within_7d,
    conversion_within_30d,
    revenue_within_7d,
    revenue_within_30d,
    source_event_rows,
    pre_90d_orders,
    pre_90d_revenue,
    pre_90d_aov,
    CASE
        WHEN conversion_within_30d = 1 THEN 1
        ELSE 0
    END AS response_flag_30d,
    CASE
        WHEN treatment_flag AND conversion_within_30d = 1 THEN 'persuaded_or_sure_thing'
        WHEN treatment_flag AND conversion_within_30d = 0 THEN 'treated_no_response'
        WHEN control_flag AND conversion_within_30d = 1 THEN 'organic_converter'
        ELSE 'non_converter'
    END AS response_bucket,
    ROW_NUMBER() OVER (
        PARTITION BY campaign_id
        ORDER BY conversion_within_30d DESC, revenue_within_30d DESC, customer_id
    ) AS campaign_response_rank,
    NTILE(10) OVER (
        PARTITION BY campaign_id
        ORDER BY revenue_within_30d DESC, conversion_within_30d DESC, customer_id
    ) AS revenue_decile_within_campaign
FROM base;
