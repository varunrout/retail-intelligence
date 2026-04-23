CREATE OR REPLACE VIEW stg_calendar AS
SELECT
    date,
    day_of_week,
    week_of_year,
    month,
    quarter,
    year,
    is_weekend,
    is_month_end,
    is_payday_period,
    holiday_name,
    is_public_holiday,
    retail_season,
    promo_season_flag,
    DATE_TRUNC('week', date) AS week_start_date
FROM calendar;

CREATE OR REPLACE VIEW stg_customers AS
SELECT
    customer_id,
    signup_date,
    birth_year,
    gender,
    region,
    city_tier,
    acquisition_channel,
    preferred_channel,
    loyalty_tier,
    loyalty_signup_date,
    income_band,
    customer_segment_seed,
    is_marketing_opt_in
FROM customers;

CREATE OR REPLACE VIEW stg_orders AS
SELECT
    order_id,
    customer_id,
    order_datetime,
    order_date,
    channel,
    store_id_nullable,
    payment_type,
    order_status,
    gross_amount,
    discount_amount,
    net_amount,
    shipping_fee,
    device_type,
    basket_size,
    used_promo_code,
    promo_code_nullable,
    campaign_attributed_flag,
    session_id_nullable,
    fraud_risk_seed
FROM orders;

CREATE OR REPLACE VIEW stg_completed_orders AS
SELECT *
FROM stg_orders
WHERE LOWER(order_status) = 'completed';

CREATE OR REPLACE VIEW stg_order_items AS
SELECT
    order_item_id,
    order_id,
    product_id,
    quantity,
    item_list_price,
    item_discount_pct,
    item_net_price,
    item_cost,
    item_margin,
    fulfillment_type
FROM order_items
WHERE quantity > 0;

CREATE OR REPLACE VIEW stg_products AS
SELECT
    product_id,
    category,
    subcategory,
    brand_type,
    price_tier,
    base_price,
    cost,
    launch_date,
    product_style,
    seasonal_flag,
    premium_flag,
    return_risk_seed,
    margin_band
FROM products;

CREATE OR REPLACE VIEW stg_product_attributes AS
SELECT
    product_id,
    color,
    size_group,
    material_type,
    eco_flag,
    bundle_candidate_flag,
    recommendation_embedding_group,
    description_keywords
FROM product_attributes;

CREATE OR REPLACE VIEW stg_stores AS
SELECT
    store_id,
    region,
    city,
    store_type,
    opening_date,
    store_size_band,
    store_format,
    latitude_fake,
    longitude_fake
FROM stores;

CREATE OR REPLACE VIEW stg_daily_inventory AS
SELECT
    date,
    product_id,
    store_id_or_online,
    starting_inventory,
    ending_inventory,
    stock_received,
    stockout_flag,
    backorder_flag
FROM daily_inventory;

CREATE OR REPLACE VIEW stg_daily_prices AS
SELECT
    date,
    product_id,
    store_id_or_online,
    listed_price,
    discount_pct,
    promo_flag,
    markdown_flag,
    bundle_flag
FROM daily_prices;

CREATE OR REPLACE VIEW stg_campaigns AS
SELECT
    campaign_id,
    campaign_name,
    campaign_type,
    start_date,
    end_date,
    channel,
    offer_type,
    offer_strength,
    target_rule_summary,
    control_group_pct
FROM campaigns;

CREATE OR REPLACE VIEW stg_campaign_targets AS
SELECT
    campaign_id,
    customer_id,
    assignment_datetime,
    treatment_flag,
    control_flag,
    eligibility_flag,
    predicted_business_segment_at_send,
    targeting_rule_source
FROM campaign_targets
WHERE eligibility_flag = TRUE;

CREATE OR REPLACE VIEW stg_campaign_events_dedup AS
SELECT
    campaign_id,
    customer_id,
    MAX(CASE WHEN delivered_flag THEN 1 ELSE 0 END) AS delivered_flag,
    MAX(CASE WHEN open_flag THEN 1 ELSE 0 END) AS open_flag,
    MAX(CASE WHEN click_flag THEN 1 ELSE 0 END) AS click_flag,
    MAX(CASE WHEN unsubscribe_flag THEN 1 ELSE 0 END) AS unsubscribe_flag,
    MAX(CASE WHEN conversion_within_7d THEN 1 ELSE 0 END) AS conversion_within_7d,
    MAX(CASE WHEN conversion_within_30d THEN 1 ELSE 0 END) AS conversion_within_30d,
    MAX(revenue_within_7d) AS revenue_within_7d,
    MAX(revenue_within_30d) AS revenue_within_30d,
    COUNT(*) AS source_event_rows
FROM campaign_events
GROUP BY campaign_id, customer_id;

CREATE OR REPLACE VIEW stg_returns AS
SELECT
    return_id,
    order_item_id,
    return_request_date,
    return_processed_date,
    return_reason,
    refund_amount,
    refund_method,
    return_status,
    return_channel
FROM returns;

CREATE OR REPLACE VIEW stg_reviews AS
SELECT
    review_id,
    customer_id,
    product_id,
    review_date,
    rating,
    review_length,
    sentiment_seed,
    helpful_votes
FROM reviews;

CREATE OR REPLACE VIEW stg_web_sessions AS
SELECT
    session_id,
    customer_id_nullable,
    session_start,
    session_end,
    traffic_source,
    device_type,
    pages_viewed,
    product_views,
    category_views,
    add_to_cart_flag,
    checkout_flag,
    purchase_flag,
    bounce_flag
FROM web_sessions;

CREATE OR REPLACE VIEW stg_session_events AS
SELECT
    event_id,
    session_id,
    customer_id_nullable,
    event_time,
    event_type,
    product_id_nullable,
    category_nullable
FROM session_events;
