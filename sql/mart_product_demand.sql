CREATE OR REPLACE TABLE mart_product_demand AS
WITH order_item_enriched AS (
    SELECT
        o.order_date,
        DATE_TRUNC('week', o.order_date) AS week_start_date,
        CASE
            WHEN LOWER(o.channel) = 'store' THEN o.store_id_nullable
            ELSE 'ONLINE'
        END AS store_id_or_online,
        oi.product_id,
        oi.quantity,
        oi.item_net_price,
        oi.item_discount_pct
    FROM stg_completed_orders o
    JOIN stg_order_items oi
        ON o.order_id = oi.order_id
),
demand_weekly AS (
    SELECT
        week_start_date,
        product_id,
        store_id_or_online,
        SUM(quantity) AS units_sold,
        COUNT(*) AS order_line_count,
        SUM(item_net_price) AS net_revenue,
        AVG(item_discount_pct) AS avg_item_discount_pct
    FROM order_item_enriched
    GROUP BY week_start_date, product_id, store_id_or_online
),
price_weekly AS (
    SELECT
        DATE_TRUNC('week', date) AS week_start_date,
        product_id,
        store_id_or_online,
        AVG(listed_price) AS avg_listed_price,
        AVG(discount_pct) AS avg_discount_pct,
        SUM(CASE WHEN promo_flag THEN 1 ELSE 0 END) AS promo_days,
        SUM(CASE WHEN markdown_flag THEN 1 ELSE 0 END) AS markdown_days
    FROM stg_daily_prices
    GROUP BY DATE_TRUNC('week', date), product_id, store_id_or_online
),
inventory_weekly AS (
    SELECT
        DATE_TRUNC('week', date) AS week_start_date,
        product_id,
        store_id_or_online,
        AVG(starting_inventory) AS avg_starting_inventory,
        AVG(ending_inventory) AS avg_ending_inventory,
        SUM(stock_received) AS stock_received_units,
        SUM(CASE WHEN stockout_flag THEN 1 ELSE 0 END) AS stockout_days,
        SUM(CASE WHEN backorder_flag THEN 1 ELSE 0 END) AS backorder_days
    FROM stg_daily_inventory
    GROUP BY DATE_TRUNC('week', date), product_id, store_id_or_online
),
assembled AS (
    SELECT
        dw.week_start_date,
        dw.product_id,
        dw.store_id_or_online,
        p.category,
        p.subcategory,
        p.price_tier,
        p.seasonal_flag,
        p.premium_flag,
        dw.units_sold,
        dw.order_line_count,
        dw.net_revenue,
        dw.avg_item_discount_pct,
        pw.avg_listed_price,
        pw.avg_discount_pct,
        pw.promo_days,
        pw.markdown_days,
        iw.avg_starting_inventory,
        iw.avg_ending_inventory,
        iw.stock_received_units,
        iw.stockout_days,
        iw.backorder_days
    FROM demand_weekly dw
    LEFT JOIN price_weekly pw
        ON dw.week_start_date = pw.week_start_date
       AND dw.product_id = pw.product_id
       AND dw.store_id_or_online = pw.store_id_or_online
    LEFT JOIN inventory_weekly iw
        ON dw.week_start_date = iw.week_start_date
       AND dw.product_id = iw.product_id
       AND dw.store_id_or_online = iw.store_id_or_online
    LEFT JOIN stg_products p
        ON dw.product_id = p.product_id
)
SELECT
    week_start_date,
    product_id,
    store_id_or_online,
    category,
    subcategory,
    price_tier,
    seasonal_flag,
    premium_flag,
    units_sold,
    order_line_count,
    net_revenue,
    avg_item_discount_pct,
    avg_listed_price,
    avg_discount_pct,
    promo_days,
    markdown_days,
    avg_starting_inventory,
    avg_ending_inventory,
    stock_received_units,
    stockout_days,
    backorder_days,
    LAG(units_sold, 1) OVER (
        PARTITION BY product_id, store_id_or_online
        ORDER BY week_start_date
    ) AS lag_1w_units_sold,
    LAG(units_sold, 4) OVER (
        PARTITION BY product_id, store_id_or_online
        ORDER BY week_start_date
    ) AS lag_4w_units_sold,
    AVG(units_sold) OVER (
        PARTITION BY product_id, store_id_or_online
        ORDER BY week_start_date
        ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
    ) AS rolling_4w_avg_units,
    SUM(net_revenue) OVER (
        PARTITION BY product_id, store_id_or_online
        ORDER BY week_start_date
        ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
    ) AS rolling_4w_revenue,
    CASE
        WHEN units_sold >= 30 THEN 'high'
        WHEN units_sold >= 10 THEN 'medium'
        ELSE 'low'
    END AS weekly_demand_band
FROM assembled;
