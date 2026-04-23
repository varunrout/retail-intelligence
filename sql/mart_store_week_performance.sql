CREATE OR REPLACE TABLE mart_store_week_performance AS
WITH order_weekly AS (
    SELECT
        DATE_TRUNC('week', o.order_date) AS week_start_date,
        CASE
            WHEN LOWER(o.channel) = 'store' THEN o.store_id_nullable
            ELSE 'ONLINE'
        END AS store_id_or_online,
        COUNT(DISTINCT o.order_id) AS order_count,
        SUM(o.net_amount) AS net_revenue,
        SUM(o.discount_amount) AS discount_amount,
        AVG(o.basket_size) AS avg_basket_size,
        SUM(CASE WHEN o.campaign_attributed_flag THEN 1 ELSE 0 END) AS campaign_attributed_orders
    FROM stg_completed_orders o
    GROUP BY DATE_TRUNC('week', o.order_date),
             CASE WHEN LOWER(o.channel) = 'store' THEN o.store_id_nullable ELSE 'ONLINE' END
),
item_weekly AS (
    SELECT
        DATE_TRUNC('week', o.order_date) AS week_start_date,
        CASE
            WHEN LOWER(o.channel) = 'store' THEN o.store_id_nullable
            ELSE 'ONLINE'
        END AS store_id_or_online,
        SUM(oi.quantity) AS units_sold,
        SUM(oi.item_margin) AS item_margin,
        AVG(oi.item_discount_pct) AS avg_item_discount_pct
    FROM stg_completed_orders o
    JOIN stg_order_items oi
        ON o.order_id = oi.order_id
    GROUP BY DATE_TRUNC('week', o.order_date),
             CASE WHEN LOWER(o.channel) = 'store' THEN o.store_id_nullable ELSE 'ONLINE' END
),
inventory_weekly AS (
    SELECT
        DATE_TRUNC('week', date) AS week_start_date,
        store_id_or_online,
        AVG(starting_inventory) AS avg_starting_inventory,
        AVG(ending_inventory) AS avg_ending_inventory,
        SUM(stock_received) AS stock_received_units,
        SUM(CASE WHEN stockout_flag THEN 1 ELSE 0 END) AS stockout_days,
        SUM(CASE WHEN backorder_flag THEN 1 ELSE 0 END) AS backorder_days
    FROM stg_daily_inventory
    GROUP BY DATE_TRUNC('week', date), store_id_or_online
),
pricing_weekly AS (
    SELECT
        DATE_TRUNC('week', date) AS week_start_date,
        store_id_or_online,
        AVG(discount_pct) AS avg_discount_pct,
        SUM(CASE WHEN promo_flag THEN 1 ELSE 0 END) AS promo_days,
        SUM(CASE WHEN markdown_flag THEN 1 ELSE 0 END) AS markdown_days
    FROM stg_daily_prices
    GROUP BY DATE_TRUNC('week', date), store_id_or_online
),
assembled AS (
    SELECT
        ow.week_start_date,
        ow.store_id_or_online,
        s.region,
        s.city,
        s.store_type,
        s.store_size_band,
        ow.order_count,
        ow.net_revenue,
        ow.discount_amount,
        ow.avg_basket_size,
        ow.campaign_attributed_orders,
        iw.units_sold,
        iw.item_margin,
        iw.avg_item_discount_pct,
        inv.avg_starting_inventory,
        inv.avg_ending_inventory,
        inv.stock_received_units,
        inv.stockout_days,
        inv.backorder_days,
        pr.avg_discount_pct,
        pr.promo_days,
        pr.markdown_days
    FROM order_weekly ow
    LEFT JOIN item_weekly iw
        ON ow.week_start_date = iw.week_start_date
       AND ow.store_id_or_online = iw.store_id_or_online
    LEFT JOIN inventory_weekly inv
        ON ow.week_start_date = inv.week_start_date
       AND ow.store_id_or_online = inv.store_id_or_online
    LEFT JOIN pricing_weekly pr
        ON ow.week_start_date = pr.week_start_date
       AND ow.store_id_or_online = pr.store_id_or_online
    LEFT JOIN stg_stores s
        ON ow.store_id_or_online = s.store_id
)
SELECT
    week_start_date,
    store_id_or_online,
    COALESCE(region, 'online') AS region,
    city,
    store_type,
    store_size_band,
    order_count,
    net_revenue,
    discount_amount,
    avg_basket_size,
    campaign_attributed_orders,
    units_sold,
    item_margin,
    avg_item_discount_pct,
    avg_starting_inventory,
    avg_ending_inventory,
    stock_received_units,
    stockout_days,
    backorder_days,
    avg_discount_pct,
    promo_days,
    markdown_days,
    SUM(net_revenue) OVER (
        PARTITION BY store_id_or_online
        ORDER BY week_start_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS running_revenue,
    AVG(net_revenue) OVER (
        PARTITION BY store_id_or_online
        ORDER BY week_start_date
        ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
    ) AS rolling_4w_avg_revenue,
    LAG(net_revenue, 1) OVER (
        PARTITION BY store_id_or_online
        ORDER BY week_start_date
    ) AS lag_1w_revenue,
    CASE
        WHEN net_revenue >= 20000 THEN 'high'
        WHEN net_revenue >= 8000 THEN 'medium'
        ELSE 'low'
    END AS performance_band
FROM assembled;
