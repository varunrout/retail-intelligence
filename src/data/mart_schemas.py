from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MartContract:
    mart_name: str
    required_columns: list[str]
    key_columns: list[str]
    date_columns: list[str]
    non_negative_columns: list[str]
    zero_to_one_columns: list[str]


MART_CONTRACTS: dict[str, MartContract] = {
    "mart_customer_features": MartContract(
        mart_name="mart_customer_features",
        required_columns=[
            "customer_id",
            "region",
            "signup_date",
            "total_orders",
            "total_net_revenue",
            "recency_days",
            "churn_flag_90d",
        ],
        key_columns=["customer_id"],
        date_columns=["signup_date"],
        non_negative_columns=[
            "total_orders",
            "total_net_revenue",
            "avg_order_value",
            "recency_days",
            "tenure_days",
            "total_units",
            "total_returns",
            "total_sessions",
            "campaigns_targeted",
        ],
        zero_to_one_columns=["online_order_share", "store_order_share", "return_rate_per_unit"],
    ),
    "mart_product_demand": MartContract(
        mart_name="mart_product_demand",
        required_columns=[
            "week_start_date",
            "product_id",
            "store_id_or_online",
            "units_sold",
            "net_revenue",
            "weekly_demand_band",
        ],
        key_columns=["week_start_date", "product_id", "store_id_or_online"],
        date_columns=["week_start_date"],
        non_negative_columns=[
            "units_sold",
            "order_line_count",
            "net_revenue",
            "promo_days",
            "markdown_days",
            "stockout_days",
            "backorder_days",
            "rolling_4w_avg_units",
        ],
        zero_to_one_columns=["avg_item_discount_pct", "avg_discount_pct"],
    ),
    "mart_campaign_response": MartContract(
        mart_name="mart_campaign_response",
        required_columns=[
            "campaign_id",
            "customer_id",
            "assignment_datetime",
            "treatment_flag",
            "control_flag",
            "response_flag_30d",
        ],
        key_columns=["campaign_id", "customer_id"],
        date_columns=["assignment_datetime"],
        non_negative_columns=[
            "offer_strength",
            "revenue_within_7d",
            "revenue_within_30d",
            "source_event_rows",
            "pre_90d_orders",
            "pre_90d_revenue",
            "campaign_response_rank",
            "revenue_decile_within_campaign",
        ],
        zero_to_one_columns=[],
    ),
    "mart_returns_risk": MartContract(
        mart_name="mart_returns_risk",
        required_columns=[
            "order_item_id",
            "order_id",
            "customer_id",
            "product_id",
            "order_date",
            "return_flag",
            "return_risk_band",
        ],
        key_columns=["order_item_id"],
        date_columns=["order_date"],
        non_negative_columns=[
            "quantity",
            "item_list_price",
            "item_discount_pct",
            "item_net_price",
            "item_cost",
            "refund_amount",
            "prior_customer_return_rate",
            "recent_product_return_events",
            "customer_item_recency_rank",
        ],
        zero_to_one_columns=["item_discount_pct", "prior_customer_return_rate"],
    ),
    "mart_recommendation_interactions": MartContract(
        mart_name="mart_recommendation_interactions",
        required_columns=[
            "customer_id",
            "product_id",
            "interaction_score",
            "last_interaction_date",
            "product_rank_for_customer",
        ],
        key_columns=["customer_id", "product_id"],
        date_columns=["first_interaction_date", "last_interaction_date"],
        non_negative_columns=[
            "active_days",
            "total_interactions",
            "total_interaction_value",
            "interaction_score",
            "recency_days",
            "product_rank_for_customer",
            "category_rank_for_customer",
        ],
        zero_to_one_columns=[],
    ),
    "mart_store_week_performance": MartContract(
        mart_name="mart_store_week_performance",
        required_columns=[
            "week_start_date",
            "store_id_or_online",
            "region",
            "order_count",
            "net_revenue",
            "performance_band",
        ],
        key_columns=["week_start_date", "store_id_or_online"],
        date_columns=["week_start_date"],
        non_negative_columns=[
            "order_count",
            "net_revenue",
            "discount_amount",
            "campaign_attributed_orders",
            "units_sold",
            "stockout_days",
            "backorder_days",
            "running_revenue",
            "rolling_4w_avg_revenue",
        ],
        zero_to_one_columns=["avg_item_discount_pct", "avg_discount_pct"],
    ),
}


MART_FILE_NAMES: dict[str, str] = {name: f"{name}.csv" for name in MART_CONTRACTS}
