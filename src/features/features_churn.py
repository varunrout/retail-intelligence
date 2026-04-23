"""Churn V2 feature engineering.

Decisions encoded here are documented and justified in
``analysis_notebooks/churn_analysis.ipynb``. Each engineered feature has a
finding number tying it back to the analysis.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ── Approved baseline columns to carry forward ─────────────────────────────
# Source: outputs/phase6_churn_approved_features.csv minus the 5 weak
# categoricals dropped per Finding 2 (each with <2pp churn-rate spread).
_DROPPED_WEAK_CATEGORICALS = [
    "region",
    "city_tier",
    "preferred_channel",
    "loyalty_tier",
    "is_marketing_opt_in",
]

CARRY_FORWARD_FEATURES = [
    # demographics / segmentation that DO carry signal
    "income_band",
    "customer_segment_seed",
    # volume / revenue
    "total_orders",
    "total_net_revenue",
    "avg_order_value",
    "total_discount_amount",
    "avg_basket_size",
    "tenure_days",
    "revenue_per_order",
    "online_order_share",
    "store_order_share",
    "total_units",
    "avg_item_discount_pct",
    "avg_item_margin",
    # returns
    "total_returns",
    "total_refund_amount",
    "avg_days_to_return",
    "return_rate_per_unit",
    # web sessions
    "total_sessions",
    "avg_session_minutes",
    "avg_pages_viewed",
    "sessions_add_to_cart",
    "sessions_with_purchase",
    # campaigns
    "campaigns_targeted",
    "campaigns_treatment",
    "campaigns_converted_30d",
    "campaign_revenue_30d",
    # value bands
    "customer_value_band",
    "spend_rank_in_region",
]

ENGINEERED_FEATURES = [
    "order_velocity_per_month",
    "purchase_session_rate",
    "cart_to_purchase_rate",
]

NULL_FLAG_FEATURES = [
    "is_loyalty_enrolled",
    "was_targeted_by_campaign",
    "has_session_data",
]

CATEGORICAL_FEATURES = [
    "income_band",
    "customer_segment_seed",
    "customer_value_band",
]

LABEL = "churn_flag_90d"
TIME_KEY = "signup_date"
ID_KEY = "customer_id"


@dataclass(frozen=True)
class FeatureSet:
    feature_columns: list[str]
    categorical_columns: list[str]


def feature_set() -> FeatureSet:
    """Final V2 feature set: 29 carry-forward + 3 engineered + 3 null flags."""
    return FeatureSet(
        feature_columns=CARRY_FORWARD_FEATURES + ENGINEERED_FEATURES + NULL_FLAG_FEATURES,
        categorical_columns=CATEGORICAL_FEATURES,
    )


def filter_active_population(mart: pd.DataFrame) -> pd.DataFrame:
    """Drop the structurally-zero ``no_purchase`` segment (Finding 1).

    These customers never purchased, so ``churn_flag_90d`` is mechanically 0
    for all of them. Including them in training tells the model that
    "all-feature-nulls => non-churn", which is true only by label
    construction, not by any retention dynamic.
    """
    return mart.loc[mart["customer_value_band"] != "no_purchase"].copy()


def filter_mature_cohort(
    mart: pd.DataFrame,
    *,
    maturity_days: int = 90,
    time_key: str = TIME_KEY,
) -> pd.DataFrame:
    """Drop customers who signed up within ``maturity_days`` of the latest
    snapshot date (Finding 7).

    The 90-day churn label is undefined for customers who have not yet had
    90 days to be inactive. Random splits hide this; time-ordered splits
    expose it.
    """
    if time_key not in mart.columns:
        raise KeyError(f"time_key column missing: {time_key}")
    dates = pd.to_datetime(mart[time_key])
    cutoff = dates.max() - pd.Timedelta(days=maturity_days)
    return mart.loc[dates <= cutoff].copy()


def build_features(mart: pd.DataFrame) -> pd.DataFrame:
    """Add engineered + null-flag features to the mart frame.

    Does not drop columns. Caller is responsible for selecting the final
    feature subset via :func:`feature_set`.
    """
    df = mart.copy()

    # ── Engineered rates (Finding 6) ───────────────────────────────────────
    # order_velocity uses orders / tenure -- no recency dependence (audit §3)
    tenure_months = df["tenure_days"] / 30.0
    df["order_velocity_per_month"] = np.where(
        df["tenure_days"] > 0,
        df["total_orders"] / tenure_months,
        np.nan,
    )

    # purchase_session_rate: conversion intent
    df["purchase_session_rate"] = np.where(
        df["total_sessions"] > 0,
        df["sessions_with_purchase"] / df["total_sessions"],
        np.nan,
    )

    # cart_to_purchase_rate: funnel completion
    df["cart_to_purchase_rate"] = np.where(
        df["sessions_add_to_cart"] > 0,
        df["sessions_with_purchase"] / df["sessions_add_to_cart"],
        np.nan,
    )

    # ── Null-flag features (Finding 4) ─────────────────────────────────────
    df["is_loyalty_enrolled"] = df["loyalty_tier"].notna().astype("int8")
    df["was_targeted_by_campaign"] = df["campaigns_targeted"].notna().astype("int8")
    df["has_session_data"] = df["total_sessions"].notna().astype("int8")

    return df


def time_ordered_split(
    df: pd.DataFrame,
    *,
    test_size: float = 0.20,
    time_key: str = TIME_KEY,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort by ``time_key`` ascending, send latest ``test_size`` to test.

    Assumes ``filter_mature_cohort`` has already been applied.
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    sorted_df = df.sort_values(time_key, kind="mergesort").reset_index(drop=True)
    split_idx = int(len(sorted_df) * (1 - test_size))
    train = sorted_df.iloc[:split_idx].copy()
    test = sorted_df.iloc[split_idx:].copy()
    return train, test


def prepare_xy(
    df: pd.DataFrame,
    *,
    fs: FeatureSet | None = None,
    label: str = LABEL,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` with the V2 feature subset and proper categorical
    dtypes for LightGBM native handling."""
    fs = fs or feature_set()
    X = df[fs.feature_columns].copy()
    for col in fs.categorical_columns:
        if col in X.columns:
            X[col] = X[col].astype("category")
    y = df[label].astype("int8")
    return X, y
