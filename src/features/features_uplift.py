"""Uplift V2 feature engineering.

Decisions encoded here are documented and justified in
``analysis_notebooks/uplift_analysis.ipynb``. Each feature set choice has a
finding number tying it back to the analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ── Label + keys ────────────────────────────────────────────────────────────
LABEL = "response_flag_30d"  # Finding 2.A: identical to conversion_within_30d
TREATMENT_KEY = "treatment_flag"
CAMPAIGN_KEY = "campaign_id"
TIME_KEY = "assignment_datetime"
ID_KEY = "customer_id"

# ── Hard-forbidden features (Finding 3.A — hard leakage) ───────────────────
# These features are either post-treatment outcomes or derived from the label.
# Do NOT include them under any circumstances.
FORBIDDEN_FEATURES: set[str] = {
    # Hard leakage: AUC >= 0.65 in single-feature test
    "conversion_within_30d",
    "conversion_within_7d",
    "revenue_within_30d",
    "days_to_first_order",  # not in mart but guard anyway
    # Post-treatment outcomes (Finding 3.B)
    "discount_amount",
    "delivery_fee_waived",
    "total_order_value",
    "revenue_within_7d",
    # Email engagement — measured after assignment, causally downstream
    "delivered_flag",
    "open_flag",
    "click_flag",
    "unsubscribe_flag",
    # Duplicate label column (Finding 2.A)
    "response_bucket",
    "campaign_response_rank",
    "revenue_decile_within_campaign",
    "source_event_rows",
}

# ── Baseline campaign-mart features (phase7 approved, minus forbidden) ──────
CAMPAIGN_MART_FEATURES = [
    "campaign_id",
    "campaign_type",
    "campaign_channel",
    "offer_type",
    "offer_strength",
    "predicted_business_segment_at_send",
    "targeting_rule_source",  # Finding 1.C: include as covariate, not filter
    "pre_90d_orders",
    "pre_90d_revenue",
    "pre_90d_aov",
    "assignment_month",  # engineered below
    "assignment_dayofweek",  # engineered below
]

# ── Customer-mart enrichment features (Finding 6.C) ─────────────────────────
# Joined from mart_customer_features on customer_id.
CUSTOMER_ENRICHMENT_FEATURES = [
    "tenure_days",
    "total_orders",
    "total_sessions",
    "avg_item_discount_pct",
    "avg_basket_size",
    "total_returns",
    "customer_value_band",
    "online_order_share",
    "income_band",
    "recency_days",
    "return_rate_per_unit",
    "campaigns_targeted",
]

# ── Engineered features ──────────────────────────────────────────────────────
ENGINEERED_FEATURES = [
    "segment_campaign",  # Finding 7.A: customer_value_band × campaign_type
    "is_new_customer",  # pre_90d_orders == 0
    "targeting_is_rule_based",  # binary flag from targeting_rule_source
    "log_pre_90d_revenue",  # log1p transform of skewed revenue
    "log_tenure_days",  # log1p
    "order_density_90d",  # pre_90d_orders / 90
]

CATEGORICAL_FEATURES = [
    "campaign_id",
    "campaign_type",
    "campaign_channel",
    "offer_type",
    "predicted_business_segment_at_send",
    "targeting_rule_source",
    "customer_value_band",
    "income_band",
    "segment_campaign",
]


@dataclass(frozen=True)
class FeatureSet:
    feature_columns: list[str]
    categorical_columns: list[str]


def feature_set() -> FeatureSet:
    """Final V2 feature set: campaign mart + customer enrichment + engineered."""
    cols = CAMPAIGN_MART_FEATURES + CUSTOMER_ENRICHMENT_FEATURES + ENGINEERED_FEATURES
    # deduplicate while preserving order
    seen: set[str] = set()
    unique_cols = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            unique_cols.append(c)

    return FeatureSet(
        feature_columns=unique_cols,
        categorical_columns=CATEGORICAL_FEATURES,
    )


def enrich_with_customer_mart(
    mart: pd.DataFrame,
    cust_mart: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join customer-mart features onto the campaign-response mart.

    Only columns in CUSTOMER_ENRICHMENT_FEATURES (plus customer_id as key)
    are carried across. Missing customers after join result in NaN rows,
    which LightGBM handles natively (Finding 6.C).
    """
    keep = [c for c in ["customer_id"] + CUSTOMER_ENRICHMENT_FEATURES if c in cust_mart.columns]
    slim = cust_mart[keep].drop_duplicates("customer_id")
    return mart.merge(slim, on="customer_id", how="left")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add assignment-date features and engineered columns in-place copy.

    Expects ``assignment_datetime`` already parsed as datetime.
    """
    out = df.copy()

    # ── Assignment-date features ────────────────────────────────────────────
    dt = pd.to_datetime(out[TIME_KEY])
    out["assignment_month"] = dt.dt.month.astype("int8")
    out["assignment_dayofweek"] = dt.dt.dayofweek.astype("int8")

    # ── Finding 7.A: segment_campaign interaction ───────────────────────────
    val_band = out["customer_value_band"].astype(str).fillna("unknown")
    ctype = out["campaign_type"].astype(str).fillna("unknown")
    out["segment_campaign"] = val_band + "_" + ctype

    # ── Finding 3.A guard: new customer flag ───────────────────────────────
    out["is_new_customer"] = (out["pre_90d_orders"].fillna(0) == 0).astype("int8")

    # ── Finding 1.C: binary rule-based flag ────────────────────────────────
    out["targeting_is_rule_based"] = (
        out["targeting_rule_source"].str.startswith("rule").fillna(False)
    ).astype("int8")

    # ── Log transforms for skewed numerics ─────────────────────────────────
    out["log_pre_90d_revenue"] = np.log1p(out["pre_90d_revenue"].fillna(0))
    out["log_tenure_days"] = np.log1p(out["tenure_days"].fillna(0))

    # ── Purchase density in pre-period ─────────────────────────────────────
    out["order_density_90d"] = out["pre_90d_orders"].fillna(0) / 90.0

    return out


def time_ordered_split(
    df: pd.DataFrame,
    *,
    test_size: float = 0.20,
    time_key: str = TIME_KEY,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort by ``time_key`` ascending, send latest ``test_size`` to test.

    Time-ordered split is consistent with the phase7 baseline split strategy
    so the two can be compared directly.
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
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return ``(X, y, treatment)`` with proper categorical dtypes for LightGBM."""
    fs = fs or feature_set()
    available = [c for c in fs.feature_columns if c in df.columns]
    X = df[available].copy()
    for col in fs.categorical_columns:
        if col in X.columns:
            X[col] = X[col].astype("category")
    y = df[LABEL].astype(int)
    t = df[TREATMENT_KEY].astype(int)
    return X, y, t
