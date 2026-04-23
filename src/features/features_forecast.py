"""Forecast V2 feature engineering.

Decisions encoded here are documented and justified in
``analysis_notebooks/forecast_analysis.ipynb``. Each feature set choice has a
finding number tying it back to the analysis.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ── Keys ───────────────────────────────────────────────────────────────────
LABEL = "units_sold"
TIME_KEY = "week_start_date"
SERIES_KEYS = ["product_id", "store_id_or_online"]

# ── Minimum series length for training (Finding 1.A) ──────────────────────
# 12.7% of series have <8 weeks — lag features are NaN/meaningless for them.
SERIES_MIN_WEEKS: int = 8

# ── Train/test split week — same 80/20 split as baseline (Finding 1.C) ────
SPLIT_WEEK: str = "2025-08-04"

# ── Forbidden columns — same-week outcomes / leakage (Finding 2.A) ────────
FORBIDDEN_FEATURES: set[str] = {
    "rolling_4w_avg_units",   # F2.A: mean incl. current week (100% match = leakage)
    "rolling_4w_revenue",     # F2.A: same leakage source
    "order_line_count",       # same-week order count
    "net_revenue",            # same-week revenue
    "avg_item_discount_pct",  # realised same-week discount rate
    "avg_discount_pct",       # realised same-week discount rate
    "avg_ending_inventory",   # end-of-week measurement
    "stockout_days",          # within-week measurement
    "backorder_days",         # within-week measurement
    "stock_received_units",   # within-week receipt
    "weekly_demand_band",     # derived from units_sold (target)
}

# ── Static product attributes (safe — no temporal leakage) ────────────────
STATIC_FEATURES: list[str] = [
    "category",       # F5.A: large inter-category sMAPE spread
    "subcategory",
    "price_tier",
    "seasonal_flag",
    "premium_flag",
]

# ── Promotional calendar — planned before week starts, safe ───────────────
PROMO_FEATURES: list[str] = [
    "promo_days",
    "markdown_days",
    "avg_listed_price",
]

# ── Inventory — partial coverage (25% null); LightGBM handles natively ────
# F4.C: avg_starting_inventory r=0.168; add binary null flag.
INVENTORY_FEATURES: list[str] = [
    "avg_starting_inventory",
    "has_inventory",       # engineered: 1 if inventory data present, else 0
]

# ── Lag features — mart-native + engineered additional lags (F4.A / F6.C) ─
LAG_FEATURES: list[str] = [
    "lag_1w_units_sold",   # already in mart
    "lag_2w",              # engineered
    "lag_3w",              # engineered
    "lag_4w_units_sold",   # already in mart
    "lag_8w",              # engineered
    "lag_13w",             # engineered
    "lag_52w",             # engineered — F6.C: ~50% coverage (YoY anchor)
    "has_yoy_lag",         # engineered: 1 if lag_52w is non-null, else 0
]

# ── Lagged rolling means — backward-looking, highest-signal features (F6.B) ─
# roll_Xw_avg = mean of lag_1 through lag_X (no current-week data).
ROLLING_FEATURES: list[str] = [
    "roll_2w_avg",    # mean(lag_1, lag_2)
    "roll_8w_avg",    # mean(lag_1 … lag_8)
    "roll_13w_avg",   # mean(lag_1 … lag_13)
]

# ── Series-level baseline demand — computed from train set only (F6.A) ────
SERIES_MEAN_FEATURES: list[str] = [
    "product_mean_demand",   # mean units_sold per product_id on train
    "store_mean_demand",     # mean units_sold per store on train
]

# ── Calendar cyclical encoding (F4.B) ─────────────────────────────────────
# Avoids week-52/1 edge discontinuity from ordinal week_of_year.
CALENDAR_FEATURES: list[str] = [
    "sin_woy",
    "cos_woy",
]

# ── Categorical columns for LightGBM native handling ─────────────────────
CATEGORICAL_FEATURES: list[str] = [
    "category",
    "subcategory",
    "price_tier",
]


@dataclass(frozen=True)
class FeatureSet:
    feature_columns: list[str]
    categorical_columns: list[str]


def feature_set() -> FeatureSet:
    """Return the full V2 feature set (deduplicated, order-preserved)."""
    all_cols = (
        STATIC_FEATURES
        + PROMO_FEATURES
        + INVENTORY_FEATURES
        + LAG_FEATURES
        + ROLLING_FEATURES
        + SERIES_MEAN_FEATURES
        + CALENDAR_FEATURES
    )
    seen: set[str] = set()
    unique_cols: list[str] = []
    for c in all_cols:
        if c not in seen:
            seen.add(c)
            unique_cols.append(c)
    return FeatureSet(
        feature_columns=unique_cols,
        categorical_columns=CATEGORICAL_FEATURES,
    )


def build_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute additional lag and rolling features on the full mart.

    Must be called on the **full (unsplit) mart** so lag windows span the
    train/test boundary correctly. All engineered features are strictly
    backward-looking — no leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Full mart DataFrame containing ``units_sold``, ``week_start_date``,
        ``product_id``, ``store_id_or_online``, and inventory columns.

    Returns
    -------
    pd.DataFrame
        Sorted by ``[product_id, store_id_or_online, week_start_date]`` with
        all engineered columns appended.
    """
    if not pd.api.types.is_datetime64_any_dtype(df[TIME_KEY]):
        df = df.copy()
        df[TIME_KEY] = pd.to_datetime(df[TIME_KEY])

    out = df.sort_values([*SERIES_KEYS, TIME_KEY], kind="mergesort").copy()

    grp = out.groupby(SERIES_KEYS, sort=False)["units_sold"]

    # ── Additional lags (F4.A / F6.C) ─────────────────────────────────────
    out["lag_2w"] = grp.shift(2)
    out["lag_3w"] = grp.shift(3)
    out["lag_8w"] = grp.shift(8)
    out["lag_13w"] = grp.shift(13)
    out["lag_52w"] = grp.shift(52)
    out["has_yoy_lag"] = out["lag_52w"].notna().astype("int8")

    # ── Lagged rolling means (F6.B) ───────────────────────────────────────
    # shift(1) ensures the window starts at lag_1 (no current-week data).
    for window, col, min_p in [
        (2,  "roll_2w_avg",  1),
        (8,  "roll_8w_avg",  4),
        (13, "roll_13w_avg", 6),
    ]:
        out[col] = out.groupby(SERIES_KEYS, sort=False)["units_sold"].transform(
            lambda s, w=window, mp=min_p: s.shift(1).rolling(w, min_periods=mp).mean()
        )

    # ── Inventory null flag (F4.C) ─────────────────────────────────────────
    out["has_inventory"] = out["avg_starting_inventory"].notna().astype("int8")

    # ── Calendar cyclical encoding (F4.B) ──────────────────────────────────
    woy = out[TIME_KEY].dt.isocalendar().week.astype(float)
    out["sin_woy"] = np.sin(2 * np.pi * woy / 52.0)
    out["cos_woy"] = np.cos(2 * np.pi * woy / 52.0)

    return out


def compute_series_means(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute product-level and store-level mean demand from the train set.

    Must be computed from train rows only to prevent future-data leakage
    (Finding 6.A). The resulting DataFrames are then joined onto both
    train and test via ``attach_series_means``.

    Returns
    -------
    product_means : pd.DataFrame
        Columns: ``product_id``, ``product_mean_demand``.
    store_means : pd.DataFrame
        Columns: ``store_id_or_online``, ``store_mean_demand``.
    """
    product_means = (
        train_df.groupby("product_id")[LABEL]
        .mean()
        .rename("product_mean_demand")
        .reset_index()
    )
    store_means = (
        train_df.groupby("store_id_or_online")[LABEL]
        .mean()
        .rename("store_mean_demand")
        .reset_index()
    )
    return product_means, store_means


def attach_series_means(
    df: pd.DataFrame,
    product_means: pd.DataFrame,
    store_means: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join product_mean_demand and store_mean_demand features onto df."""
    out = df.merge(product_means, on="product_id", how="left")
    out = out.merge(store_means, on="store_id_or_online", how="left")
    return out


def filter_short_series(df: pd.DataFrame, min_weeks: int = SERIES_MIN_WEEKS) -> pd.DataFrame:
    """Drop training rows for series with fewer than ``min_weeks`` observations.

    Applied to the training set only (Finding 1.A: 12.7% of series have <8 weeks;
    their lag features are NaN and add noise without predictive signal).
    Not applied to the test set — short-series predictions are still produced.
    """
    lengths = df.groupby(SERIES_KEYS)[TIME_KEY].transform("count")
    return df[lengths >= min_weeks].copy()


def week_time_split(
    df: pd.DataFrame,
    split_week: str = SPLIT_WEEK,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split mart on a fixed week boundary (80/20 week-ordered split).

    Returns ``(train_df, test_df)`` where train contains weeks strictly before
    ``split_week`` and test contains weeks >= ``split_week``.
    """
    if not pd.api.types.is_datetime64_any_dtype(df[TIME_KEY]):
        df = df.copy()
        df[TIME_KEY] = pd.to_datetime(df[TIME_KEY])
    cut = pd.Timestamp(split_week)
    return df[df[TIME_KEY] < cut].copy(), df[df[TIME_KEY] >= cut].copy()


def prepare_xy(
    df: pd.DataFrame,
    fs: FeatureSet | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return ``(X, y)`` for model training / evaluation.

    Only columns present in ``df`` are included (missing engineered features
    are silently dropped — LightGBM will train on whatever is available).
    """
    if fs is None:
        fs = feature_set()
    avail = [c for c in fs.feature_columns if c in df.columns]
    X = df[avail].copy()
    y = df[LABEL].values.astype(float)
    return X, y
