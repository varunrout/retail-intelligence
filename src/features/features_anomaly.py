"""Anomaly Detection V2 feature engineering.

Decisions encoded here are documented and justified in
``analysis_notebooks/anomaly_analysis.ipynb``. Each choice has a
finding number tying it back to the analysis.

V2 improvements over phase11:
  Finding 1.C  Supervised training — returns_hidden_labels.csv provides 1,352
               abuse labels (0.72% prevalence). Phase11 used unsupervised-only
               despite labels being available.
  Finding 2.A  Top numeric signals — item_net_price (AUROC 0.925),
               refund_amount (0.924), item_margin (0.913),
               prior_customer_return_rate (0.874).
  Finding 2.B  Category & discount signal — electronics = 75.5% of abuse
               returns vs 19% non-abuse; low_discount = 59.6% of abuse.
  Finding 2.C  Return reason — suspected_abuse reason carries 5.7% abuse rate
               vs 0.5% baseline; encoded as binary OHE flag.
  Finding 1.B  loyalty_tier 29.5% null — imputed to "unknown" for OHE to
               avoid dropping rows; phase11 had no null handling.
  Finding 4.C  scale_pos_weight — 0.72% prevalence requires ~125 weight.
               Computed from training split to avoid test leakage.
  Finding 5.C  Time-ordered split — prior_customer_return_rate encodes
               historical return behaviour; temporal ordering is mandatory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ABUSE_LABEL = "is_abuse"
TIME_KEY = "order_date"

# Numeric features (all already in mart_returns_risk)
NUMERIC_COLS: list[str] = [
    "item_net_price",
    "refund_amount",
    "prior_customer_return_rate",
    "item_discount_pct",
    "customer_item_recency_rank",
    "item_margin",
    "recent_product_return_events",
    "days_to_return",
]

# Binary OHE flags derived from categoricals (Finding 2.B, 2.C)
OHE_FLAG_COLS: list[str] = [
    "is_suspected_abuse_reason",  # return_reason == "suspected_abuse"
    "is_electronics",  # category == "electronics"
    "is_low_discount",  # discount_band == "low_discount"
    "is_high_risk_band",  # return_risk_band == "high"
]

FEATURE_COLS: list[str] = NUMERIC_COLS + OHE_FLAG_COLS


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def engineer_features(ret_df: pd.DataFrame) -> pd.DataFrame:
    """Add OHE binary flag columns to a returns-only DataFrame.

    Parameters
    ----------
    ret_df:
        DataFrame of return rows.  Must contain columns:
        ``return_reason``, ``category``, ``discount_band``,
        ``return_risk_band``, ``loyalty_tier``, and all ``NUMERIC_COLS``.

    Returns
    -------
    DataFrame with all ``FEATURE_COLS`` present (NaN-filled to 0 in numeric
    cols; loyalty_tier nulls filled to "unknown" for downstream grouping).
    """
    fe = ret_df.copy()

    # OHE flags (Finding 2.B, 2.C)
    fe["is_suspected_abuse_reason"] = (fe["return_reason"].fillna("") == "suspected_abuse").astype(
        int
    )
    fe["is_electronics"] = (fe["category"].fillna("") == "electronics").astype(int)
    fe["is_low_discount"] = (fe["discount_band"].fillna("") == "low_discount").astype(int)
    fe["is_high_risk_band"] = (fe["return_risk_band"].fillna("") == "high").astype(int)

    # Impute loyalty_tier nulls to "unknown" (Finding 1.B — 29.5% null)
    fe["loyalty_tier"] = fe["loyalty_tier"].fillna("unknown")

    # Fill remaining numeric nulls with 0
    fe[NUMERIC_COLS] = fe[NUMERIC_COLS].fillna(0)

    return fe


# ---------------------------------------------------------------------------
# FeatureSet dataclass
# ---------------------------------------------------------------------------


@dataclass
class FeatureSet:
    """Holds train/test splits and metadata for the anomaly model."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    split_date: str  # ISO date string at 80% temporal boundary
    scale_pos_weight: float  # neg/pos ratio from training split
    train_df: pd.DataFrame  # full training rows (for profiling)
    test_df: pd.DataFrame  # full test rows (for profiling)

    @property
    def n_train(self) -> int:
        return len(self.X_train)

    @property
    def n_test(self) -> int:
        return len(self.X_test)

    @property
    def train_prevalence(self) -> float:
        return float(self.y_train.mean())

    @property
    def test_prevalence(self) -> float:
        return float(self.y_test.mean())


def build_feature_set(ret_labeled: pd.DataFrame) -> FeatureSet:
    """Engineer features and perform time-ordered 80/20 split.

    Parameters
    ----------
    ret_labeled:
        Returns DataFrame with ``is_abuse`` label column already merged.
        Must be complete (no prior filtering).

    Returns
    -------
    ``FeatureSet`` with train/test arrays, metadata, and full row DataFrames.
    """
    fe = engineer_features(ret_labeled)
    fe = fe.sort_values(TIME_KEY).reset_index(drop=True)

    split_idx = int(len(fe) * 0.80)
    train_df = fe.iloc[:split_idx].copy()
    test_df = fe.iloc[split_idx:].copy()
    split_date = str(fe[TIME_KEY].iloc[split_idx].date())

    X_train = train_df[FEATURE_COLS].values.astype(np.float32)
    y_train = train_df[ABUSE_LABEL].values.astype(np.int32)
    X_test = test_df[FEATURE_COLS].values.astype(np.float32)
    y_test = test_df[ABUSE_LABEL].values.astype(np.int32)

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    return FeatureSet(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=list(FEATURE_COLS),
        split_date=split_date,
        scale_pos_weight=scale_pos_weight,
        train_df=train_df,
        test_df=test_df,
    )
