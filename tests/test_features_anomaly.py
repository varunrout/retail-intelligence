"""Unit tests for src/features/features_anomaly.py — data-independent, in-memory frames."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import features_anomaly as fa


def _returns_frame() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=20, freq="D")
    return pd.DataFrame(
        {
            "order_date": dates,
            "return_reason": ["suspected_abuse"] + ["changed_mind"] * 19,
            "category": ["electronics"] * 5 + ["fashion"] * 15,
            "discount_band": ["low_discount"] * 3 + ["high_discount"] * 17,
            "return_risk_band": ["high"] * 2 + ["low"] * 18,
            "loyalty_tier": [None] * 6 + ["gold"] * 14,
            "item_net_price": np.arange(20, dtype=float),
            "refund_amount": np.arange(20, dtype=float),
            "prior_customer_return_rate": np.linspace(0, 1, 20),
            "item_discount_pct": np.linspace(0, 0.5, 20),
            "customer_item_recency_rank": range(20),
            "item_margin": np.linspace(0.1, 0.4, 20),
            "recent_product_return_events": [0] * 20,
            "days_to_return": [5] * 20,
            "is_abuse": [1] + [0] * 19,
        }
    )


def test_engineer_features_ohe_flags():
    out = fa.engineer_features(_returns_frame())
    assert out["is_suspected_abuse_reason"].tolist()[:2] == [1, 0]
    assert out["is_electronics"].sum() == 5
    assert out["is_low_discount"].sum() == 3
    assert out["is_high_risk_band"].sum() == 2


def test_engineer_features_imputes_loyalty_tier_and_numeric_nulls():
    df = _returns_frame()
    df.loc[0, "item_net_price"] = np.nan
    out = fa.engineer_features(df)
    assert (out["loyalty_tier"].iloc[:6] == "unknown").all()
    assert out["item_net_price"].iloc[0] == 0


def test_build_feature_set_splits_time_ordered_80_20():
    fset = fa.build_feature_set(_returns_frame())
    assert fset.n_train == 16
    assert fset.n_test == 4
    assert fset.X_train.shape == (16, len(fa.FEATURE_COLS))
    assert fset.X_test.shape == (4, len(fa.FEATURE_COLS))


def test_build_feature_set_scale_pos_weight_matches_train_class_ratio():
    fset = fa.build_feature_set(_returns_frame())
    n_pos = int(fset.y_train.sum())
    n_neg = fset.n_train - n_pos
    assert fset.scale_pos_weight == pytest.approx(n_neg / max(n_pos, 1))


def test_build_feature_set_prevalence_properties():
    fset = fa.build_feature_set(_returns_frame())
    assert fset.train_prevalence == pytest.approx(fset.y_train.mean())
    assert fset.test_prevalence == pytest.approx(fset.y_test.mean())
