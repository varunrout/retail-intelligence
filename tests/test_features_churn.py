"""Unit tests for src/features/features_churn.py — data-independent, in-memory frames."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import features_churn as fc


def _mart_row(**overrides) -> dict:
    row = {
        "customer_id": "C1",
        "customer_value_band": "mid",
        "signup_date": "2024-01-01",
        "tenure_days": 300,
        "total_orders": 6,
        "total_sessions": 10,
        "sessions_with_purchase": 4,
        "sessions_add_to_cart": 5,
        "loyalty_tier": "gold",
        "campaigns_targeted": 2,
        "churn_flag_90d": 0,
        "income_band": "mid",
    }
    row.update(overrides)
    return row


def test_filter_active_population_drops_no_purchase_band():
    mart = pd.DataFrame([_mart_row(customer_value_band="no_purchase"), _mart_row()])
    out = fc.filter_active_population(mart)
    assert len(out) == 1
    assert (out["customer_value_band"] != "no_purchase").all()


def test_filter_mature_cohort_drops_recent_signups():
    mart = pd.DataFrame(
        [
            _mart_row(customer_id="C1", signup_date="2024-01-01"),  # mature
            _mart_row(customer_id="C2", signup_date="2024-09-20"),  # too recent
        ]
    )
    # latest signup is 2024-09-20; maturity_days=90 cutoff is 2024-06-22
    out = fc.filter_mature_cohort(mart, maturity_days=90)
    assert list(out["customer_id"]) == ["C1"]


def test_filter_mature_cohort_missing_time_key_raises():
    mart = pd.DataFrame([_mart_row()]).drop(columns=["signup_date"])
    with pytest.raises(KeyError):
        fc.filter_mature_cohort(mart)


def test_build_features_order_velocity_and_rates():
    mart = pd.DataFrame(
        [
            _mart_row(
                customer_id="C1",
                tenure_days=60,  # 2 months
                total_orders=4,  # 2/month
                total_sessions=10,
                sessions_with_purchase=5,  # rate 0.5
                sessions_add_to_cart=8,  # cart->purchase 5/8
            ),
            _mart_row(
                customer_id="C2",
                tenure_days=0,  # guards against div by zero
                total_orders=0,
                total_sessions=0,
                sessions_with_purchase=0,
                sessions_add_to_cart=0,
                loyalty_tier=None,
                campaigns_targeted=None,
            ),
        ]
    )
    out = fc.build_features(mart)

    c1 = out.loc[out["customer_id"] == "C1"].iloc[0]
    assert c1["order_velocity_per_month"] == pytest.approx(2.0)
    assert c1["purchase_session_rate"] == pytest.approx(0.5)
    assert c1["cart_to_purchase_rate"] == pytest.approx(5 / 8)
    assert c1["is_loyalty_enrolled"] == 1
    assert c1["was_targeted_by_campaign"] == 1

    c2 = out.loc[out["customer_id"] == "C2"].iloc[0]
    assert np.isnan(c2["order_velocity_per_month"])  # tenure_days == 0 guard
    assert np.isnan(c2["purchase_session_rate"])  # total_sessions == 0 guard
    assert np.isnan(c2["cart_to_purchase_rate"])  # sessions_add_to_cart == 0 guard
    assert c2["is_loyalty_enrolled"] == 0
    assert c2["was_targeted_by_campaign"] == 0
    assert c2["has_session_data"] == 1  # total_sessions == 0 is not null


def test_time_ordered_split_is_time_ordered_and_sized():
    mart = pd.DataFrame(
        [_mart_row(customer_id=f"C{i}", signup_date=f"2024-01-{i + 1:02d}") for i in range(10)]
    )
    train, test = fc.time_ordered_split(mart, test_size=0.3)
    assert len(train) == 7
    assert len(test) == 3
    assert train["signup_date"].max() <= test["signup_date"].min()


@pytest.mark.parametrize("bad_size", [0, 1, -0.1, 1.5])
def test_time_ordered_split_rejects_invalid_test_size(bad_size):
    mart = pd.DataFrame([_mart_row()])
    with pytest.raises(ValueError):
        fc.time_ordered_split(mart, test_size=bad_size)


def test_prepare_xy_returns_feature_subset_with_categorical_dtype():
    mart = fc.build_features(pd.DataFrame([_mart_row(), _mart_row(customer_id="C2")]))
    fs = fc.feature_set()
    for col in fs.feature_columns:
        if col not in mart.columns:
            mart[col] = 0
    X, y = fc.prepare_xy(mart, fs=fs)
    assert list(X.columns) == fs.feature_columns
    assert "customer_segment_seed" not in X.columns
    for col in fs.categorical_columns:
        assert str(X[col].dtype) == "category"
    assert y.dtype == np.int8
