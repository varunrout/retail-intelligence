"""Unit tests for src/features/features_uplift.py — data-independent, in-memory frames."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import features_uplift as fu


def test_enrich_with_customer_mart_keeps_only_declared_columns():
    mart = pd.DataFrame({"customer_id": ["C1", "C2"], "campaign_id": ["CMP1", "CMP1"]})
    cust_mart = pd.DataFrame(
        {
            "customer_id": ["C1", "C2"],
            "tenure_days": [100, 200],
            "total_orders": [3, 5],
            "some_unrelated_column": ["x", "y"],
        }
    )
    out = fu.enrich_with_customer_mart(mart, cust_mart)
    assert "tenure_days" in out.columns
    assert "total_orders" in out.columns
    assert "some_unrelated_column" not in out.columns
    assert len(out) == 2


def test_enrich_with_customer_mart_deduplicates_customer_rows():
    mart = pd.DataFrame({"customer_id": ["C1"], "campaign_id": ["CMP1"]})
    cust_mart = pd.DataFrame(
        {"customer_id": ["C1", "C1"], "tenure_days": [100, 999]}
    )  # duplicate customer row
    out = fu.enrich_with_customer_mart(mart, cust_mart)
    assert len(out) == 1


def _base_row(**overrides) -> dict:
    row = {
        "customer_id": "C1",
        "campaign_id": "CMP1",
        "assignment_datetime": "2025-03-15 10:00:00",
        "customer_value_band": "mid",
        "campaign_type": "retention",
        "pre_90d_orders": 2,
        "pre_90d_revenue": 150.0,
        "targeting_rule_source": "rule_based_v1",
        "treatment_flag": True,
        "response_flag_30d": 1,
        "tenure_days": 400,
    }
    row.update(overrides)
    return row


def test_build_features_engineers_expected_columns():
    df = pd.DataFrame([_base_row(), _base_row(customer_id="C2", pre_90d_orders=0)])
    out = fu.build_features(df)

    assert out["assignment_month"].iloc[0] == 3
    assert out["assignment_dayofweek"].iloc[0] == pd.Timestamp("2025-03-15").dayofweek
    assert out["segment_campaign"].iloc[0] == "mid_retention"
    assert out["is_new_customer"].iloc[0] == 0  # pre_90d_orders=2
    assert out["is_new_customer"].iloc[1] == 1  # pre_90d_orders=0
    assert out["targeting_is_rule_based"].iloc[0] == 1  # starts with "rule"
    assert out["log_pre_90d_revenue"].iloc[0] == pytest.approx(np.log1p(150.0))
    assert out["order_density_90d"].iloc[0] == pytest.approx(2 / 90.0)


def test_build_features_handles_missing_targeting_rule_source():
    df = pd.DataFrame([_base_row(targeting_rule_source=None)])
    out = fu.build_features(df)
    assert out["targeting_is_rule_based"].iloc[0] == 0


def test_time_ordered_split_orders_by_assignment_datetime():
    rows = [
        _base_row(customer_id=f"C{i}", assignment_datetime=f"2025-01-{i + 1:02d} 00:00:00")
        for i in range(10)
    ]
    df = pd.DataFrame(rows)
    train, test = fu.time_ordered_split(df, test_size=0.2)
    assert len(test) == 2
    assert pd.to_datetime(train[fu.TIME_KEY]).max() <= pd.to_datetime(test[fu.TIME_KEY]).min()


def test_prepare_xy_applies_categorical_dtypes():
    df = fu.build_features(pd.DataFrame([_base_row()]))
    fs = fu.feature_set()
    for col in fs.feature_columns:
        if col not in df.columns:
            df[col] = 0
    X, y, t = fu.prepare_xy(df, fs=fs)
    for col in fs.categorical_columns:
        if col in X.columns:
            assert str(X[col].dtype) == "category"
    assert set(y.unique()) <= {0, 1}
    assert set(t.unique()) <= {0, 1}
