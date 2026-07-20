"""Unit tests for src/features/features_segmentation.py — data-independent, in-memory frames."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import features_segmentation as fs


def _cust_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["C1", "C2"],
            "total_orders": [10.0, np.nan],  # C2 is a non-purchaser
            "total_net_revenue": [500.0, np.nan],
            "avg_order_value": [50.0, np.nan],
            "avg_basket_size": [2.0, np.nan],
            "revenue_per_order": [50.0, np.nan],
            "total_units": [20.0, np.nan],
            "avg_item_discount_pct": [0.1, np.nan],
            "total_returns": [1.0, np.nan],
            "return_rate_per_unit": [0.05, np.nan],
            "online_order_share": [0.6, np.nan],
            "store_order_share": [0.4, np.nan],
            "avg_item_margin": [0.3, np.nan],
            "total_discount_amount": [5.0, np.nan],
            "total_refund_amount": [1.0, np.nan],
            "recency_days": [10.0, np.nan],
            "tenure_days": [365.0, 100.0],
            "total_sessions": [20.0, 5.0],
            "avg_session_minutes": [8.0, 3.0],
            "avg_pages_viewed": [12.0, 4.0],
            "sessions_add_to_cart": [5.0, 0.0],
            "sessions_with_purchase": [4.0, 0.0],
        }
    )


def test_engineer_features_flags_non_purchasers():
    out = fs.engineer_features(_cust_frame())
    assert out.loc[out["customer_id"] == "C1", "is_non_purchaser"].iloc[0] == 0
    assert out.loc[out["customer_id"] == "C2", "is_non_purchaser"].iloc[0] == 1


def test_engineer_features_fills_purchase_columns_with_zero_for_non_purchasers():
    out = fs.engineer_features(_cust_frame())
    c2 = out.loc[out["customer_id"] == "C2"].iloc[0]
    for col in fs.PURCHASE_COLS:
        assert c2[col] == 0


def test_engineer_features_recency_falls_back_to_tenure_for_non_purchasers():
    out = fs.engineer_features(_cust_frame())
    c2 = out.loc[out["customer_id"] == "C2"].iloc[0]
    assert c2["recency_days"] == pytest.approx(100.0)  # tenure_days for C2


def test_engineer_features_log1p_transforms():
    out = fs.engineer_features(_cust_frame())
    c1 = out.loc[out["customer_id"] == "C1"].iloc[0]
    assert c1["log_total_orders"] == pytest.approx(np.log1p(10.0))
    assert c1["log_total_net_revenue"] == pytest.approx(np.log1p(500.0))


def test_engineer_features_behavioural_ratios():
    out = fs.engineer_features(_cust_frame())
    c1 = out.loc[out["customer_id"] == "C1"].iloc[0]
    assert c1["purchase_rate"] == pytest.approx(4.0 / 20.0)
    assert c1["order_rate_per_month"] == pytest.approx(10.0 / (365.0 / 30.0))

    c2 = out.loc[out["customer_id"] == "C2"].iloc[0]
    assert c2["purchase_rate"] == 0.0  # total_sessions > 0 but zero purchase -> 0, not div-by-zero
    assert c2["order_rate_per_month"] == 0.0  # total_orders was filled to 0


def _varied_buyer_frame(n: int = 8) -> pd.DataFrame:
    """n buyer rows with actual variance across BUYER_FEATURES (avoids a
    degenerate zero-variance PCA input, which would emit an sklearn
    RuntimeWarning and defeat the point of exercising the real fit path)."""
    rng = np.random.default_rng(0)
    fe = fs.engineer_features(
        pd.DataFrame(
            {
                "customer_id": [f"C{i}" for i in range(n)],
                "total_orders": rng.integers(1, 30, n).astype(float),
                "total_net_revenue": rng.uniform(50, 2000, n),
                "avg_order_value": rng.uniform(10, 200, n),
                "avg_basket_size": rng.uniform(1, 5, n),
                "revenue_per_order": rng.uniform(10, 200, n),
                "total_units": rng.integers(1, 60, n).astype(float),
                "avg_item_discount_pct": rng.uniform(0, 0.4, n),
                "total_returns": rng.integers(0, 5, n).astype(float),
                "return_rate_per_unit": rng.uniform(0, 0.2, n),
                "online_order_share": rng.uniform(0, 1, n),
                "store_order_share": rng.uniform(0, 1, n),
                "avg_item_margin": rng.uniform(0.1, 0.5, n),
                "total_discount_amount": rng.uniform(0, 50, n),
                "total_refund_amount": rng.uniform(0, 20, n),
                "recency_days": rng.integers(1, 300, n).astype(float),
                "tenure_days": rng.integers(30, 900, n).astype(float),
                "total_sessions": rng.integers(1, 50, n).astype(float),
                "avg_session_minutes": rng.uniform(1, 15, n),
                "avg_pages_viewed": rng.uniform(1, 20, n),
                "sessions_add_to_cart": rng.integers(0, 20, n).astype(float),
                "sessions_with_purchase": rng.integers(0, 15, n).astype(float),
            }
        )
    )
    return fe


def test_segmentation_pipeline_fit_transform_shape_matches_pca_components():
    buyers = _varied_buyer_frame(n=8)
    pipeline = fs.SegmentationPipeline()
    reduced = pipeline.fit_transform(buyers)
    assert reduced.shape == (len(buyers), fs.PCA_N_COMPONENTS)


def test_segmentation_pipeline_transform_reuses_fitted_scaler():
    buyers = _varied_buyer_frame(n=8)
    pipeline = fs.SegmentationPipeline()
    fitted = pipeline.fit_transform(buyers)
    transformed = pipeline.transform(buyers)
    assert transformed.shape == (len(buyers), fs.PCA_N_COMPONENTS)
    # transform() on the same rows the pipeline was just fit on should reproduce fit_transform()
    np.testing.assert_allclose(fitted, transformed)
