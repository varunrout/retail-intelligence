"""Smoke tests for src/models/train_segmentation.py — tiny synthetic fits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.features_segmentation import BUYER_FEATURES
from src.models import train_segmentation as ts


def _engineered_frame(n_buyers=60, n_non_purchasers=10, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_buyers + n_non_purchasers
    is_non = np.array([0] * n_buyers + [1] * n_non_purchasers)
    data = {col: rng.uniform(0, 5, n) for col in BUYER_FEATURES}
    data.update(
        {
            "customer_id": [f"C{i}" for i in range(n)],
            "is_non_purchaser": is_non,
            "total_net_revenue": np.where(is_non, 0.0, rng.uniform(50, 5000, n)),
            "churn_flag_90d": rng.integers(0, 2, n),
            "avg_order_value": rng.uniform(10, 200, n),
            "recency_days": rng.uniform(0, 300, n),
            "order_rate_per_month": rng.uniform(0, 5, n),
        }
    )
    return pd.DataFrame(data)


def test_pipeline_n_components_within_bounds():
    fe = _engineered_frame()
    buyers = fe[fe["is_non_purchaser"] == 0]
    n = ts.pipeline_n_components(buyers)
    assert 2 <= n <= len(BUYER_FEATURES)


def test_train_segmentation_labels_non_purchasers_separately():
    fe = _engineered_frame(n_buyers=60, n_non_purchasers=10)
    model = ts.train_segmentation(fe, n_buyer_clusters=3, n_init=3)

    assert len(model.labels_) == len(fe)
    non_purch_labels = model.labels_[fe["is_non_purchaser"].to_numpy() == 1]
    assert (non_purch_labels == model.n_buyer_clusters).all()

    buyer_labels = model.labels_[fe["is_non_purchaser"].to_numpy() == 0]
    assert buyer_labels.min() >= 0
    assert buyer_labels.max() < model.n_buyer_clusters
    assert not np.isnan(model.silhouette_buyers)


def test_cluster_profiles_aggregates_correctly():
    fe = _engineered_frame(n_buyers=30, n_non_purchasers=5)
    model = ts.train_segmentation(fe, n_buyer_clusters=2, n_init=2)
    profiles = ts.cluster_profiles(fe, model.labels_, n_buyer_clusters=2)

    assert profiles["n"].sum() == len(fe)
    assert set(profiles["cluster_id"]) == {0, 1, 2}
    non_purch_row = profiles[profiles["segment_name"] == "Non-Purchasers"].iloc[0]
    assert non_purch_row["n"] == 5
    assert non_purch_row["pct_non_purchaser"] == pytest.approx(1.0)


def test_segment_metrics_reports_churn_spread_and_sizes():
    fe = _engineered_frame(n_buyers=30, n_non_purchasers=5)
    model = ts.train_segmentation(fe, n_buyer_clusters=2, n_init=2)
    profiles = ts.cluster_profiles(fe, model.labels_, n_buyer_clusters=2)
    metrics = ts.segment_metrics(model, profiles)

    assert metrics["n_total"] == len(fe)
    assert metrics["n_total_segments"] == 3
    assert metrics["churn_spread_pp"] == pytest.approx(
        (profiles["churn_rate"].max() - profiles["churn_rate"].min()) * 100, abs=0.01
    )
    assert set(metrics["cluster_sizes"].keys()) == {0, 1, 2}
