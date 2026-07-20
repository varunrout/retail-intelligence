"""Unit tests for src/features/features_forecast.py — data-independent, in-memory frames."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import features_forecast as ff


def _series_frame(n_weeks: int, product_id="P1", store="S1", start="2025-01-06") -> pd.DataFrame:
    weeks = pd.date_range(start, periods=n_weeks, freq="7D")
    return pd.DataFrame(
        {
            "product_id": product_id,
            "store_id_or_online": store,
            "week_start_date": weeks,
            "units_sold": np.arange(1, n_weeks + 1, dtype=float),
            "avg_starting_inventory": [10.0] * n_weeks,
        }
    )


def test_feature_set_deduplicates_and_preserves_order():
    fs = ff.feature_set()
    assert len(fs.feature_columns) == len(set(fs.feature_columns))
    assert "category" in fs.feature_columns
    assert fs.categorical_columns == ff.CATEGORICAL_FEATURES


def test_build_lag_features_shifts_are_strictly_backward_looking():
    df = _series_frame(n_weeks=15)
    out = ff.build_lag_features(df)

    # lag_2w at row i should equal units_sold at row i-2 within the series
    units = out["units_sold"].to_numpy()
    lag2 = out["lag_2w"].to_numpy()
    assert np.isnan(lag2[0]) and np.isnan(lag2[1])
    assert lag2[2] == units[0]
    assert lag2[5] == units[3]

    # has_yoy_lag is 0 until 52 weeks of history exist (we only have 15)
    assert (out["has_yoy_lag"] == 0).all()

    # roll_2w_avg at row i = mean(lag_1, lag_2), i.e. excludes current week
    roll2 = out["roll_2w_avg"].to_numpy()
    assert roll2[2] == pytest.approx((units[1] + units[0]) / 2)


def test_build_lag_features_computes_cyclical_calendar_encoding():
    df = _series_frame(n_weeks=3)
    out = ff.build_lag_features(df)
    assert (
        (out["sin_woy"] ** 2 + out["cos_woy"] ** 2).apply(lambda v: v == pytest.approx(1.0)).all()
    )


def test_build_lag_features_flags_missing_inventory():
    df = _series_frame(n_weeks=3)
    df.loc[0, "avg_starting_inventory"] = np.nan
    out = ff.build_lag_features(df)
    assert out["has_inventory"].tolist() == [0, 1, 1]


def test_compute_and_attach_series_means_use_train_only():
    train = pd.DataFrame(
        {
            "product_id": ["P1", "P1", "P2"],
            "store_id_or_online": ["S1", "S1", "S1"],
            "units_sold": [10.0, 20.0, 5.0],
        }
    )
    product_means, store_means = ff.compute_series_means(train)
    assert product_means.set_index("product_id")["product_mean_demand"]["P1"] == pytest.approx(15.0)

    test = pd.DataFrame({"product_id": ["P1"], "store_id_or_online": ["S1"]})
    out = ff.attach_series_means(test, product_means, store_means)
    assert out["product_mean_demand"].iloc[0] == pytest.approx(15.0)
    assert out["store_mean_demand"].iloc[0] == pytest.approx(35.0 / 3)


def test_filter_short_series_drops_below_threshold():
    short = _series_frame(n_weeks=3, product_id="SHORT")
    long = _series_frame(n_weeks=10, product_id="LONG")
    df = pd.concat([short, long], ignore_index=True)
    out = ff.filter_short_series(df, min_weeks=8)
    assert set(out["product_id"].unique()) == {"LONG"}


def test_week_time_split_boundary_is_exclusive_on_train_side():
    df = _series_frame(n_weeks=6, start="2025-01-06")  # weeks: 01-06 .. 02-03
    train, test = ff.week_time_split(df, split_week="2025-01-27")
    assert (pd.to_datetime(train["week_start_date"]) < pd.Timestamp("2025-01-27")).all()
    assert (pd.to_datetime(test["week_start_date"]) >= pd.Timestamp("2025-01-27")).all()
    assert len(train) + len(test) == len(df)


def test_prepare_xy_only_includes_available_columns():
    df = pd.DataFrame({"category": ["fashion"], "units_sold": [3.0]})
    X, y = ff.prepare_xy(df)
    assert list(X.columns) == ["category"]  # other feature_set() columns absent from df
    assert y[0] == 3.0
