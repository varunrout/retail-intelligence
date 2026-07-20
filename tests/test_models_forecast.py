"""Smoke tests for src/models/train_forecast.py — tiny synthetic fits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.features_forecast import FeatureSet
from src.models import train_forecast as tf


def _synthetic_series_df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "category": pd.Categorical(rng.choice(["fashion", "beauty"], n)),
            "roll_8w_avg": rng.uniform(0, 20, n),
            "avg_starting_inventory": rng.uniform(0, 50, n),
            "units_sold": rng.uniform(0, 30, n),
        }
    )
    return df


def _fs() -> FeatureSet:
    return FeatureSet(
        feature_columns=["category", "roll_8w_avg", "avg_starting_inventory"],
        categorical_columns=["category"],
    )


def test_train_forecast_fits_and_predicts_nonnegative():
    df = _synthetic_series_df(n=300)
    train_df, valid_df = df.iloc[:200], df.iloc[200:250]
    model = tf.train_forecast(
        train_df, valid_df, fs=_fs(), num_boost_round=30, early_stopping_rounds=10
    )
    assert model.best_iteration >= 1
    assert model.available_features == ["category", "roll_8w_avg", "avg_starting_inventory"]

    preds = model.predict(df.iloc[250:])
    assert preds.shape == (50,)
    assert (preds >= 0).all()  # ForecastModel.predict clips negatives to 0


def test_train_forecast_drops_unavailable_feature_columns():
    df = _synthetic_series_df(n=200).drop(columns=["avg_starting_inventory"])
    train_df, valid_df = df.iloc[:150], df.iloc[150:]
    model = tf.train_forecast(train_df, valid_df, fs=_fs(), num_boost_round=20)
    assert "avg_starting_inventory" not in model.available_features


def test_smape_hand_computed():
    y_true = np.array([10.0, 0.0])
    y_pred = np.array([5.0, 0.0])
    # row0: 2*|10-5|/(10+5) = 2/3; row1: denom=0 -> masked out
    assert tf._smape(y_true, y_pred) == pytest.approx(2 / 3)


def test_evaluate_forecast_summary_and_per_category():
    test_df = pd.DataFrame(
        {
            "units_sold": [10.0, 20.0, 5.0, 15.0],
            "category": ["fashion", "fashion", "beauty", "beauty"],
        }
    )
    preds = np.array([12.0, 18.0, 5.0, 10.0])
    summary, per_cat = tf.evaluate_forecast(test_df, preds)

    assert summary["mae"].iloc[0] == pytest.approx(np.mean(np.abs(test_df["units_sold"] - preds)))
    assert summary["n_test"].iloc[0] == 4
    assert set(per_cat["category"]) == {"fashion", "beauty"}
    fashion_row = per_cat[per_cat["category"] == "fashion"].iloc[0]
    assert fashion_row["n"] == 2
    assert fashion_row["mean_bias"] == pytest.approx(np.mean([12.0 - 10.0, 18.0 - 20.0]))


def test_feature_importance_table_after_training():
    df = _synthetic_series_df(n=200)
    model = tf.train_forecast(df.iloc[:150], df.iloc[150:], fs=_fs(), num_boost_round=20)
    imp = tf.feature_importance_table(model, top_n=2)
    assert len(imp) <= 2
    assert list(imp.columns) == ["rank", "feature", "gain", "split"]
    assert imp["rank"].tolist() == list(range(1, len(imp) + 1))
