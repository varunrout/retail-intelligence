"""Smoke tests for src/models/train_churn.py — tiny synthetic fits, not full training runs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import train_churn as tc


def _synthetic_xy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    X = pd.DataFrame(
        {
            "total_orders": rng.integers(0, 20, n).astype(float),
            "recency_days": rng.integers(0, 365, n).astype(float),
            # already cast to category, matching what features_churn.prepare_xy does
            # before handing X to train_churn_model — the trainer itself does not cast.
            "income_band": pd.Categorical(rng.choice(["low", "mid", "high"], n)),
        }
    )
    y = pd.Series((signal + rng.normal(scale=0.5, size=n) > 0).astype(int), name="churn_flag_90d")
    return X, y


def test_train_churn_model_fits_and_predicts_probabilities():
    X, y = _synthetic_xy(n=300)
    X_train, X_valid = X.iloc[:200], X.iloc[200:250]
    y_train, y_valid = y.iloc[:200], y.iloc[200:250]

    from src.features.features_churn import FeatureSet

    fs = FeatureSet(
        feature_columns=["total_orders", "recency_days", "income_band"],
        categorical_columns=["income_band"],
    )
    model = tc.train_churn_model(
        X_train, y_train, X_valid, y_valid, fs=fs, num_boost_round=50, early_stopping_rounds=10
    )
    assert model.best_iteration >= 1

    X_test = X.iloc[250:]
    proba = model.predict_proba(X_test)
    assert proba.shape == (len(X_test),)
    assert np.all((proba >= 0) & (proba <= 1))


def test_evaluate_returns_expected_columns_and_bounds():
    rng = np.random.default_rng(1)
    y_true = pd.Series(rng.integers(0, 2, 200))
    y_score = np.clip(y_true.to_numpy() * 0.6 + rng.uniform(0, 0.4, 200), 0.01, 0.99)
    out = tc.evaluate(y_true, y_score, label="test")
    assert len(out) == 1
    assert out["model"].iloc[0] == "lightgbm_test"
    assert 0.0 <= out["roc_auc"].iloc[0] <= 1.0
    assert 0.0 <= out["accuracy_at_0_50"].iloc[0] <= 1.0


def test_threshold_diagnostics_matches_hand_computed_confusion_matrix():
    y_true = pd.Series([1, 1, 0, 0, 1])
    y_score = np.array([0.9, 0.4, 0.3, 0.2, 0.6])
    diag = tc.threshold_diagnostics(y_true, y_score, thresholds=np.array([0.5]))
    row = diag.iloc[0]
    # at threshold 0.5: predicted positive = {0.9, 0.6} -> indices 0, 4 (both true=1)
    assert row["tp"] == 2
    assert row["fp"] == 0
    assert row["fn"] == 1  # index 1 (y=1, score=0.4) missed
    assert row["tn"] == 2
    assert row["precision"] == pytest.approx(1.0)
    assert row["recall"] == pytest.approx(2 / 3)


def test_threshold_selection_picks_max_f1_and_precision_floor():
    diag = pd.DataFrame(
        {
            "threshold": [0.3, 0.5, 0.7],
            "precision": [0.4, 0.6, 0.9],
            "recall": [0.9, 0.7, 0.3],
            "f1": [0.55, 0.65, 0.45],
            "specificity": [0.5, 0.7, 0.9],
        }
    )
    out = tc.threshold_selection(diag, precision_floor=0.70)
    max_f1_row = out[out["selection_rule"] == "max_f1"].iloc[0]
    assert max_f1_row["threshold"] == pytest.approx(0.5)

    floor_row = out[out["selection_rule"].str.startswith("precision_floor")].iloc[0]
    assert floor_row["threshold"] == pytest.approx(0.7)  # only row with precision >= 0.70


def test_threshold_selection_omits_floor_row_when_none_qualify():
    diag = pd.DataFrame(
        {
            "threshold": [0.3, 0.5],
            "precision": [0.2, 0.3],
            "recall": [0.9, 0.7],
            "f1": [0.4, 0.5],
            "specificity": [0.5, 0.6],
        }
    )
    out = tc.threshold_selection(diag, precision_floor=0.70)
    assert len(out) == 1
    assert out["selection_rule"].iloc[0] == "max_f1"


def test_pr_curve_points_shape():
    y_true = pd.Series([0, 1, 1, 0, 1])
    y_score = np.array([0.1, 0.8, 0.6, 0.3, 0.9])
    out = tc.pr_curve_points(y_true, y_score)
    assert set(out.columns) == {"precision", "recall", "threshold"}
    assert len(out) > 0


def test_feature_importance_table_after_training():
    X, y = _synthetic_xy(n=250)
    from src.features.features_churn import FeatureSet

    fs = FeatureSet(
        feature_columns=["total_orders", "recency_days", "income_band"],
        categorical_columns=["income_band"],
    )
    model = tc.train_churn_model(
        X.iloc[:150],
        y.iloc[:150],
        X.iloc[150:200],
        y.iloc[150:200],
        fs=fs,
        num_boost_round=30,
        early_stopping_rounds=10,
    )
    imp = tc.feature_importance_table(model, top_n=2)
    assert len(imp) <= 2
    assert set(imp.columns) == {"feature", "importance_gain", "importance_split"}
