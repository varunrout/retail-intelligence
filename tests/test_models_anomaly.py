"""Smoke tests for src/models/train_anomaly.py — tiny synthetic fits."""

from __future__ import annotations

import numpy as np
import pytest

from src.models import train_anomaly as ta


def _synthetic_xy(n=300, seed=0, pos_rate=0.1):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pos_rate).astype(int)
    # signal-bearing feature so the model has something real to learn
    X = np.column_stack([y * 2 + rng.normal(scale=0.5, size=n), rng.normal(size=n)])
    return X, y


def test_train_anomaly_model_fits_and_predicts_proba():
    X, y = _synthetic_xy(n=400, pos_rate=0.15)
    X_train, y_train = X[:300], y[:300]
    X_val, y_val = X[300:], y[300:]
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    clf = ta.train_anomaly_model(X_train, y_train, scale_pos_weight, X_val, y_val)
    proba = clf.predict_proba(X_val)[:, 1]
    assert proba.shape == (len(X_val),)
    assert np.all((proba >= 0) & (proba <= 1))


def test_train_anomaly_model_without_validation_set():
    X, y = _synthetic_xy(n=200, pos_rate=0.15)
    scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    clf = ta.train_anomaly_model(X, y, scale_pos_weight)
    assert hasattr(clf, "booster_")


def test_threshold_at_flag_rate_matches_percentile():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.9])
    thr = ta.threshold_at_flag_rate(scores, flag_rate=0.2)  # top 20% = top 1 of 5
    assert thr == pytest.approx(np.percentile(scores, 80.0))
    assert (scores >= thr).sum() == 1


def test_evaluate_computes_precision_recall_f1_at_flag_rate():
    y_true = np.array([1, 0, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    # flag_rate=0.4 -> top 2 scores flagged: index 0 (y=1, TP), index 1 (y=0, FP).
    # The other positive (index 2, score 0.3) is not in the top 2 -> FN.
    out = ta.evaluate(y_true, scores, flag_rate=0.4)
    assert out["flagged"] == 2
    assert out["precision"] == pytest.approx(0.5)
    assert out["recall"] == pytest.approx(0.5)  # 1 of 2 true positives flagged
    assert 0.0 <= out["ap"] <= 1.0


def test_evaluate_handles_no_flags_gracefully():
    y_true = np.array([0, 0, 0])
    scores = np.array([0.1, 0.1, 0.1])
    out = ta.evaluate(y_true, scores, flag_rate=0.0001)
    assert out["precision"] == 0.0
    assert out["recall"] == 0.0
    assert out["f1"] == 0.0


def test_feature_importance_table_after_training():
    X, y = _synthetic_xy(n=200, pos_rate=0.15)
    scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    clf = ta.train_anomaly_model(X, y, scale_pos_weight)
    imp = ta.feature_importance_table(clf, feature_names=["f1", "f2"])
    assert set(imp.columns) == {"feature", "importance"}
    assert len(imp) == 2
    assert imp["importance"].is_monotonic_decreasing


def test_pr_curve_df_shape():
    y_true = np.array([0, 1, 1, 0, 1])
    scores = np.array([0.1, 0.8, 0.6, 0.3, 0.9])
    out = ta.pr_curve_df(y_true, scores)
    assert set(out.columns) == {"precision", "recall", "threshold"}
    assert len(out) > 0
