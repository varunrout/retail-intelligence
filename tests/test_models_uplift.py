"""Smoke tests for src/models/train_uplift.py — tiny synthetic fits."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.features_uplift import FeatureSet
from src.models import train_uplift as tu


def _synthetic_campaign_df(n=400, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    campaign_id = rng.choice(["CMP1", "CMP2"], n)
    treatment = rng.integers(0, 2, n)
    persuadability = rng.uniform(0, 1, n)
    base_conv = 0.1
    p_conv = base_conv + treatment * 0.15 * persuadability
    response = (rng.random(n) < p_conv).astype(int)
    return pd.DataFrame(
        {
            "customer_id": [f"C{i}" for i in range(n)],
            "campaign_id": campaign_id,
            "treatment_flag": treatment,
            "response_flag_30d": response,
            "pre_90d_orders": rng.integers(0, 10, n).astype(float),
            "pre_90d_revenue": rng.uniform(0, 500, n),
        }
    )


def _fs() -> FeatureSet:
    return FeatureSet(feature_columns=["pre_90d_orders", "pre_90d_revenue"], categorical_columns=[])


def test_train_xlearner_fits_and_predicts_finite_ite():
    df = _synthetic_campaign_df(n=400)
    train_df, valid_df = df.iloc[:300], df.iloc[300:]
    model = tu.train_xlearner(
        train_df,
        valid_df,
        fs=_fs(),
        num_boost_round=30,
        early_stopping_rounds=10,
    )
    assert set(model.campaign_propensity.keys()) <= {"CMP1", "CMP2"}

    X = df[["pre_90d_orders", "pre_90d_revenue"]]
    ite = model.predict_ite(X, df["campaign_id"])
    assert ite.shape == (len(df),)
    assert np.isfinite(ite).all()


def test_train_xlearner_unseen_campaign_falls_back_to_half_propensity():
    df = _synthetic_campaign_df(n=300)
    model = tu.train_xlearner(df.iloc[:250], df.iloc[250:], fs=_fs(), num_boost_round=20)
    X = df[["pre_90d_orders", "pre_90d_revenue"]].iloc[:5]
    ite_unseen = model.predict_ite(X, pd.Series(["CMP_NEVER_SEEN"] * 5))
    assert np.isfinite(ite_unseen).all()


def test_evaluate_uplift_decile_table_and_summary():
    rng = np.random.default_rng(1)
    n = 200
    df = pd.DataFrame(
        {
            "treatment_flag": rng.integers(0, 2, n),
            "response_flag_30d": rng.integers(0, 2, n),
        }
    )
    ite_scores = rng.uniform(-1, 1, n)
    summary, decile_df = tu.evaluate_uplift(df, ite_scores, n_deciles=5, label="test_model")

    assert len(decile_df) == 5
    assert decile_df["decile"].tolist() == [1, 2, 3, 4, 5]
    assert summary["model"].iloc[0] == "test_model"
    assert summary["test_rows"].iloc[0] == n
    assert -1.0 <= summary["overall_ate_test"].iloc[0] <= 1.0


def test_feature_importance_table_after_training():
    df = _synthetic_campaign_df(n=250)
    model = tu.train_xlearner(df.iloc[:200], df.iloc[200:], fs=_fs(), num_boost_round=20)
    imp = tu.feature_importance_table(model, top_n=2)
    assert len(imp) <= 2
    assert set(imp.columns) == {"feature", "importance_gain", "importance_split"}
