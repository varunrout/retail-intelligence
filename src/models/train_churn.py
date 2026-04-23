"""Churn V2 model training (LightGBM).

Single LightGBM classifier with native handling of nulls and categoricals.
The hyperparameters are intentionally close to defaults plus the
``is_unbalance`` flag for the 19% positive rate. Tuning is deferred (see
analysis notebook §8 "Out of scope").
"""
from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
)

from src.features.features_churn import FeatureSet, feature_set


# ── Hyperparameters ────────────────────────────────────────────────────────
DEFAULT_LGBM_PARAMS: dict = {
    "objective": "binary",
    "metric": "binary_logloss",
    # Note: is_unbalance/scale_pos_weight intentionally NOT set. The 19%
    # positive rate is moderate; using them produced uncalibrated scores
    # (all probabilities < 0.5) and triggered early-stopping at iter 9.
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
    "deterministic": True,
}

DEFAULT_NUM_BOOST_ROUND = 1500
DEFAULT_EARLY_STOPPING_ROUNDS = 75


@dataclass
class TrainedChurnModel:
    booster: lgb.Booster
    feature_set: FeatureSet
    best_iteration: int

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy()
        for col in self.feature_set.categorical_columns:
            if col in X.columns:
                X[col] = X[col].astype("category")
        return self.booster.predict(X, num_iteration=self.best_iteration)


# ── Training ───────────────────────────────────────────────────────────────
def train_churn_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    *,
    fs: FeatureSet | None = None,
    params: dict | None = None,
    num_boost_round: int = DEFAULT_NUM_BOOST_ROUND,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
) -> TrainedChurnModel:
    fs = fs or feature_set()
    params = {**DEFAULT_LGBM_PARAMS, **(params or {})}

    cat_cols = [c for c in fs.categorical_columns if c in X_train.columns]
    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols, free_raw_data=False)
    valid_set = lgb.Dataset(X_valid, label=y_valid, categorical_feature=cat_cols, reference=train_set, free_raw_data=False)

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return TrainedChurnModel(
        booster=booster,
        feature_set=fs,
        best_iteration=booster.best_iteration or num_boost_round,
    )


# ── Evaluation ─────────────────────────────────────────────────────────────
def evaluate(
    y_true: pd.Series,
    y_score: np.ndarray,
    *,
    label: str = "v2",
) -> pd.DataFrame:
    """Return single-row evaluation summary aligned with phase6 columns."""
    y_arr = y_true.values
    y_pred_05 = (y_score >= 0.5).astype(int)
    accuracy = float((y_pred_05 == y_arr).mean())

    return pd.DataFrame([{
        "model": f"lightgbm_{label}",
        "n_test": int(len(y_true)),
        "churn_rate_test": float(y_true.mean()),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "log_loss": float(log_loss(y_arr, np.clip(y_score, 1e-6, 1 - 1e-6))),
        "brier_score": float(brier_score_loss(y_true, y_score)),
        "accuracy_at_0_50": accuracy,
        "balanced_accuracy_at_0_50": float(balanced_accuracy_score(y_arr, y_pred_05)),
        "mcc_at_0_50": float(matthews_corrcoef(y_arr, y_pred_05)),
    }])


def threshold_diagnostics(
    y_true: pd.Series,
    y_score: np.ndarray,
    *,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.round(np.arange(0.10, 0.95, 0.05), 2)
    y_arr = y_true.values
    rows = []
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        tp = int(((pred == 1) & (y_arr == 1)).sum())
        fp = int(((pred == 1) & (y_arr == 0)).sum())
        tn = int(((pred == 0) & (y_arr == 0)).sum())
        fn = int(((pred == 0) & (y_arr == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        rows.append({
            "threshold": float(t),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "positive_rate": float(pred.mean()),
        })
    return pd.DataFrame(rows)


def threshold_selection(diag: pd.DataFrame, *, precision_floor: float = 0.70) -> pd.DataFrame:
    """Two operating points: max-F1 and precision >= floor."""
    rows = []
    max_f1 = diag.loc[diag["f1"].idxmax()]
    rows.append({
        "selection_rule": "max_f1",
        "threshold": float(max_f1["threshold"]),
        "precision": float(max_f1["precision"]),
        "recall": float(max_f1["recall"]),
        "f1": float(max_f1["f1"]),
        "specificity": float(max_f1["specificity"]),
    })
    above = diag[diag["precision"] >= precision_floor]
    if len(above):
        pf = above.loc[above["recall"].idxmax()]
        rows.append({
            "selection_rule": f"precision_floor_{precision_floor:.2f}".replace(".", "_"),
            "threshold": float(pf["threshold"]),
            "precision": float(pf["precision"]),
            "recall": float(pf["recall"]),
            "f1": float(pf["f1"]),
            "specificity": float(pf["specificity"]),
        })
    return pd.DataFrame(rows)


def feature_importance_table(model: TrainedChurnModel, top_n: int = 30) -> pd.DataFrame:
    booster = model.booster
    imp = pd.DataFrame({
        "feature": booster.feature_name(),
        "importance_gain": booster.feature_importance(importance_type="gain"),
        "importance_split": booster.feature_importance(importance_type="split"),
    })
    imp = imp.sort_values("importance_gain", ascending=False).head(top_n).reset_index(drop=True)
    return imp


def pr_curve_points(y_true: pd.Series, y_score: np.ndarray) -> pd.DataFrame:
    p, r, t = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns thresholds of length n-1
    return pd.DataFrame({
        "precision": p[:-1],
        "recall": r[:-1],
        "threshold": t,
    })
