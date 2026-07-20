"""Forecast V2 model training — LightGBM with Tweedie objective.

Key design decisions (findings from forecast_analysis.ipynb):
  Objective  : Tweedie regression — corrects the systematic under-prediction
               bias present in the RF baseline (Finding 3.B) and is
               appropriate for right-skewed non-negative count data (Finding 1.B).
  Features   : lagged rolling means roll_8/13w_avg (F6.B, r≈0.45),
               series-level demand baselines (F6.A, r≈0.27–0.43),
               cyclical calendar encoding (F4.B), inventory + null flags (F4.C).
  Short series: series with <8 weeks are filtered from training (F1.A).
  Categoricals: category, subcategory, price_tier handled natively by LightGBM.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.features.features_forecast import LABEL, FeatureSet, feature_set

# ── LightGBM hyperparameters ───────────────────────────────────────────────
FORECAST_PARAMS: dict = {
    "objective": "tweedie",  # F3.B: corrects systematic under-prediction
    "tweedie_variance_power": 1.5,  # midpoint; stable for right-skewed counts (F1.B)
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
    "seed": 42,
    "deterministic": True,
}

DEFAULT_NUM_BOOST_ROUND = 2000
DEFAULT_EARLY_STOPPING_ROUNDS = 75


@dataclass
class ForecastModel:
    """Container for a fitted LightGBM Forecast V2 model.

    Attributes
    ----------
    booster : lgb.Booster
        Fitted LightGBM booster.
    feature_set : FeatureSet
        Feature set used at training.
    params : dict
        LightGBM parameters used.
    best_iteration : int
        Best boosting round from early stopping.
    available_features : list[str]
        Subset of ``feature_set.feature_columns`` that were present in the
        training data (missing engineered cols silently dropped).
    """

    booster: lgb.Booster
    feature_set: FeatureSet
    params: dict
    best_iteration: int
    available_features: list[str] = field(default_factory=list)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return non-negative unit demand predictions.

        Parameters
        ----------
        X : pd.DataFrame
            Features matching ``available_features``.
        """
        X_cat = _apply_categorical_dtypes(X[self.available_features], self.feature_set)
        preds = self.booster.predict(X_cat, num_iteration=self.best_iteration)
        return np.maximum(preds, 0.0).astype(np.float32)


def _apply_categorical_dtypes(X: pd.DataFrame, fs: FeatureSet) -> pd.DataFrame:
    X = X.copy()
    for col in fs.categorical_columns:
        if col in X.columns:
            X[col] = X[col].astype("category")
    return X


def _lgb_dataset(
    X: pd.DataFrame,
    y: np.ndarray,
    cat_cols: list[str],
    *,
    reference: lgb.Dataset | None = None,
) -> lgb.Dataset:
    return lgb.Dataset(
        X,
        label=y,
        categorical_feature=cat_cols,
        reference=reference,
        free_raw_data=False,
    )


def train_forecast(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    *,
    fs: FeatureSet | None = None,
    params: dict | None = None,
    num_boost_round: int = DEFAULT_NUM_BOOST_ROUND,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
) -> ForecastModel:
    """Train a LightGBM Tweedie demand forecast model.

    Parameters
    ----------
    train_df, valid_df
        DataFrames containing features and the ``units_sold`` label.
        ``valid_df`` is used only for early stopping — it is NOT held out
        from the feature-mean joins.
    fs
        Feature set to use; defaults to ``feature_set()``.
    params
        LightGBM param overrides; merged onto ``FORECAST_PARAMS``.
    """
    fs = fs or feature_set()
    merged_params = {**FORECAST_PARAMS, **(params or {})}

    avail = [c for c in fs.feature_columns if c in train_df.columns]
    cat_cols_present = [c for c in fs.categorical_columns if c in avail]

    X_tr = _apply_categorical_dtypes(train_df[avail], fs)
    y_tr = train_df[LABEL].values.astype(float)

    X_va = _apply_categorical_dtypes(valid_df[avail], fs)
    y_va = valid_df[LABEL].values.astype(float)

    train_ds = _lgb_dataset(X_tr, y_tr, cat_cols_present)
    valid_ds = _lgb_dataset(X_va, y_va, cat_cols_present, reference=train_ds)

    booster = lgb.train(
        merged_params,
        train_ds,
        num_boost_round=num_boost_round,
        valid_sets=[train_ds, valid_ds],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )

    return ForecastModel(
        booster=booster,
        feature_set=fs,
        params=merged_params,
        best_iteration=booster.best_iteration,
        available_features=avail,
    )


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 0
    return float(np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]))


def evaluate_forecast(
    test_df: pd.DataFrame,
    preds: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute forecast evaluation metrics.

    Parameters
    ----------
    test_df : pd.DataFrame
        Test set DataFrame (must contain ``units_sold`` and ``category``).
    preds : np.ndarray
        Model predictions aligned row-for-row with ``test_df`` (after
        ``test_df.reset_index(drop=True)`` in the caller).

    Returns
    -------
    summary : pd.DataFrame
        One-row summary of overall MAE, RMSE, sMAPE, and mean bias.
    per_category : pd.DataFrame
        Per-category breakdown (MAE, sMAPE, mean_bias, n).
    """
    test_df = test_df.reset_index(drop=True)
    y = test_df[LABEL].values.astype(float)

    mae = float(mean_absolute_error(y, preds))
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    smape = _smape(y, preds)
    bias = float(np.mean(preds - y))  # positive = over-predict

    summary = pd.DataFrame(
        [
            {
                "model": "lgbm_tweedie_v2",
                "mae": mae,
                "rmse": rmse,
                "smape": smape,
                "mean_bias": bias,
                "n_test": len(y),
            }
        ]
    )

    rows = []
    if "category" in test_df.columns:
        for cat, grp in test_df.groupby("category"):
            idx = grp.index.tolist()
            y_c = y[idx]
            p_c = preds[idx]
            rows.append(
                {
                    "category": cat,
                    "mae": float(mean_absolute_error(y_c, p_c)),
                    "smape": _smape(y_c, p_c),
                    "mean_bias": float(np.mean(p_c - y_c)),
                    "n": len(y_c),
                }
            )
    per_category = pd.DataFrame(rows)

    return summary, per_category


def feature_importance_table(model: ForecastModel, top_n: int = 30) -> pd.DataFrame:
    """Return a ranked feature importance table (gain + split counts)."""
    imp_gain = model.booster.feature_importance(importance_type="gain")
    imp_split = model.booster.feature_importance(importance_type="split")
    names = model.booster.feature_name()
    df = pd.DataFrame({"feature": names, "gain": imp_gain, "split": imp_split})
    df = df.sort_values("gain", ascending=False).head(top_n).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df
