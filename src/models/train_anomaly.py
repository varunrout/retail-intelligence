"""Anomaly Detection V2 model training — supervised LightGBM binary classifier.

Key design decisions (findings from anomaly_analysis.ipynb):

  Algorithm    : LightGBM binary classifier (Finding 4.A).
                 Phase11 used unsupervised-only (IF AP=0.058) despite
                 returns_hidden_labels.csv being available.  A supervised
                 model on the same features achieves CV AP=0.791 ± 0.017
                 (14× improvement) and test AP=0.595 on time-ordered holdout.

  Class weight : scale_pos_weight=125 derived from training split only
                 (Finding 4.C).  0.72% prevalence ≈ 138 negatives per
                 positive overall; training-split ratio is computed at runtime.

  Features     : 12 features: 8 numeric + 4 binary OHE flags for return_reason
                 (suspected_abuse), category (electronics), discount_band
                 (low_discount), return_risk_band (high).  Top AUROC signals:
                 item_net_price (0.925), refund_amount (0.924), item_margin
                 (0.913), prior_customer_return_rate (0.874).

  Split        : Time-ordered 80/20 (Finding 5.C).  prior_customer_return_rate
                 encodes temporal history — shuffled CV leaks future labels.

  Threshold    : Flag rate configurable as a score percentile (Finding 6.C).
                 Default 1% flag rate gives precision≈33%, recall≈79%
                 vs phase11 IQR at 12% flag rate with 2.7% precision.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LGBM_PARAMS: dict = dict(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    colsample_bytree=0.8,
    subsample=0.8,
    subsample_freq=1,
    random_state=42,
    verbose=-1,
)

DEFAULT_FLAG_RATE: float = 0.01  # 1% of returns flagged for review


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_anomaly_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    scale_pos_weight: float,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> lgb.LGBMClassifier:
    """Fit a LightGBM binary classifier for abuse-return detection.

    Parameters
    ----------
    X_train, y_train:
        Training features and binary abuse labels.
    scale_pos_weight:
        neg/pos ratio from the training split (Finding 4.C).
    X_val, y_val:
        Optional validation set.  Passed to ``eval_set`` for early-stopping
        monitoring only (not used for hyper-parameter tuning).

    Returns
    -------
    Fitted ``LGBMClassifier``.
    """
    clf = lgb.LGBMClassifier(scale_pos_weight=scale_pos_weight, **LGBM_PARAMS)

    fit_kwargs: dict = {}
    if X_val is not None and y_val is not None:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["callbacks"] = [lgb.log_evaluation(period=-1)]

    clf.fit(X_train, y_train, **fit_kwargs)
    return clf


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def threshold_at_flag_rate(scores: np.ndarray, flag_rate: float) -> float:
    """Return the score threshold that flags ``flag_rate`` fraction of returns.

    Parameters
    ----------
    scores:
        Model probability scores for positive class.
    flag_rate:
        Fraction of records to flag (e.g. 0.01 = 1%).

    Returns
    -------
    Score threshold (float).
    """
    return float(np.percentile(scores, 100.0 * (1.0 - flag_rate)))


def evaluate(
    y_true: np.ndarray,
    scores: np.ndarray,
    flag_rate: float = DEFAULT_FLAG_RATE,
) -> dict[str, float]:
    """Compute precision, recall, F1 and AP at a given flag rate.

    Parameters
    ----------
    y_true:
        Binary ground-truth labels.
    scores:
        Predicted positive-class probabilities.
    flag_rate:
        Operational flag rate (fraction of records flagged).

    Returns
    -------
    dict with keys: ap, flagged, flag_rate_pct, precision, recall, f1,
    threshold.
    """
    ap = float(average_precision_score(y_true, scores))

    thr = threshold_at_flag_rate(scores, flag_rate)
    flags = scores >= thr
    tp = int(((flags) & (y_true == 1)).sum())
    fp = int(((flags) & (y_true == 0)).sum())
    fn = int(((~flags) & (y_true == 1)).sum())
    flagged = int(flags.sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "ap": ap,
        "flagged": flagged,
        "flag_rate_pct": flag_rate * 100,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": thr,
    }


def feature_importance_table(
    model: lgb.LGBMClassifier,
    feature_names: list[str],
    importance_type: str = "gain",
) -> pd.DataFrame:
    """Return a sorted feature-importance DataFrame.

    Parameters
    ----------
    model:
        Fitted LGBMClassifier.
    feature_names:
        List of feature names in the same order as training columns.
    importance_type:
        ``"gain"`` (default) or ``"split"``.

    Returns
    -------
    DataFrame with columns ``feature``, ``importance``, sorted descending.
    """
    importances = model.booster_.feature_importance(importance_type=importance_type)
    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def pr_curve_df(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> pd.DataFrame:
    """Return precision-recall curve as a DataFrame for plotting.

    Returns
    -------
    DataFrame with columns ``precision``, ``recall``, ``threshold``
    (threshold column has one fewer row; aligned by dropping the last
    precision/recall entry per sklearn convention).
    """
    prec, rec, thr = precision_recall_curve(y_true, scores)
    # sklearn returns len(thr) = len(prec) - 1
    return pd.DataFrame(
        {
            "precision": prec[:-1],
            "recall": rec[:-1],
            "threshold": thr,
        }
    )
