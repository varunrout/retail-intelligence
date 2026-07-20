"""Churn: like-for-like incremental lift of LightGBM V2 over baselines.

The committed ``phase_churn_v2_vs_baseline.csv`` compares the LightGBM V2 model
against the phase6 Random Forest, but the two were measured under *different
regimes*: the baseline used a stratified-random split with recency-derived
features, the V2 used a time-ordered split, a 90-day maturity filter and no
recency. That comparison confounds the algorithm with the split and the feature
policy, so it cannot answer "does LightGBM actually beat a simple baseline?".

This script builds the ONE matched comparison. Every model is trained and scored
on the IDENTICAL time-ordered split, the SAME mature-cohort population, and the
SAME feature policy (recency excluded, customer_segment_seed excluded). The only
thing that varies is the algorithm. A paired bootstrap over the shared held-out
rows gives a 95 percent CI on the lift, and the verdict only says "adds value"
when that CI excludes zero.

Run:
    python -m analysis.churn_incremental_lift            # real marts
    python -m analysis.churn_incremental_lift --smoke    # synthetic, no data
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import OUTPUTS_DIR, PROCESSED_DIR
from src.features.features_churn import (
    LABEL,
    build_features,
    feature_set,
    filter_active_population,
    filter_mature_cohort,
    prepare_xy,
    time_ordered_split,
)

PREFIX = "churn_incremental_lift"
MATURITY_DAYS = 90
TEST_SIZE = 0.20
N_BOOTSTRAP = int(os.environ.get("LIFT_N_BOOTSTRAP", "2000"))
RF_TREES = int(os.environ.get("LIFT_RF_TREES", "300"))
SEED = 42


def _sklearn_pipeline(estimator, numeric: list[str], categorical: list[str]) -> Pipeline:
    """Fair baseline: median-impute numerics, one-hot the same categoricals."""
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    return Pipeline([("pre", pre), ("clf", estimator)])


def _paired_bootstrap_ci(
    y_true: np.ndarray,
    score_candidate: np.ndarray,
    score_reference: np.ndarray,
    metric,
    n_boot: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> tuple[float, float, float, float]:
    """Bootstrap the paired lift (candidate - reference) on the SAME rows.

    Returns (point_lift, ci_low, ci_high, share_of_draws_above_zero).
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = metric(y_true, score_candidate) - metric(y_true, score_reference)
    lifts = np.empty(n_boot)
    drawn = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if yt.min() == yt.max():  # need both classes for AUC/AP
            continue
        lifts[drawn] = metric(yt, score_candidate[idx]) - metric(yt, score_reference[idx])
        drawn += 1
    lifts = lifts[:drawn]
    ci_low, ci_high = np.percentile(lifts, [2.5, 97.5])
    share_pos = float((lifts > 0).mean())
    return float(point), float(ci_low), float(ci_high), share_pos


def _make_smoke_frame(n: int = 6000, seed: int = SEED) -> pd.DataFrame:
    """Synthetic frame with the columns the churn feature builder needs."""
    rng = np.random.default_rng(seed)
    signup = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 500, n), unit="D")
    tenure = rng.integers(30, 900, n)
    total_orders = rng.poisson(6, n) + 1
    sessions = rng.poisson(20, n) + 1
    add_cart = np.minimum(sessions, rng.poisson(8, n))
    with_purchase = np.minimum(add_cart, rng.poisson(4, n))
    # a real signal: fewer orders per month -> higher churn
    velocity = total_orders / (tenure / 30.0)
    churn_p = 1 / (1 + np.exp(2.0 * (velocity - 1.0)))
    churn = (rng.random(n) < churn_p).astype("int8")
    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:06d}" for i in range(n)],
            "signup_date": signup,
            "customer_value_band": rng.choice(["low", "mid", "high"], n),
            "income_band": rng.choice(["a", "b", "c"], n),
            "total_orders": total_orders,
            "total_net_revenue": rng.gamma(2, 200, n),
            "avg_order_value": rng.gamma(2, 40, n),
            "total_discount_amount": rng.gamma(1, 20, n),
            "avg_basket_size": rng.gamma(2, 2, n),
            "tenure_days": tenure,
            "revenue_per_order": rng.gamma(2, 30, n),
            "online_order_share": rng.random(n),
            "store_order_share": rng.random(n),
            "total_units": rng.poisson(12, n),
            "avg_item_discount_pct": rng.random(n),
            "avg_item_margin": rng.random(n),
            "total_returns": rng.poisson(1, n),
            "total_refund_amount": rng.gamma(1, 10, n),
            "avg_days_to_return": rng.gamma(2, 3, n),
            "return_rate_per_unit": rng.random(n),
            "total_sessions": sessions,
            "avg_session_minutes": rng.gamma(2, 3, n),
            "avg_pages_viewed": rng.gamma(2, 4, n),
            "sessions_add_to_cart": add_cart,
            "sessions_with_purchase": with_purchase,
            "campaigns_targeted": rng.poisson(2, n),
            "campaigns_treatment": rng.poisson(1, n),
            "campaigns_converted_30d": rng.poisson(1, n),
            "campaign_revenue_30d": rng.gamma(1, 15, n),
            "loyalty_tier": rng.choice(["none", "silver", "gold", None], n),
            "spend_rank_in_region": rng.random(n),
            LABEL: churn,
        }
    )
    return df


def main(smoke: bool = False) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    if smoke:
        mart = _make_smoke_frame()
        print(f"[smoke] synthetic rows: {len(mart):,}")
        mature = mart
    else:
        from src.data.mart_loaders import load_mart

        mart = load_mart("mart_customer_features", processed_dir=PROCESSED_DIR)
        mart = filter_active_population(mart)
        mature = filter_mature_cohort(mart, maturity_days=MATURITY_DAYS)
        print(f"active={len(mart):,}  mature={len(mature):,}")

    enriched = build_features(mature)
    fs = feature_set()  # customer_segment_seed already excluded
    assert not any(c.endswith("_seed") for c in fs.feature_columns), "seed leaked into features"

    train_df, test_df = time_ordered_split(enriched, test_size=TEST_SIZE)
    X_train, y_train = prepare_xy(train_df, fs=fs)
    X_test, y_test = prepare_xy(test_df, fs=fs)
    y_test_arr = y_test.to_numpy()

    numeric = [c for c in fs.feature_columns if c not in fs.categorical_columns]
    categorical = list(fs.categorical_columns)

    # ── Candidate: LightGBM V2 (native categorical + null handling) ─────────
    from src.models.train_churn import train_churn_model

    val_cut = int(len(X_train) * 0.85)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lgbm = train_churn_model(
            X_train.iloc[:val_cut],
            y_train.iloc[:val_cut],
            X_train.iloc[val_cut:],
            y_train.iloc[val_cut:],
            fs=fs,
        )
    scores = {"lightgbm_v2": lgbm.predict_proba(X_test)}

    # ── Baselines on the identical split / features ─────────────────────────
    baselines = {
        "rf_balanced": RandomForestClassifier(
            n_estimators=RF_TREES, class_weight="balanced", random_state=SEED, n_jobs=-1
        ),
        "logreg_balanced": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=SEED
        ),
    }
    for name, est in baselines.items():
        pipe = _sklearn_pipeline(est, numeric, categorical)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X_train, y_train)
        scores[name] = pipe.predict_proba(X_test)[:, 1]

    # ── Metric table (all models, identical held-out rows) ──────────────────
    rows = []
    for name, s in scores.items():
        rows.append(
            {
                "model": name,
                "is_candidate": name == "lightgbm_v2",
                "roc_auc": float(roc_auc_score(y_test_arr, s)),
                "pr_auc": float(average_precision_score(y_test_arr, s)),
                "n_test": int(len(y_test_arr)),
                "churn_rate_test": float(y_test_arr.mean()),
                "split": f"time_ordered_80_20_mature_{MATURITY_DAYS}d",
                "feature_policy": "recency_excluded; customer_segment_seed excluded",
            }
        )
    metrics = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)

    # Best baseline is the reference for the paired lift.
    best_baseline = (
        metrics[~metrics["is_candidate"]].sort_values("roc_auc", ascending=False).iloc[0]["model"]
    )

    lift_rows = []
    for metric_name, metric_fn in [("roc_auc", roc_auc_score), ("pr_auc", average_precision_score)]:
        point, lo, hi, share = _paired_bootstrap_ci(
            y_test_arr, scores["lightgbm_v2"], scores[best_baseline], metric_fn
        )
        ci_excludes_zero = lo > 0 or hi < 0
        lift_rows.append(
            {
                "metric": metric_name,
                "candidate": "lightgbm_v2",
                "reference": best_baseline,
                "candidate_value": float(metric_fn(y_test_arr, scores["lightgbm_v2"])),
                "reference_value": float(metric_fn(y_test_arr, scores[best_baseline])),
                "lift": point,
                "ci95_low": lo,
                "ci95_high": hi,
                "bootstrap_share_lift_positive": share,
                "ci_excludes_zero": bool(ci_excludes_zero),
                "verdict": (
                    "adds value" if (ci_excludes_zero and point > 0) else "does not beat baseline"
                ),
            }
        )
    lift = pd.DataFrame(lift_rows)

    metrics_path = OUTPUTS_DIR / f"{PREFIX}_metrics.csv"
    lift_path = OUTPUTS_DIR / f"{PREFIX}_paired_bootstrap.csv"
    if not smoke:
        metrics.to_csv(metrics_path, index=False)
        lift.to_csv(lift_path, index=False)
        print(f"Wrote: {metrics_path}")
        print(f"Wrote: {lift_path}")

    print("\nMatched metrics (identical held-out rows):")
    print(metrics[["model", "roc_auc", "pr_auc"]].to_string(index=False))
    print(f"\nPaired lift, LightGBM V2 vs best baseline ({best_baseline}):")
    print(
        lift[["metric", "lift", "ci95_low", "ci95_high", "ci_excludes_zero", "verdict"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run on synthetic data, no marts")
    args = parser.parse_args()
    main(smoke=args.smoke)
