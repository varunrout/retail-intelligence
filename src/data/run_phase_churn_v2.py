"""Phase Churn V2 runner.

Produces V2 churn artefacts that mirror the phase6 baseline schema so the
two can be compared side by side. Implementation reflects the 11
decisions documented in ``analysis_notebooks/churn_analysis.ipynb`` §8.
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from src.config import OUTPUTS_DIR, PROCESSED_DIR
from src.data.mart_loaders import load_mart
from src.features.features_churn import (
    LABEL,
    build_features,
    feature_set,
    filter_active_population,
    filter_mature_cohort,
    prepare_xy,
    time_ordered_split,
)
from src.models.train_churn import (
    DEFAULT_LGBM_PARAMS,
    evaluate,
    feature_importance_table,
    threshold_diagnostics,
    threshold_selection,
    train_churn_model,
)

PREFIX = "phase_churn_v2"
MATURITY_DAYS = 90
TEST_SIZE = 0.20


def _plot_pr_curve(y_true: pd.Series, y_score: np.ndarray, path) -> None:
    p, r, _ = precision_recall_curve(y_true, y_score)
    base = float(y_true.mean())
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(r, p, color="#2E86AB", linewidth=2, label="lightgbm_v2")
    ax.axhline(base, color="grey", linestyle="--", linewidth=1, label=f"baseline = {base:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Churn V2 — Precision/Recall")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_roc_curve(y_true: pd.Series, y_score: np.ndarray, path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(fpr, tpr, color="#2E86AB", linewidth=2, label="lightgbm_v2")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Churn V2 — ROC")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _baseline_comparison(v2_eval: pd.DataFrame) -> pd.DataFrame:
    """Join V2 metrics next to the best phase6 baseline row (by ROC AUC)."""
    baseline_path = OUTPUTS_DIR / "phase6_churn_model_comparison.csv"
    if not baseline_path.exists():
        return pd.DataFrame()

    base = pd.read_csv(baseline_path)
    metric_cols = [
        "roc_auc",
        "pr_auc",
        "log_loss",
        "brier_score",
        "accuracy_at_0_50",
        "balanced_accuracy_at_0_50",
        "mcc_at_0_50",
    ]
    best = base.sort_values("roc_auc", ascending=False).iloc[0]
    note = (
        "NOT like-for-like: baseline=stratified random split incl. recency features; "
        "V2=time-ordered split, 90d maturity, recency excluded. Confounds algorithm, "
        "split and features. See analysis/churn_incremental_lift.py + "
        "docs/churn_incremental_lift_and_reconciliation.md for the matched comparison "
        "(logistic regression beats LightGBM V2 on identical rows)."
    )

    rows = []
    for col in metric_cols:
        v2_val = float(v2_eval[col].iloc[0])
        base_val = float(best[col])
        delta = v2_val - base_val
        rows.append(
            {
                "metric": col,
                "baseline_best_model": str(best["model"]),
                "baseline_value": base_val,
                "v2_value": v2_val,
                "delta_v2_minus_baseline": delta,
                "note": note,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load + filter ──────────────────────────────────────────────────────
    mart = load_mart("mart_customer_features", processed_dir=PROCESSED_DIR)
    n_full = len(mart)
    mart_active = filter_active_population(mart)
    n_active = len(mart_active)
    mart_mature = filter_mature_cohort(mart_active, maturity_days=MATURITY_DAYS)
    n_mature = len(mart_mature)

    print(f"Population: full={n_full:,}  active={n_active:,}  mature={n_mature:,}")

    # ── Feature engineering + split ────────────────────────────────────────
    enriched = build_features(mart_mature)
    fs = feature_set()
    train_df, test_df = time_ordered_split(enriched, test_size=TEST_SIZE)
    X_train, y_train = prepare_xy(train_df, fs=fs)
    X_test, y_test = prepare_xy(test_df, fs=fs)

    # Carve a small validation tail off the train block for early stopping
    val_cut = int(len(X_train) * 0.85)
    X_tr, y_tr = X_train.iloc[:val_cut], y_train.iloc[:val_cut]
    X_val, y_val = X_train.iloc[val_cut:], y_train.iloc[val_cut:]

    print(
        f"Splits: train={len(X_tr):,}  valid={len(X_val):,}  test={len(X_test):,}  "
        f"churn_train={y_tr.mean():.4f}  churn_test={y_test.mean():.4f}"
    )

    # ── Train ──────────────────────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = train_churn_model(X_tr, y_tr, X_val, y_val, fs=fs)

    print(f"Best iteration: {model.best_iteration}")

    # ── Score test set ─────────────────────────────────────────────────────
    y_score = model.predict_proba(X_test)

    # ── Evaluation row aligned with phase6 schema ──────────────────────────
    eval_row = evaluate(y_test, y_score, label="v2")
    eval_row.insert(1, "split_strategy", f"time_ordered_80_20_mature_{MATURITY_DAYS}d")
    eval_row.insert(2, "approved_feature_count", len(fs.feature_columns))
    eval_row.insert(3, "train_rows", int(len(X_tr)))
    eval_row.insert(4, "valid_rows", int(len(X_val)))
    eval_row.insert(5, "test_rows", int(len(X_test)))
    eval_row.insert(6, "churn_rate_train", float(y_tr.mean()))

    # ── Threshold artefacts ────────────────────────────────────────────────
    diag = threshold_diagnostics(y_test, y_score)
    diag.insert(0, "model", "lightgbm_v2")
    sel = threshold_selection(diag)
    sel.insert(0, "model", "lightgbm_v2")

    # ── Feature importance ─────────────────────────────────────────────────
    fi = feature_importance_table(model, top_n=30)

    # ── Top-500 scored sample ──────────────────────────────────────────────
    scored = test_df[["customer_id", "signup_date", LABEL]].copy()
    scored["churn_score"] = y_score
    scored = scored.sort_values("churn_score", ascending=False).head(500).reset_index(drop=True)
    scored.insert(0, "rank", scored.index + 1)

    # ── Baseline comparison ────────────────────────────────────────────────
    comparison = _baseline_comparison(eval_row)

    # ── Write outputs ──────────────────────────────────────────────────────
    paths: dict = {
        "model_comparison": OUTPUTS_DIR / f"{PREFIX}_model_comparison.csv",
        "threshold_diagnostics": OUTPUTS_DIR / f"{PREFIX}_threshold_diagnostics.csv",
        "threshold_selection": OUTPUTS_DIR / f"{PREFIX}_threshold_selection.csv",
        "feature_importance": OUTPUTS_DIR / f"{PREFIX}_feature_importance_top30.csv",
        "scored_sample": OUTPUTS_DIR / f"{PREFIX}_scored_sample_top500.csv",
        "vs_baseline": OUTPUTS_DIR / f"{PREFIX}_vs_baseline.csv",
        "pr_curve": OUTPUTS_DIR / f"{PREFIX}_pr_curve.png",
        "roc_curve": OUTPUTS_DIR / f"{PREFIX}_roc_curve.png",
        "params": OUTPUTS_DIR / f"{PREFIX}_params.csv",
    }

    eval_row.to_csv(paths["model_comparison"], index=False)
    diag.to_csv(paths["threshold_diagnostics"], index=False)
    sel.to_csv(paths["threshold_selection"], index=False)
    fi.to_csv(paths["feature_importance"], index=False)
    scored.to_csv(paths["scored_sample"], index=False)
    if not comparison.empty:
        comparison.to_csv(paths["vs_baseline"], index=False)
    pd.DataFrame(
        [{"param": k, "value": v} for k, v in DEFAULT_LGBM_PARAMS.items()]
        + [
            {"param": "best_iteration", "value": model.best_iteration},
            {"param": "maturity_days", "value": MATURITY_DAYS},
            {"param": "test_size", "value": TEST_SIZE},
        ]
    ).to_csv(paths["params"], index=False)

    _plot_pr_curve(y_test, y_score, paths["pr_curve"])
    _plot_roc_curve(y_test, y_score, paths["roc_curve"])

    for _label, path in paths.items():
        if not path.exists():
            continue
        print(f"Wrote: {path}")

    # ── Headline ───────────────────────────────────────────────────────────
    print("\nV2 headline metrics (test set):")
    print(
        eval_row[["roc_auc", "pr_auc", "log_loss", "brier_score", "mcc_at_0_50"]].to_string(
            index=False
        )
    )
    if not comparison.empty:
        print("\nV2 vs baseline:")
        print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
