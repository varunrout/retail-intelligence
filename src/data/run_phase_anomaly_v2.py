"""Phase Anomaly V2 runner.

Produces V2 abuse-return detection artefacts that mirror the phase11 baseline
schema so the two can be compared side by side.  Implementation reflects the
18 decisions documented in ``analysis_notebooks/anomaly_analysis.ipynb``.

V2 Algorithm — Supervised LightGBM
====================================
  Phase11 used unsupervised anomaly detection (IQR rules, Isolation Forest,
  LOF) despite returns_hidden_labels.csv being available.  The labels expose
  1,352 abuse positives in 188,399 returns (0.72% prevalence).  A supervised
  LightGBM classifier on 12 features achieves:
    CV AP = 0.791 ± 0.017        (14× vs IF AP = 0.058)
    Test AP = 0.595 (time-ordered holdout, Finding 6.B)
    1% flag rate → precision=33%, recall=79%, F1=47%

  Phase11 IQR at 12% flag rate → precision=2.7%, F1=5.2%.

Outputs
-------
  {PREFIX}_metrics.json            scalar evaluation metrics
  {PREFIX}_feature_importance.csv  gain-based feature importance
  {PREFIX}_review_queue.csv        top-flagged returns for review
  {PREFIX}_model_comparison.csv    V2 vs phase11 method comparison
  {PREFIX}_vs_baseline.csv         metric-level comparison table
  {PREFIX}_pr_curve.png            precision-recall curve
  {PREFIX}_feature_importance.png  bar chart of feature importance
"""

from __future__ import annotations

import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import OUTPUTS_DIR, PROCESSED_DIR
from src.features.features_anomaly import build_feature_set
from src.models.train_anomaly import (
    DEFAULT_FLAG_RATE,
    evaluate,
    feature_importance_table,
    train_anomaly_model,
)

PREFIX = "phase_anomaly_v2"

PALETTE = {
    "primary": "#2E86AB",
    "secondary": "#E63946",
    "tertiary": "#2a9d8f",
    "neutral": "#6c757d",
    "highlight": "#f4a261",
}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_pr_curve(
    y_test: np.ndarray,
    scores: np.ndarray,
    metrics: dict,
    path,
) -> None:
    """Precision-Recall curve with operational flag-rate markers."""
    from sklearn.metrics import precision_recall_curve

    prec, rec, _ = precision_recall_curve(y_test, scores)
    ap = metrics["ap"]
    prevalence = float(y_test.mean())

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rec, prec, color=PALETTE["primary"], linewidth=2, label=f"V2 LightGBM  AP={ap:.3f}")
    ax.axhline(
        prevalence,
        color="gray",
        linestyle="--",
        linewidth=1,
        label=f"Baseline prevalence {prevalence * 100:.2f}%",
    )

    # Mark the configured flag rate threshold
    metrics["flag_rate_pct"] / 100.0
    ax.scatter(
        metrics["recall"],
        metrics["precision"],
        s=100,
        color=PALETTE["secondary"],
        zorder=6,
        label=f"{metrics['flag_rate_pct']:.0f}% flag rate  (prec={metrics['precision'] * 100:.1f}%)",
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Anomaly Detection V2 — Precision-Recall Curve\n(time-ordered test set)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_feature_importance(fi_df: pd.DataFrame, path) -> None:
    """Horizontal bar chart of top feature importances (gain)."""
    top = fi_df.head(12)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [
        PALETTE["secondary"] if f.startswith("is_") else PALETTE["primary"] for f in top["feature"]
    ]
    ax.barh(range(len(top)), top["importance"], color=colors, alpha=0.85)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"], fontsize=9)
    ax.set_xlabel("Feature importance (gain)")
    ax.set_title("Anomaly Detection V2 — Feature Importance\n(red = OHE categorical flag)")
    for i, v in enumerate(top["importance"]):
        ax.text(v + top["importance"].max() * 0.01, i, f"{v:,.0f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Vs-baseline comparison
# ---------------------------------------------------------------------------


def _build_vs_baseline(metrics: dict) -> pd.DataFrame:
    """Construct a metric-level V2 vs phase11 comparison table."""
    rows = [
        {
            "metric": "ap",
            "phase11_method": "Isolation Forest",
            "phase11_value": 0.0580,
            "v2_value": round(metrics["ap"], 4),
            "delta_v2_minus_baseline": round(metrics["ap"] - 0.0580, 4),
            "note": "Average precision on full return universe",
        },
        {
            "metric": "precision",
            "phase11_method": "IQR / Rules",
            "phase11_value": 0.0274,
            "v2_value": round(metrics["precision"], 4),
            "delta_v2_minus_baseline": round(metrics["precision"] - 0.0274, 4),
            "note": f"At {metrics['flag_rate_pct']:.0f}% flag rate (p11 at 12% flag rate)",
        },
        {
            "metric": "recall",
            "phase11_method": "IQR / Rules",
            "phase11_value": 0.4601,
            "v2_value": round(metrics["recall"], 4),
            "delta_v2_minus_baseline": round(metrics["recall"] - 0.4601, 4),
            "note": f"At {metrics['flag_rate_pct']:.0f}% flag rate (p11 at 12% flag rate)",
        },
        {
            "metric": "flag_rate_pct",
            "phase11_method": "IQR / Rules",
            "phase11_value": 12.04,
            "v2_value": round(metrics["flag_rate_pct"], 2),
            "delta_v2_minus_baseline": round(metrics["flag_rate_pct"] - 12.04, 2),
            "note": "Fraction of returns flagged for review (lower = more targeted)",
        },
        {
            "metric": "f1",
            "phase11_method": "Isolation Forest",
            "phase11_value": 0.1174,
            "v2_value": round(metrics["f1"], 4),
            "delta_v2_minus_baseline": round(metrics["f1"] - 0.1174, 4),
            "note": f"F1 at {metrics['flag_rate_pct']:.0f}% flag rate",
        },
    ]
    return pd.DataFrame(rows)


def _build_model_comparison(metrics: dict) -> pd.DataFrame:
    """Full method comparison table (all phase11 methods + V2)."""
    rows = [
        {
            "method": "IQR / Rules",
            "type": "unsupervised",
            "flag_rate_pct": 12.04,
            "precision": 0.0274,
            "recall": 0.4601,
            "f1": 0.0518,
            "ap": 0.0165,
        },
        {
            "method": "Isolation Forest",
            "type": "unsupervised",
            "flag_rate_pct": 1.00,
            "precision": 0.1008,
            "recall": 0.1405,
            "f1": 0.1174,
            "ap": 0.0580,
        },
        {
            "method": "LOF (n=20)",
            "type": "unsupervised",
            "flag_rate_pct": 1.00,
            "precision": 0.1680,
            "recall": 0.0621,
            "f1": 0.0907,
            "ap": 0.0799,
        },
        {
            "method": "LightGBM V2",
            "type": "supervised",
            "flag_rate_pct": round(metrics["flag_rate_pct"], 2),
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1": round(metrics["f1"], 4),
            "ap": round(metrics["ap"], 4),
        },
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────
    mart_path = PROCESSED_DIR / "mart_returns_risk.csv"
    labels_path = PROCESSED_DIR / "returns_hidden_labels.csv"

    df_all = pd.read_csv(mart_path, parse_dates=["order_date"])
    labels = pd.read_csv(labels_path)

    ret = df_all[df_all["return_flag"] == 1].copy()
    ret_labeled = ret.merge(labels, on="return_id", how="inner")
    ret_labeled["is_abuse"] = ret_labeled["abuse_flag_hidden_for_validation"].astype(int)

    print(f"All order items     : {len(df_all):>10,}")
    print(f"Returns (flag=1)    : {len(ret):>10,}")
    print(f"Returns w/ label    : {len(ret_labeled):>10,}")
    print(
        f"Abuse positives     : {ret_labeled['is_abuse'].sum():>10,}  "
        f"({ret_labeled['is_abuse'].mean() * 100:.2f}%)"
    )

    # ── Feature engineering & split ───────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fs = build_feature_set(ret_labeled)

    print(f"\nTime-ordered split at {fs.split_date}")
    print(f"  Train: {fs.n_train:,} rows  prevalence={fs.train_prevalence * 100:.2f}%")
    print(f"  Test : {fs.n_test:,} rows   prevalence={fs.test_prevalence * 100:.2f}%")
    print(f"  scale_pos_weight = {fs.scale_pos_weight:.1f}")

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"\nTraining LightGBM (scale_pos_weight={fs.scale_pos_weight:.1f})…")
    model = train_anomaly_model(
        fs.X_train,
        fs.y_train,
        scale_pos_weight=fs.scale_pos_weight,
        X_val=fs.X_test,
        y_val=fs.y_test,
    )
    print("Training complete.")

    # ── Evaluate ──────────────────────────────────────────────────────────
    scores = model.predict_proba(fs.X_test)[:, 1]
    metrics = evaluate(fs.y_test, scores, flag_rate=DEFAULT_FLAG_RATE)

    print(f"\nTest AP: {metrics['ap']:.4f}")
    print(f"At {metrics['flag_rate_pct']:.0f}% flag rate ({metrics['flagged']:,} flagged):")
    print(f"  Precision = {metrics['precision'] * 100:.1f}%")
    print(f"  Recall    = {metrics['recall'] * 100:.1f}%")
    print(f"  F1        = {metrics['f1'] * 100:.1f}%")

    # ── Feature importance ────────────────────────────────────────────────
    fi_df = feature_importance_table(model, fs.feature_names)

    # ── Review queue (top flagged test returns) ───────────────────────────
    threshold = metrics["threshold"]
    test_copy = fs.test_df.copy()
    test_copy["abuse_score"] = scores
    test_copy["is_flagged"] = (scores >= threshold).astype(int)
    review_queue = (
        test_copy[test_copy["is_flagged"] == 1][
            [
                "return_id",
                "customer_id",
                "order_date",
                "category",
                "item_net_price",
                "refund_amount",
                "return_reason",
                "discount_band",
                "return_risk_band",
                "abuse_score",
                "is_abuse",
            ]
        ]
        .sort_values("abuse_score", ascending=False)
        .reset_index(drop=True)
    )

    # ── Build comparison tables ───────────────────────────────────────────
    vs_baseline = _build_vs_baseline(metrics)
    model_comparison = _build_model_comparison(metrics)

    print("\nVs baseline:")
    print(
        vs_baseline[["metric", "phase11_value", "v2_value", "delta_v2_minus_baseline"]].to_string(
            index=False
        )
    )

    # ── Save outputs ──────────────────────────────────────────────────────
    pr_path = OUTPUTS_DIR / f"{PREFIX}_pr_curve.png"
    fi_path = OUTPUTS_DIR / f"{PREFIX}_feature_importance.png"

    _plot_pr_curve(fs.y_test, scores, metrics, pr_path)
    _plot_feature_importance(fi_df, fi_path)

    metrics_out = {
        **metrics,
        "split_date": fs.split_date,
        "n_train": fs.n_train,
        "n_test": fs.n_test,
        "train_prevalence": round(fs.train_prevalence, 6),
        "test_prevalence": round(fs.test_prevalence, 6),
        "scale_pos_weight": round(fs.scale_pos_weight, 2),
        "n_features": len(fs.feature_names),
        "feature_names": fs.feature_names,
    }

    paths = {
        "metrics": OUTPUTS_DIR / f"{PREFIX}_metrics.json",
        "feature_importance": OUTPUTS_DIR / f"{PREFIX}_feature_importance.csv",
        "review_queue": OUTPUTS_DIR / f"{PREFIX}_review_queue.csv",
        "model_comparison": OUTPUTS_DIR / f"{PREFIX}_model_comparison.csv",
        "vs_baseline": OUTPUTS_DIR / f"{PREFIX}_vs_baseline.csv",
        "pr_curve": pr_path,
        "fi_plot": fi_path,
    }

    with open(paths["metrics"], "w") as f:
        json.dump(metrics_out, f, indent=2)

    fi_df.to_csv(paths["feature_importance"], index=False)
    review_queue.to_csv(paths["review_queue"], index=False)
    model_comparison.to_csv(paths["model_comparison"], index=False)
    vs_baseline.to_csv(paths["vs_baseline"], index=False)

    print("\nOutputs written:")
    for key, path in paths.items():
        print(f"  {key:<20} → {path.name}")


if __name__ == "__main__":
    main()
