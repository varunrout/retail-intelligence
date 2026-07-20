"""Phase Forecast V2 runner.

Produces V2 demand-forecast artefacts that mirror the phase8 baseline schema so the
two can be compared side by side. Implementation reflects the 14 decisions
documented in ``analysis_notebooks/forecast_analysis.ipynb``.
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import OUTPUTS_DIR, PROCESSED_DIR
from src.data.mart_loaders import load_mart
from src.features.features_forecast import (
    LABEL,
    SERIES_KEYS,
    SPLIT_WEEK,
    TIME_KEY,
    attach_series_means,
    build_lag_features,
    compute_series_means,
    feature_set,
    filter_short_series,
    prepare_xy,
    week_time_split,
)
from src.models.train_forecast import (
    evaluate_forecast,
    feature_importance_table,
    train_forecast,
)

PREFIX = "phase_forecast_v2"

# Validation tail fraction carved off the training set for early stopping
VALID_FRAC = 0.15


# ── Plots ──────────────────────────────────────────────────────────────────


def _plot_model_comparison(summary: pd.DataFrame, path) -> None:
    """Horizontal bar chart comparing V2 MAE against baseline models."""
    baseline_path = OUTPUTS_DIR / "phase8_forecast_model_comparison.csv"
    rows = [{"model": summary["model"].iloc[0], "mae": float(summary["mae"].iloc[0])}]
    if baseline_path.exists():
        base = pd.read_csv(baseline_path)
        for _, r in base.iterrows():
            rows.append({"model": str(r["model"]), "mae": float(r["mae"])})

    df = pd.DataFrame(rows).drop_duplicates("model").sort_values("mae", ascending=False)

    fig, ax = plt.subplots(figsize=(9, max(3, len(df) * 0.65)))
    colors = ["#2E86AB" if "v2" in m else "#A8DADC" for m in df["model"]]
    y = range(len(df))
    ax.barh(list(y), df["mae"], color=colors)
    ax.set_yticks(list(y))
    ax.set_yticklabels(df["model"].tolist())
    ax.set_xlabel("MAE (units)")
    ax.set_title("Forecast V2 vs Baseline — MAE comparison")
    for yi, val in zip(y, df["mae"], strict=False):
        ax.text(val + 0.005, yi, f"{val:.3f}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, path, *, bias: float) -> None:
    """Residual histogram to confirm bias reduction vs RF baseline (Finding 3.B)."""
    residuals = y_pred - y_true
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(residuals, bins=60, color="#2E86AB", alpha=0.8, edgecolor="white")
    ax.axvline(0, color="black", linewidth=1)
    ax.axvline(
        bias, color="#E63946", linewidth=1.5, linestyle="--", label=f"Mean bias = {bias:+.3f}"
    )
    ax.set_xlabel("Residual (pred − actual)")
    ax.set_ylabel("Count")
    ax.set_title("Forecast V2 — Residual distribution (bias check)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_per_category(per_cat: pd.DataFrame, path) -> None:
    """Per-category MAE bar chart."""
    if per_cat.empty:
        return
    df = per_cat.sort_values("mae", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(3, len(df) * 0.65)))
    y = range(len(df))
    ax.barh(list(y), df["mae"], color="#2E86AB", alpha=0.85)
    ax.set_yticks(list(y))
    ax.set_yticklabels(df["category"].tolist())
    ax.set_xlabel("MAE (units)")
    ax.set_title("Forecast V2 — Per-category MAE")
    for yi, val in zip(y, df["mae"], strict=False):
        ax.text(val + 0.005, yi, f"{val:.3f}", va="center", fontsize=8.5)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_weekly_aggregate(test_df: pd.DataFrame, preds: np.ndarray, path) -> None:
    """Weekly aggregate actual vs predicted (mirrors phase8 chart for comparison)."""
    df = test_df.reset_index(drop=True).copy()
    df["_pred"] = preds
    wa = (
        df.groupby(TIME_KEY)
        .agg(actual=(LABEL, "sum"), predicted=("_pred", "sum"))
        .reset_index()
        .sort_values(TIME_KEY)
    )

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(wa[TIME_KEY], wa["actual"], color="#2E86AB", linewidth=2, label="Actual")
    ax.plot(
        wa[TIME_KEY],
        wa["predicted"],
        color="#E63946",
        linewidth=2,
        linestyle="--",
        label="LightGBM V2",
    )
    ax.set_xlabel("Week")
    ax.set_ylabel("Total units sold")
    ax.set_title("Forecast V2 — Weekly aggregate: actual vs predicted")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ── Baseline comparison ────────────────────────────────────────────────────


def _baseline_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    """Compare V2 metrics against the phase8 RF baseline."""
    baseline_path = OUTPUTS_DIR / "phase8_forecast_model_comparison.csv"
    if not baseline_path.exists():
        return pd.DataFrame()

    base = pd.read_csv(baseline_path)
    # Best baseline = smallest MAE
    rf_row = base[base["model"] == "rf_feature_model_quality"]
    if rf_row.empty:
        rf_row = base.sort_values("mae").iloc[[0]]

    metric_cols = ["mae", "rmse", "smape"]
    note = (
        "baseline=RF (phase8, 14 features, no leakage fix); "
        "V2=LightGBM Tweedie, lagged rolling means, series baselines, "
        "cyclical calendar, inventory flags, short-series filter"
    )
    rows = []
    for col in metric_cols:
        v2_val = float(summary[col].iloc[0]) if col in summary.columns else np.nan
        base_val = float(rf_row[col].iloc[0]) if col in rf_row.columns else np.nan
        rows.append(
            {
                "metric": col,
                "baseline_model": "rf_feature_model_quality",
                "baseline_value": base_val,
                "v2_value": v2_val,
                "delta_v2_minus_baseline": v2_val - base_val
                if not (np.isnan(v2_val) or np.isnan(base_val))
                else np.nan,
                "pct_improvement": (base_val - v2_val) / base_val * 100
                if not (np.isnan(v2_val) or np.isnan(base_val)) and base_val > 0
                else np.nan,
                "note": note,
            }
        )
    return pd.DataFrame(rows)


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load mart ──────────────────────────────────────────────────────────
    print("Loading mart_product_demand …")
    mart = load_mart("mart_product_demand", processed_dir=PROCESSED_DIR)
    print(f"  Raw mart: {len(mart):,} rows × {mart.shape[1]} cols")

    # ── Build lag + rolling + calendar features on full dataset ───────────
    print("Building lag / rolling / calendar features …")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mart_feats = build_lag_features(mart)
    print(f"  After feature engineering: {mart_feats.shape[1]} cols")

    # ── Train / test split (week-ordered 80/20) ────────────────────────────
    train_raw, test_df = week_time_split(mart_feats, split_week=SPLIT_WEEK)
    print(
        f"  Split @ {SPLIT_WEEK}: train={len(train_raw):,} rows  "
        f"test={len(test_df):,} rows  "
        f"test_weeks={test_df[TIME_KEY].nunique()}"
    )

    # ── Filter short series from training set (Finding 1.A) ────────────────
    train_filtered = filter_short_series(train_raw)
    n_dropped = len(train_raw) - len(train_filtered)
    print(
        f"  Short-series filter: dropped {n_dropped:,} train rows "
        f"({n_dropped / len(train_raw) * 100:.1f}%)"
    )

    # ── Series-level demand means (train-set only, Finding 6.A) ───────────
    print("Computing product / store mean demand (train only) …")
    product_means, store_means = compute_series_means(train_filtered)
    train_feats = attach_series_means(train_filtered, product_means, store_means)
    test_feats = attach_series_means(test_df, product_means, store_means)

    # ── Carve validation tail for LightGBM early stopping ─────────────────
    train_sorted = train_feats.sort_values(TIME_KEY, kind="mergesort").reset_index(drop=True)
    val_cut = int(len(train_sorted) * (1 - VALID_FRAC))
    tr_df = train_sorted.iloc[:val_cut].copy()
    va_df = train_sorted.iloc[val_cut:].copy()
    print(
        f"  Train={len(tr_df):,}  valid={len(va_df):,}  test={len(test_feats):,}"
        f"  mean_units_train={tr_df[LABEL].mean():.3f}"
        f"  mean_units_test={test_feats[LABEL].mean():.3f}"
    )

    # ── Train LightGBM Tweedie model ───────────────────────────────────────
    print("Training LightGBM Tweedie model …")
    fs = feature_set()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = train_forecast(tr_df, va_df, fs=fs)
    print(f"  Best iteration: {model.best_iteration}")

    # ── Score test set ─────────────────────────────────────────────────────
    X_test, y_test = prepare_xy(test_feats, fs=fs)
    test_reset = test_feats.reset_index(drop=True)
    preds = model.predict(X_test)

    # ── Evaluation ─────────────────────────────────────────────────────────
    summary, per_cat = evaluate_forecast(test_reset, preds)
    mae = float(summary["mae"].iloc[0])
    rmse = float(summary["rmse"].iloc[0])
    smape = float(summary["smape"].iloc[0])
    bias = float(summary["mean_bias"].iloc[0])

    print(f"\n  MAE  = {mae:.4f}  (baseline RF = 1.2512)")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  sMAPE= {smape:.4f}")
    print(f"  Bias = {bias:+.4f}  (baseline RF ≈ −0.44)")

    # ── Feature importance ─────────────────────────────────────────────────
    fi = feature_importance_table(model, top_n=30)

    # ── Baseline comparison ────────────────────────────────────────────────
    comparison = _baseline_comparison(summary)

    # ── Scored predictions table (full test set) ───────────────────────────
    scored = test_reset[[TIME_KEY, *SERIES_KEYS, LABEL]].copy()
    if "category" in test_reset.columns:
        scored["category"] = test_reset["category"].values
    scored["predicted_units"] = preds
    scored["residual"] = preds - test_reset[LABEL].values

    # ── LightGBM hyperparameters record ───────────────────────────────────
    params_df = pd.DataFrame(
        [{"param": k, "value": str(v)} for k, v in model.params.items()]
        + [{"param": "best_iteration", "value": str(model.best_iteration)}]
    )

    # ── Write outputs ──────────────────────────────────────────────────────
    paths = {
        "model_comparison": OUTPUTS_DIR / f"{PREFIX}_model_comparison.csv",
        "per_category": OUTPUTS_DIR / f"{PREFIX}_per_category.csv",
        "feature_importance": OUTPUTS_DIR / f"{PREFIX}_feature_importance.csv",
        "vs_baseline": OUTPUTS_DIR / f"{PREFIX}_vs_baseline.csv",
        "scored_predictions": OUTPUTS_DIR / f"{PREFIX}_scored_predictions.csv",
        "params": OUTPUTS_DIR / f"{PREFIX}_params.csv",
        "mae_comparison_chart": OUTPUTS_DIR / f"{PREFIX}_mae_comparison.png",
        "residual_chart": OUTPUTS_DIR / f"{PREFIX}_residuals.png",
        "per_category_chart": OUTPUTS_DIR / f"{PREFIX}_per_category.png",
        "weekly_aggregate_chart": OUTPUTS_DIR / f"{PREFIX}_weekly_aggregate.png",
    }

    summary.to_csv(paths["model_comparison"], index=False)
    per_cat.to_csv(paths["per_category"], index=False)
    fi.to_csv(paths["feature_importance"], index=False)
    params_df.to_csv(paths["params"], index=False)
    scored.to_csv(paths["scored_predictions"], index=False)
    if not comparison.empty:
        comparison.to_csv(paths["vs_baseline"], index=False)

    _plot_model_comparison(summary, paths["mae_comparison_chart"])
    _plot_residuals(y_test, preds, paths["residual_chart"], bias=bias)
    _plot_per_category(per_cat, paths["per_category_chart"])
    _plot_weekly_aggregate(test_reset, preds, paths["weekly_aggregate_chart"])

    print("\nOutputs written:")
    for _key, p in paths.items():
        print(f"  {p.name}")

    if not comparison.empty:
        print("\nV2 vs Baseline (RF phase8):")
        for _, row in comparison.iterrows():
            delta = row["delta_v2_minus_baseline"]
            pct = row["pct_improvement"]
            sign = "+" if delta >= 0 else ""
            print(
                f"  {row['metric']:8s}  baseline={row['baseline_value']:.4f}  "
                f"v2={row['v2_value']:.4f}  delta={sign}{delta:.4f}  "
                f"improvement={pct:+.1f}%"
            )


if __name__ == "__main__":
    main()
