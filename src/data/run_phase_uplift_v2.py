"""Phase Uplift V2 runner.

Produces V2 uplift artefacts that mirror the phase7 baseline schema so the
two can be compared side by side. Implementation reflects the 14 decisions
documented in ``analysis_notebooks/uplift_analysis.ipynb`` §8.
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
from src.features.features_uplift import (
    CAMPAIGN_KEY,
    ID_KEY,
    LABEL,
    TIME_KEY,
    TREATMENT_KEY,
    build_features,
    enrich_with_customer_mart,
    feature_set,
    prepare_xy,
    time_ordered_split,
)
from src.models.train_uplift import (
    STAGE1_PARAMS,
    STAGE2_PARAMS,
    evaluate_uplift,
    feature_importance_table,
    train_xlearner,
)

PREFIX = "phase_uplift_v2"
TEST_SIZE = 0.20


# ── Plots ──────────────────────────────────────────────────────────────────


def _plot_decile_uplift(decile_df: pd.DataFrame, path, *, ate_overall: float) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = decile_df["decile"].values
    uplift_pp = decile_df["observed_uplift"].values * 100
    ax.bar(x, uplift_pp, color="#2E86AB", alpha=0.85)
    ax.axhline(
        ate_overall * 100,
        color="grey",
        linestyle="--",
        linewidth=1.2,
        label=f"Overall ATE = {ate_overall * 100:.2f}pp",
    )
    ax.set_xlabel("Decile (1 = highest predicted uplift)")
    ax.set_ylabel("Observed uplift (pp)")
    ax.set_title("Uplift V2 — Observed uplift by ITE decile")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_qini_comparison(decile_df_v2: pd.DataFrame, path) -> None:
    """Qini curve for V2; overlay phase7 baseline if available."""
    fig, ax = plt.subplots(figsize=(7, 5))

    def _qini_curve(decile_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        # support both 'decile' (V2) and 'uplift_decile' (baseline) column names
        dec_col = "decile" if "decile" in decile_df.columns else "uplift_decile"
        decile_df = decile_df.sort_values(dec_col)
        cumulative_n = []
        cumulative_incr = []
        running = 0.0
        running_n = 0
        for _, row in decile_df.iterrows():
            n_t = row["n_treatment"]
            n_c = row["n_control"]
            obs = row["observed_uplift"]
            if not np.isnan(obs) and n_t:
                running += obs * n_t
            running_n += n_t + n_c
            cumulative_n.append(running_n)
            cumulative_incr.append(running)
        return np.array(cumulative_n), np.array(cumulative_incr)

    n_v2, incr_v2 = _qini_curve(decile_df_v2)
    ax.plot(n_v2, incr_v2, color="#2E86AB", linewidth=2, label="X-learner V2")

    # Overlay phase7 baseline decile table if it exists
    baseline_path = OUTPUTS_DIR / "phase7_uplift_decile_summary.csv"
    if baseline_path.exists():
        base_dec = pd.read_csv(baseline_path)
        # Use best model by qini if multiple
        if "model" in base_dec.columns:
            models = base_dec["model"].unique()
            cmap = plt.get_cmap("tab10")
            for i, m in enumerate(models):
                dec_col = "decile" if "decile" in base_dec.columns else "uplift_decile"
                sub = base_dec[base_dec["model"] == m].sort_values(dec_col)
                n_b, incr_b = _qini_curve(sub)
                ax.plot(
                    n_b,
                    incr_b,
                    linestyle="--",
                    linewidth=1.2,
                    color=cmap(i + 1),
                    label=f"Baseline: {m}",
                )
        else:
            n_b, incr_b = _qini_curve(base_dec.sort_values("decile"))
            ax.plot(n_b, incr_b, linestyle="--", linewidth=1.2, color="grey", label="Baseline")

    ax.plot(
        [0, n_v2[-1]], [0, incr_v2[-1]], color="black", linestyle=":", linewidth=1, label="Random"
    )
    ax.set_xlabel("Customers targeted (cumulative)")
    ax.set_ylabel("Cumulative incremental responses")
    ax.set_title("Uplift V2 — Qini curve (V2 vs baseline)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_per_campaign_ate(summary: pd.DataFrame, path) -> None:
    """Horizontal bar: per-campaign observed ATE on test set."""
    if "campaign_id" not in summary.columns:
        return
    fig, ax = plt.subplots(figsize=(8, max(4, len(summary) * 0.5)))
    y = range(len(summary))
    ax.barh(list(y), summary["ate_test"] * 100, color="#2E86AB", alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_yticks(list(y))
    ax.set_yticklabels(summary["campaign_id"].tolist())
    ax.set_xlabel("Observed ATE on test rows (pp)")
    ax.set_title("Uplift V2 — Per-campaign ATE")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ── Baseline comparison ────────────────────────────────────────────────────


def _baseline_comparison(v2_summary: pd.DataFrame) -> pd.DataFrame:
    baseline_path = OUTPUTS_DIR / "phase7_uplift_model_comparison.csv"
    if not baseline_path.exists():
        return pd.DataFrame()

    base = pd.read_csv(baseline_path)
    metric_cols = [
        "overall_ate_test",
        "top3_decile_observed_uplift",
        "top5_decile_observed_uplift",
        "qini_like_area",
    ]
    best = base.sort_values("qini_like_area", ascending=False).iloc[0]
    note = (
        "baseline=T-learner logreg/RF, 14 raw campaign features; "
        "V2=X-learner LightGBM, enriched features, per-campaign stratification"
    )
    rows = []
    for col in metric_cols:
        v2_val = float(v2_summary[col].iloc[0]) if col in v2_summary.columns else np.nan
        base_val = float(best[col]) if col in best.index else np.nan
        rows.append(
            {
                "metric": col,
                "baseline_best_model": str(best.get("model", "unknown")),
                "baseline_value": base_val,
                "v2_value": v2_val,
                "delta_v2_minus_baseline": v2_val - base_val
                if not (np.isnan(v2_val) or np.isnan(base_val))
                else np.nan,
                "note": note,
            }
        )
    return pd.DataFrame(rows)


# ── Per-campaign ATE table ─────────────────────────────────────────────────


def _per_campaign_ate(test_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cmp, grp in test_df.groupby(CAMPAIGN_KEY):
        t = grp[TREATMENT_KEY].astype(int)
        y = grp[LABEL].astype(int)
        n_t = int((t == 1).sum())
        n_c = int((t == 0).sum())
        r_t = float(y[t == 1].mean()) if n_t else np.nan
        r_c = float(y[t == 0].mean()) if n_c else np.nan
        ate = (r_t - r_c) if (n_t and n_c) else np.nan
        rows.append(
            {
                "campaign_id": cmp,
                "n_treatment_test": n_t,
                "n_control_test": n_c,
                "response_rate_treatment": r_t,
                "response_rate_control": r_c,
                "ate_test": ate,
            }
        )
    return pd.DataFrame(rows).sort_values("ate_test", ascending=False)


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load marts ─────────────────────────────────────────────────────────
    print("Loading marts …")
    mart = load_mart("mart_campaign_response", processed_dir=PROCESSED_DIR)
    cust_mart = load_mart("mart_customer_features", processed_dir=PROCESSED_DIR)
    n_raw = len(mart)

    # ── Enrich + build features ────────────────────────────────────────────
    print("Enriching and building features …")
    enriched = enrich_with_customer_mart(mart, cust_mart)
    enriched = build_features(enriched)
    n_enriched = len(enriched)
    print(f"Rows: raw={n_raw:,}  after enrichment={n_enriched:,}")

    # ── Time-ordered split ─────────────────────────────────────────────────
    train_df, test_df = time_ordered_split(enriched, test_size=TEST_SIZE)

    # Carve 15% validation tail off train for early-stopping
    val_cut = int(len(train_df) * 0.85)
    sorted_train = train_df.sort_values(TIME_KEY, kind="mergesort").reset_index(drop=True)
    tr_df = sorted_train.iloc[:val_cut].copy()
    va_df = sorted_train.iloc[val_cut:].copy()

    print(
        f"Splits: train={len(tr_df):,}  valid={len(va_df):,}  test={len(test_df):,}"
        f"  response_train={tr_df[LABEL].mean():.4f}"
        f"  response_test={test_df[LABEL].mean():.4f}"
    )

    # ── Train X-learner ────────────────────────────────────────────────────
    print("Training X-learner …")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = train_xlearner(tr_df, va_df, fs=feature_set())

    # ── Score test set ─────────────────────────────────────────────────────
    fs = feature_set()
    X_test, y_test, t_test = prepare_xy(test_df, fs=fs)
    ite_scores = model.predict_ite(X_test, test_df[CAMPAIGN_KEY].astype(str))

    # ── Evaluation ─────────────────────────────────────────────────────────
    summary, decile_df = evaluate_uplift(test_df, ite_scores, label="xlearner_v2")
    ate_overall = float(summary["overall_ate_test"].iloc[0])

    print(f"Overall ATE (test): {ate_overall * 100:.3f}pp")
    print(
        f"Top-1 decile observed uplift: {float(summary['top1_decile_observed_uplift'].iloc[0]) * 100:.3f}pp"
    )
    print(
        f"Top-5 decile observed uplift: {float(summary['top5_decile_observed_uplift'].iloc[0]) * 100:.3f}pp"
    )
    print(f"Qini-like area: {float(summary['qini_like_area'].iloc[0]):.1f}")
    print(f"Spearman rank corr: {float(summary['spearman_rank_corr'].iloc[0]):.4f}")

    # ── Per-campaign ATE ───────────────────────────────────────────────────
    per_cmp = _per_campaign_ate(test_df)

    # ── Feature importance ─────────────────────────────────────────────────
    fi = feature_importance_table(model, top_n=30)

    # ── Top-500 scored sample ──────────────────────────────────────────────
    scored = test_df[[ID_KEY, CAMPAIGN_KEY, TIME_KEY, LABEL, TREATMENT_KEY]].copy()
    scored["ite_score"] = ite_scores
    scored = scored.sort_values("ite_score", ascending=False).head(500).reset_index(drop=True)
    scored.insert(0, "rank", scored.index + 1)

    # ── Baseline comparison ────────────────────────────────────────────────
    comparison = _baseline_comparison(summary)

    # ── Params table ──────────────────────────────────────────────────────
    s1_rows = [{"stage": "stage1", "param": k, "value": v} for k, v in STAGE1_PARAMS.items()]
    s2_rows = [{"stage": "stage2", "param": k, "value": v} for k, v in STAGE2_PARAMS.items()]
    iter_rows = [
        {"stage": "stage2_best_iter", "param": k, "value": v}
        for k, v in model.stage2_best_iters.items()
    ]
    params_df = pd.DataFrame(s1_rows + s2_rows + iter_rows)

    # ── Write outputs ──────────────────────────────────────────────────────
    paths = {
        "model_comparison": OUTPUTS_DIR / f"{PREFIX}_model_comparison.csv",
        "decile_summary": OUTPUTS_DIR / f"{PREFIX}_decile_summary.csv",
        "scored_sample": OUTPUTS_DIR / f"{PREFIX}_scored_sample_top500.csv",
        "per_campaign_ate": OUTPUTS_DIR / f"{PREFIX}_per_campaign_ate.csv",
        "feature_importance": OUTPUTS_DIR / f"{PREFIX}_feature_importance.csv",
        "vs_baseline": OUTPUTS_DIR / f"{PREFIX}_vs_baseline.csv",
        "params": OUTPUTS_DIR / f"{PREFIX}_params.csv",
        "decile_chart": OUTPUTS_DIR / f"{PREFIX}_decile_uplift.png",
        "qini_chart": OUTPUTS_DIR / f"{PREFIX}_qini_curve.png",
        "campaign_ate_chart": OUTPUTS_DIR / f"{PREFIX}_per_campaign_ate.png",
    }

    summary.to_csv(paths["model_comparison"], index=False)
    decile_df.to_csv(paths["decile_summary"], index=False)
    scored.to_csv(paths["scored_sample"], index=False)
    per_cmp.to_csv(paths["per_campaign_ate"], index=False)
    fi.to_csv(paths["feature_importance"], index=False)
    params_df.to_csv(paths["params"], index=False)
    if not comparison.empty:
        comparison.to_csv(paths["vs_baseline"], index=False)

    _plot_decile_uplift(decile_df, paths["decile_chart"], ate_overall=ate_overall)
    _plot_qini_comparison(decile_df, paths["qini_chart"])
    _plot_per_campaign_ate(per_cmp, paths["campaign_ate_chart"])

    print("\nOutputs written:")
    for _key, p in paths.items():
        print(f"  {p.name}")

    if not comparison.empty:
        print("\nV2 vs Baseline:")
        for _, row in comparison.iterrows():
            delta = row["delta_v2_minus_baseline"]
            sign = "+" if delta >= 0 else ""
            print(
                f"  {row['metric']:40s}  baseline={row['baseline_value']:.4f}  "
                f"v2={row['v2_value']:.4f}  delta={sign}{delta:.4f}"
            )


if __name__ == "__main__":
    main()
