"""Phase Segmentation V2 runner.

Produces V2 customer-segmentation artefacts that mirror the phase9 baseline
schema so the two can be compared side by side.  Implementation reflects the
14 decisions documented in ``analysis_notebooks/segmentation_analysis.ipynb``.

V2 Algorithm — Nested k-means
==============================
  Stage 1  Separate non-purchasers (11.1 % of customers) as a dedicated
           "Non-Purchasers" segment.  Phase9 silently imputed zeros and
           conflated non-buyers with churned actives into "Dormant 1"
           (Finding 1.A / 3.B / 6.B).

  Stage 2  K-means (k=3, n_init=20) on the buyer sub-population (44 432 rows)
           using 13 log + ratio features reduced by PCA-5 (Finding 5.C).
           k=3 buyers + 1 non-purchaser segment = 4 total segments.

  Silhouette on buyer PCA space ≈ 0.38 vs phase9 0.25 (Finding 3.A / 4.A).
  Expected churn spread ≥ 30 pp (Finding 3.C target ≥ 33 pp).
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
from src.data.mart_loaders import load_mart
from src.features.features_segmentation import engineer_features
from src.models.train_segmentation import (
    N_BUYER_CLUSTERS,
    cluster_profiles,
    segment_metrics,
    train_segmentation,
)

PREFIX = "phase_segmentation_v2"

PALETTE = {
    "Champions": "#2E86AB",
    "Mid-Tier Active": "#2a9d8f",
    "Dormant Buyers": "#f4a261",
    "Non-Purchasers": "#6c757d",
}
DEFAULT_COLOR = "#A8DADC"


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_segment_distribution(profiles: pd.DataFrame, path) -> None:
    """Horizontal bar chart of segment sizes."""
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = [PALETTE.get(name, DEFAULT_COLOR) for name in profiles["segment_name"]]
    y = range(len(profiles))
    ax.barh(list(y), profiles["n"].values, color=colors, alpha=0.88)
    ax.set_yticks(list(y))
    ax.set_yticklabels(
        [f"{row.segment_name}\n(cluster {row.cluster_id})" for row in profiles.itertuples()],
        fontsize=9,
    )
    ax.set_xlabel("Number of customers")
    ax.set_title("Segmentation V2 — Customer distribution by segment")
    for yi, row in zip(y, profiles.itertuples(), strict=False):
        ax.text(
            row.n + 80,
            yi,
            f"{row.n:,}  ({row.pct_of_total * 100:.1f}%)",
            va="center",
            fontsize=8.5,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_churn_profile(profiles: pd.DataFrame, path) -> None:
    """Grouped bar: churn rate + non-purchaser pct per segment."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(profiles))
    width = 0.38
    colors = [PALETTE.get(name, DEFAULT_COLOR) for name in profiles["segment_name"]]

    bars = ax.bar(
        x - width / 2,
        profiles["churn_rate"] * 100,
        width,
        label="Churn rate 90d (%)",
        color=colors,
        alpha=0.88,
    )
    ax.bar(
        x + width / 2,
        profiles["pct_non_purchaser"] * 100,
        width,
        label="% Non-purchaser",
        color=colors,
        alpha=0.45,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"C{row.cluster_id}\n{row.segment_name}" for row in profiles.itertuples()],
        fontsize=8.5,
    )
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Segmentation V2 — Churn rate & Non-purchaser share per segment")
    ax.legend(fontsize=8)

    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.3,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_vs_baseline(profiles: pd.DataFrame, baseline_profiles: pd.DataFrame, path) -> None:
    """2-panel comparison: V2 vs phase9 churn rate spread + segment sizes."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Panel 1 — churn rates
    for ax, prof, title in [
        (axes[0], baseline_profiles, "Phase9 (k=4)"),
        (axes[1], profiles, "V2 Nested k-means"),
    ]:
        churn_col = "churn_flag_90d" if "churn_flag_90d" in prof.columns else "churn_rate"
        name_col = "segment_name" if "segment_name" in prof.columns else "segment_name"
        vals = prof[churn_col].values * 100
        names = prof[name_col].tolist()
        y = range(len(prof))
        ax.barh(list(y), vals, color="#E63946", alpha=0.80)
        ax.set_yticks(list(y))
        ax.set_yticklabels(names, fontsize=8.5)
        ax.set_xlabel("Churn rate 90d (%)")
        ax.set_title(f"{title}\nChurn rate per segment")
        for yi, v in zip(y, vals, strict=False):
            ax.text(v + 0.2, yi, f"{v:.1f}%", va="center", fontsize=8)

    fig.suptitle(
        "Segmentation V2 vs Phase9 — Churn discriminability", fontsize=11, fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _build_vs_baseline(metrics: dict, baseline_profiles: pd.DataFrame) -> pd.DataFrame:
    """Build a flat comparison DataFrame vs phase9."""
    p9_churn_spread = float(
        baseline_profiles["churn_flag_90d"].max() - baseline_profiles["churn_flag_90d"].min()
    )
    rows = [
        {
            "metric": "n_total_segments",
            "phase9_value": 4,
            "v2_value": metrics["n_total_segments"],
            "delta_v2_minus_baseline": metrics["n_total_segments"] - 4,
            "note": "V2 = 3 buyer segments + 1 Non-Purchasers segment",
        },
        {
            "metric": "silhouette_buyers",
            "phase9_value": 0.252,
            "v2_value": metrics["silhouette_buyers"],
            "delta_v2_minus_baseline": round(metrics["silhouette_buyers"] - 0.252, 4),
            "note": "Phase9 silhouette on k=4 PCA-6; V2 on buyer k=3 PCA-5",
        },
        {
            "metric": "churn_spread_pp",
            "phase9_value": round(p9_churn_spread * 100, 2),
            "v2_value": metrics["churn_spread_pp"],
            "delta_v2_minus_baseline": round(metrics["churn_spread_pp"] - p9_churn_spread * 100, 2),
            "note": "Churn rate range across segments (max - min) in percentage points",
        },
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────
    cust = load_mart("mart_customer_features", processed_dir=PROCESSED_DIR)
    print(f"Loaded: {len(cust):,} customers × {cust.shape[1]} cols")

    # ── Feature engineering ───────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fe = engineer_features(cust)

    n_buyers = int((fe["is_non_purchaser"] == 0).sum())
    n_nonpurch = int((fe["is_non_purchaser"] == 1).sum())
    print(
        f"Buyers: {n_buyers:,}  Non-purchasers: {n_nonpurch:,}  ({n_nonpurch / len(fe) * 100:.1f}%)"
    )

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"Training nested k-means (k={N_BUYER_CLUSTERS} buyers + 1 non-purchaser)…")
    model = train_segmentation(fe)
    print(
        f"Silhouette (buyers): {model.silhouette_buyers:.4f}  "
        f"Davies-Bouldin: {model.davies_bouldin_buyers:.4f}"
    )

    # ── Profiles & metrics ────────────────────────────────────────────────
    profiles = cluster_profiles(fe, model.labels_, model.n_buyer_clusters)
    metrics = segment_metrics(model, profiles)

    print("Segment profiles:")
    print(
        profiles[
            [
                "cluster_id",
                "segment_name",
                "n",
                "pct_of_total",
                "churn_rate",
                "pct_non_purchaser",
                "median_aov",
                "avg_revenue",
            ]
        ].to_string(index=False)
    )
    print(f"Churn spread: {metrics['churn_spread_pp']:.1f} pp")

    # ── Cluster assignments ───────────────────────────────────────────────
    assignments = fe[["customer_id"]].copy()
    assignments["cluster_id"] = model.labels_
    segment_name_map: dict[int, str] = profiles.set_index("cluster_id")["segment_name"].to_dict()
    assignments["segment_name"] = assignments["cluster_id"].map(segment_name_map)

    # ── PCA loadings ──────────────────────────────────────────────────────
    loadings = pd.DataFrame(
        model.pipeline.pca.components_.T,
        index=model.buyer_features,
        columns=[f"PC{i + 1}" for i in range(model.pipeline.pca.n_components_)],
    ).round(4)

    # ── Load baseline for comparison ──────────────────────────────────────
    baseline_path = OUTPUTS_DIR / "phase9_segmentation_cluster_profiles.csv"
    vs_baseline = pd.DataFrame()
    if baseline_path.exists():
        baseline_profiles = pd.read_csv(baseline_path)
        _plot_vs_baseline(profiles, baseline_profiles, OUTPUTS_DIR / f"{PREFIX}_vs_baseline.png")
        vs_baseline = _build_vs_baseline(metrics, baseline_profiles)
        print("\nVs baseline:")
        print(vs_baseline.to_string(index=False))

    # ── Plots ─────────────────────────────────────────────────────────────
    _plot_segment_distribution(profiles, OUTPUTS_DIR / f"{PREFIX}_segment_distribution.png")
    _plot_churn_profile(profiles, OUTPUTS_DIR / f"{PREFIX}_churn_profile.png")

    # ── Write outputs ─────────────────────────────────────────────────────
    paths = {
        "cluster_assignments": OUTPUTS_DIR / f"{PREFIX}_cluster_assignments.csv",
        "cluster_profiles": OUTPUTS_DIR / f"{PREFIX}_cluster_profiles.csv",
        "metrics": OUTPUTS_DIR / f"{PREFIX}_metrics.json",
        "pca_loadings": OUTPUTS_DIR / f"{PREFIX}_pca_loadings.csv",
        "vs_baseline": OUTPUTS_DIR / f"{PREFIX}_vs_baseline.csv",
        "segment_distribution": OUTPUTS_DIR / f"{PREFIX}_segment_distribution.png",
        "churn_profile": OUTPUTS_DIR / f"{PREFIX}_churn_profile.png",
        "vs_baseline_plot": OUTPUTS_DIR / f"{PREFIX}_vs_baseline.png",
    }

    assignments.to_csv(paths["cluster_assignments"], index=False)
    profiles.to_csv(paths["cluster_profiles"], index=False)
    loadings.to_csv(paths["pca_loadings"])
    if not vs_baseline.empty:
        vs_baseline.to_csv(paths["vs_baseline"], index=False)

    with open(paths["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nOutputs written:")
    for name, p in paths.items():
        if p.exists():
            print(f"  {name}: {p.name}")

    print(
        f"\nDone. Silhouette={model.silhouette_buyers:.4f}  Churn spread={metrics['churn_spread_pp']:.1f}pp"
    )


if __name__ == "__main__":
    main()
