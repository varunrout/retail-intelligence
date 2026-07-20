"""Phase Recsys V2 runner.

Produces V2 recommendation artefacts that mirror the phase10 baseline
schema so the two can be compared side by side.  Implementation reflects the
21 decisions documented in ``analysis_notebooks/recsys_analysis.ipynb``.

V2 Algorithm — Multi-Signal SVD with Hybrid Blending
======================================================
  Phase10 used buy-only implicit SVD-50 evaluated without a proper holdout,
  reporting hit@10=0.0104.  V2 improves along four axes:

    1. Signal richness: log1p(qty)×1.0 + log1p(views)×0.15 +
                        log1p(wishes)×0.40 + pos_reviews×0.5
       → 2.19M multi-signal pairs vs 1.34M buy-only (+63%)

    2. Optimal rank: k=200 (confirmed by §5 k-sweep)
       → CF-only hit@10=0.1043 (vs 0.0104 phase10)

    3. Hybrid blending: alpha=0.8 CF + 0.2 CB content features
       → hybrid hit@10=0.1225 (further +17% over CF-only)

    4. Cold-start routing: customers with <5 purchases get
       category-aware popularity top-10 fallback

  Time-ordered train/test split at 2025-11-01.
  Eval: hit_rate@10, MRR@10 on novel test products only (97.9% of test
  purchases are products never bought by that customer in train).

Outputs
-------
  {PREFIX}_metrics.json              scalar evaluation metrics
  {PREFIX}_recommendations_sample.csv  top-10 recs for 500 customers
  {PREFIX}_model_comparison.csv      V2 vs phase10 method comparison
  {PREFIX}_vs_baseline.csv           metric-level comparison table
  {PREFIX}_hr_at_k_curve.png         hit_rate@K chart
  {PREFIX}_signal_breakdown.png      signal contribution chart
"""

from __future__ import annotations

import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.config import OUTPUTS_DIR, RAW_DIR
from src.features.features_recsys import (
    COLD_START_THRESHOLD,
    SPLIT_DATE,
    build_content_matrix,
    build_interaction_matrix,
    category_top_n_popularity,
)
from src.models.train_recsys import (
    BEST_ALPHA,
    BEST_K,
    cold_start_recs,
    evaluate_hitrate_mrr,
    hr_at_k_curve,
    recommend,
    train_svd,
)

PREFIX = "phase_recsys_v2"

PALETTE = {
    "primary": "#2E86AB",
    "secondary": "#E63946",
    "tertiary": "#2a9d8f",
    "neutral": "#6c757d",
    "highlight": "#f4a261",
}


def main() -> None:  # noqa: C901 (complexity acceptable for pipeline runner)
    warnings.filterwarnings("ignore")
    print(f"[{PREFIX}] Starting Recommendation Systems V2 pipeline")

    # ── Load raw data ──────────────────────────────────────────────────────
    print("  Loading raw data...")
    orders = pd.read_csv(RAW_DIR / "orders.csv", parse_dates=["order_date"])
    oi = pd.read_csv(RAW_DIR / "order_items.csv")
    products = pd.read_csv(RAW_DIR / "products.csv")
    pa = pd.read_csv(RAW_DIR / "product_attributes.csv")
    reviews = pd.read_csv(RAW_DIR / "reviews.csv")
    sessions = pd.read_csv(RAW_DIR / "session_events.csv")

    # ── Prepare interaction frames ─────────────────────────────────────────
    cust_prod = oi.merge(orders[["order_id", "customer_id", "order_date"]], on="order_id")

    view_ev = sessions[
        (sessions.event_type == "view_product")
        & sessions.customer_id_nullable.notna()
        & sessions.product_id_nullable.notna()
    ].rename(columns={"customer_id_nullable": "customer_id", "product_id_nullable": "product_id"})
    wish_ev = sessions[
        (sessions.event_type == "wishlist_add")
        & sessions.customer_id_nullable.notna()
        & sessions.product_id_nullable.notna()
    ].rename(columns={"customer_id_nullable": "customer_id", "product_id_nullable": "product_id"})
    pos_rev = reviews[reviews.rating >= 4][["customer_id", "product_id"]].copy()

    # ── Build interaction matrix ───────────────────────────────────────────
    print("  Building interaction matrix (time-ordered split at 2025-11-01)...")
    inter_data = build_interaction_matrix(cust_prod, view_ev, wish_ev, pos_rev, products)

    print(f"    R_train shape: {inter_data.R_train.shape}  nnz={inter_data.R_train.nnz:,}")
    print(f"    Eval customers (novel test): {inter_data.n_eval_customers:,}")

    # ── Identify cold-start customers ──────────────────────────────────────
    purchase_counts = cust_prod.groupby("customer_id")["product_id"].count()
    cold_start_ids = set(purchase_counts[purchase_counts < COLD_START_THRESHOLD].index)
    print(f"    Cold-start customers (<{COLD_START_THRESHOLD} purchases): {len(cold_start_ids):,}")

    # ── Build content features ─────────────────────────────────────────────
    print("  Building product content feature matrix and cosine similarity...")
    content_matrix, prod_idx_map = build_content_matrix(products, pa)
    cos_sim_matrix = cosine_similarity(content_matrix)  # (5000, 5000) — computed once
    print(f"    Content matrix: {content_matrix.shape}  cos_sim: {cos_sim_matrix.shape}")

    # ── Build category popularity (for cold-start) ─────────────────────────
    train_buy = cust_prod[cust_prod["order_date"] < SPLIT_DATE]
    category_pop = category_top_n_popularity(train_buy, products, N=10)

    # ── Train SVD ─────────────────────────────────────────────────────────
    print(f"  Training SVD (k={BEST_K}) on sparse multi-signal matrix...")
    U, s, Vt = train_svd(inter_data.R_train, k=BEST_K)
    print(f"    SVD complete.  U={U.shape}  Vt={Vt.shape}")

    # ── Evaluate (full hybrid on all eval customers) ───────────────────────
    print(
        f"  Evaluating hybrid (alpha={BEST_ALPHA}) on all {inter_data.n_eval_customers:,} eval customers..."
    )
    metrics = evaluate_hitrate_mrr(
        U,
        s,
        Vt,
        inter_data,
        cos_sim_matrix=cos_sim_matrix,
        prod_idx_map=prod_idx_map,
        alpha=BEST_ALPHA,
        n_eval=None,  # full evaluation
        K=10,
    )
    print(f"    hit_rate@10={metrics['hit_rate']:.4f}  MRR@10={metrics['mrr']:.4f}")

    # ── Hit@K curve data ──────────────────────────────────────────────────
    print("  Computing HR@K curve...")
    curve_df = hr_at_k_curve(
        U,
        s,
        Vt,
        inter_data,
        cos_sim_matrix=cos_sim_matrix,
        prod_idx_map=prod_idx_map,
        alpha=BEST_ALPHA,
        k_values=[1, 5, 10, 20, 50],
        n_eval=3000,
    )

    # ── Generate sample recommendations (500 customers) ───────────────────
    print("  Generating sample recommendations for 500 customers...")
    sample_custs = list(inter_data.test_novel.keys())[:500]
    prods_arr = inter_data.le_p.classes_
    prod_col_map = {p: i for i, p in enumerate(prods_arr)}
    # Batch CF for all warm sample customers at once
    warm_custs = [
        c for c in sample_custs if c not in cold_start_ids and c in inter_data.le_c.classes_
    ]
    cold_custs = [
        c for c in sample_custs if c in cold_start_ids or c not in inter_data.le_c.classes_
    ]
    warm_cidx = inter_data.le_c.transform(warm_custs) if warm_custs else []
    warm_cf = (U[warm_cidx] * s) @ Vt if len(warm_cidx) > 0 else np.empty((0, len(prods_arr)))

    # Build CB scores for warm customers (sparse×dense)
    n_prods = len(prods_arr)
    n_cat = cos_sim_matrix.shape[0]
    if warm_custs:
        wr, wc = [], []
        for li, cust in enumerate(warm_custs):
            for p in inter_data.train_seen.get(cust, set()):
                ci = prod_idx_map.get(p)
                if ci is not None:
                    wr.append(li)
                    wc.append(ci)
        from scipy.sparse import csr_matrix as _csr
        from scipy.sparse import diags as _diags

        if wr:
            pm = _csr((np.ones(len(wr)), (wr, wc)), shape=(len(warm_custs), n_cat))
            rs = np.asarray(pm.sum(axis=1)).flatten()
            rs[rs == 0] = 1.0
            pm = _diags(1.0 / rs) @ pm
            warm_cb_cat = pm @ cos_sim_matrix  # (n_warm, 5000)
            cat_prods = list(prod_idx_map.keys())
            cat_to_svd = np.array([prod_col_map.get(p, -1) for p in cat_prods])
            valid = cat_to_svd >= 0
            warm_cb = np.zeros((len(warm_custs), n_prods), dtype=np.float32)
            warm_cb[:, cat_to_svd[valid]] = warm_cb_cat[:, valid]
        else:
            warm_cb = np.zeros((len(warm_custs), n_prods), dtype=np.float32)
    else:
        warm_cb = np.empty((0, n_prods))

    rec_rows = []
    for li, cust in enumerate(warm_custs):
        cf_sc = warm_cf[li].copy()
        cb_sc = warm_cb[li]
        cf_min, cf_max = cf_sc.min(), cf_sc.max()
        cf_norm = (cf_sc - cf_min) / (cf_max - cf_min + 1e-9)
        cb_min, cb_max = cb_sc.min(), cb_sc.max()
        cb_norm = (cb_sc - cb_min) / (cb_max - cb_min + 1e-9)
        sc = BEST_ALPHA * cf_norm + (1.0 - BEST_ALPHA) * cb_norm
        for p in inter_data.train_seen.get(cust, set()):
            col = prod_col_map.get(p)
            if col is not None:
                sc[col] = -np.inf
        recs = recommend(sc, inter_data.le_p, K=10)
        for rank, pid in enumerate(recs, 1):
            rec_rows.append(
                {"customer_id": cust, "rank": rank, "product_id": pid, "method": "hybrid_svd"}
            )

    for cust in cold_custs:
        recs = cold_start_recs(cust, cust_prod, products, category_pop, K=10)
        for rank, pid in enumerate(recs, 1):
            rec_rows.append(
                {"customer_id": cust, "rank": rank, "product_id": pid, "method": "cold_start"}
            )

    rec_df = pd.DataFrame(rec_rows)

    # ── Model comparison table ─────────────────────────────────────────────
    model_comparison = pd.DataFrame(
        [
            {
                "method": "Popularity (global)",
                "hit_rate@10": 0.0036,
                "mrr@10": 0.0016,
                "eval": "phase10",
            },
            {
                "method": "Content-Based (CB)",
                "hit_rate@10": 0.0062,
                "mrr@10": 0.0022,
                "eval": "phase10",
            },
            {
                "method": "SVD-50 (buy-only)",
                "hit_rate@10": 0.0104,
                "mrr@10": 0.0035,
                "eval": "phase10",
            },
            {
                "method": "SVD-50 (time-split)",
                "hit_rate@10": 0.0630,
                "mrr@10": 0.0178,
                "eval": "v2_timesplit",
            },
            {
                "method": "Multi-signal SVD-200",
                "hit_rate@10": 0.1043,
                "mrr@10": 0.0549,
                "eval": "v2_timesplit",
            },
            {
                "method": "Hybrid SVD+CB (α=0.8)",
                "hit_rate@10": metrics["hit_rate"],
                "mrr@10": metrics["mrr"],
                "eval": "v2_timesplit",
            },
        ]
    )

    # ── vs_baseline comparison ─────────────────────────────────────────────
    vs_baseline = pd.DataFrame(
        [
            {
                "metric": "hit_rate@10",
                "phase10_value": 0.0104,
                "v2_value": metrics["hit_rate"],
                "delta": metrics["hit_rate"] - 0.0104,
            },
            {
                "metric": "mrr@10",
                "phase10_value": 0.0035,
                "v2_value": metrics["mrr"],
                "delta": metrics["mrr"] - 0.0035,
            },
        ]
    )

    # ── Persist artefacts ──────────────────────────────────────────────────
    print("  Saving artefacts...")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # metrics.json
    metrics_out = {
        "hit_rate_at_10": round(metrics["hit_rate"], 6),
        "mrr_at_10": round(metrics["mrr"], 6),
        "n_evaluated": metrics["n_evaluated"],
        "svd_k": int(BEST_K),
        "alpha": float(BEST_ALPHA),
        "cold_start_threshold": COLD_START_THRESHOLD,
        "cold_start_customers": len(cold_start_ids),
        "phase10_hit_rate_at_10": 0.0104,
        "improvement_factor": round(metrics["hit_rate"] / 0.0104, 2),
    }
    with open(OUTPUTS_DIR / f"{PREFIX}_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    # recommendations_sample.csv
    rec_df.to_csv(OUTPUTS_DIR / f"{PREFIX}_recommendations_sample.csv", index=False)

    # model_comparison.csv
    model_comparison.to_csv(OUTPUTS_DIR / f"{PREFIX}_model_comparison.csv", index=False)

    # vs_baseline.csv
    vs_baseline.to_csv(OUTPUTS_DIR / f"{PREFIX}_vs_baseline.csv", index=False)

    # ── HR@K curve chart ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(
        curve_df["K"],
        curve_df["hit_rate"] * 100,
        marker="o",
        color=PALETTE["primary"],
        linewidth=2,
        markersize=7,
        label="V2 Hybrid",
    )
    axes[0].axhline(
        0.0104 * 100,
        color=PALETTE["neutral"],
        linestyle="--",
        linewidth=1.5,
        label="Phase10 SVD@K=10 (0.0104)",
    )
    for _, row in curve_df.iterrows():
        axes[0].annotate(
            f"{row['hit_rate'] * 100:.1f}%",
            (row["K"], row["hit_rate"] * 100),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
        )
    axes[0].set_xlabel("K (recommendations per customer)")
    axes[0].set_ylabel("Hit Rate@K (%)")
    axes[0].set_title("V2 Hit Rate@K — Hybrid SVD (α=0.8)")
    axes[0].legend(fontsize=8)

    # Panel 2: Model comparison bar chart
    comp_methods = model_comparison["method"].tolist()
    comp_hr = model_comparison["hit_rate@10"].tolist()
    colors2 = [PALETTE["neutral"]] * 3 + [PALETTE["primary"]] * 2 + [PALETTE["secondary"]]
    axes[1].bar(range(len(comp_methods)), comp_hr, color=colors2, alpha=0.85)
    axes[1].set_xticks(range(len(comp_methods)))
    axes[1].set_xticklabels(comp_methods, rotation=20, ha="right", fontsize=8.5)
    axes[1].set_ylabel("Hit Rate@10")
    axes[1].set_title("Method comparison — Hit Rate@10")
    for i, v in enumerate(comp_hr):
        axes[1].text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=8)

    fig.suptitle(f"{PREFIX} — Recommendation Systems V2", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUTPUTS_DIR / f"{PREFIX}_hr_at_k_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Signal breakdown chart ─────────────────────────────────────────────
    signal_names = ["buy_only", "buy+view", "buy+view+wish", "buy+view+wish+rev"]
    signal_hr = [0.0630, 0.0610, 0.0617, 0.0783]  # from §4 analysis (SVD-50)
    signal_colors = [PALETTE["neutral"]] * 3 + [PALETTE["secondary"]]

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.bar(range(len(signal_names)), signal_hr, color=signal_colors, alpha=0.85)
    ax2.set_xticks(range(len(signal_names)))
    ax2.set_xticklabels(signal_names, rotation=15, ha="right", fontsize=9)
    ax2.set_ylabel("Hit Rate@10 (SVD-50)")
    ax2.set_title("Signal scheme comparison (§4 analysis — SVD-50)")
    for i, v in enumerate(signal_hr):
        ax2.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=9)
    fig2.tight_layout()
    fig2.savefig(OUTPUTS_DIR / f"{PREFIX}_signal_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print(f"[{PREFIX}] ── Results ──────────────────────────────")
    print(
        f"  Hit Rate@10 : {metrics['hit_rate']:.4f}   (phase10: 0.0104 → ×{metrics['hit_rate'] / 0.0104:.1f})"
    )
    print(f"  MRR@10      : {metrics['mrr']:.4f}   (phase10: 0.0035)")
    print(f"  Customers evaluated : {metrics['n_evaluated']:,}")
    print(f"  Cold-start customers: {len(cold_start_ids):,}")
    print()
    print(f"  Outputs written to {OUTPUTS_DIR}/")
    for stem in [
        f"{PREFIX}_metrics.json",
        f"{PREFIX}_recommendations_sample.csv",
        f"{PREFIX}_model_comparison.csv",
        f"{PREFIX}_vs_baseline.csv",
        f"{PREFIX}_hr_at_k_curve.png",
        f"{PREFIX}_signal_breakdown.png",
    ]:
        print(f"    {stem}")


if __name__ == "__main__":
    main()
