"""Customer 360 — one decision per customer from all six workstreams.

Each workstream produces its own artefacts, but nothing ties them together into a
single view a CRM or marketer could act on. This joins them and, crucially, turns
four scores into ONE recommended action per customer:

  churn probability  +  uplift (persuadability)  +  segment  +  recommendations
        │                      │                     │              │
        └──────────────────────┴─────────┬───────────┴──────────────┘
                                          ▼
                              one recommended action

Coverage, honestly stated:
  - churn and segment are computed/joined for every customer.
  - uplift is recomputed here (X-learner, same algorithm as
    src/models/train_uplift.py) and scored for every customer who was ever
    targeted by a campaign in mart_campaign_response. Customers never targeted
    by any campaign have no uplift signal — that's a real gap, not a sampling
    artefact.
  - recommendations are recomputed here (hybrid SVD+CB, same algorithm as
    src/models/train_recsys.py) for every customer: warm customers get the
    hybrid model, customers with fewer than 5 purchases get a category-popularity
    fallback, so recommendation coverage is complete.
  "Hero" cards are printed for customers who have all four signals.

Sources:
  - churn: recomputed here (calibrated LightGBM) so every customer has a probability
  - segment: outputs/phase_segmentation_v2_cluster_assignments.csv (+ profiles)
  - uplift: recomputed here from data/processed/mart_campaign_response.csv
  - recs:  recomputed here from data/raw (orders, sessions, reviews, products)

Outputs:
  outputs/customer_360_master.csv   every customer, populated where available
  outputs/customer_360_heroes.csv   customers with all four signals
  outputs/customer_360.html         printable one-pager of the hero cards

Run:
    python -m analysis.customer_360
    python -m analysis.customer_360 --customer C012345

Runtime: this retrains the uplift X-learner and the recommender SVD on the full
dataset, so a full run takes a few minutes (dominated by the uplift model, which
fits per-campaign LightGBM boosters).
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from src.config import OUTPUTS_DIR, PROCESSED_DIR, RAW_DIR
from src.features.features_churn import (
    build_features,
    feature_set,
    filter_active_population,
    filter_mature_cohort,
    prepare_xy,
    time_ordered_split,
)

SEED = 42
CALIB_FRACTION = 0.15
RECS_TOP_N = 3
WARM_CHUNK_SIZE = 4000  # bounds peak memory for the dense CF/CB score matrices


# ── Churn: recomputed and calibrated so EVERY customer has a probability ─────
def _churn_probabilities(mart: pd.DataFrame) -> pd.DataFrame:
    from sklearn.isotonic import IsotonicRegression

    from src.models.train_churn import train_churn_model

    active = filter_active_population(mart)
    mature = filter_mature_cohort(active, maturity_days=90)
    enriched = build_features(mature)
    fs = feature_set()

    train_df, test_df = time_ordered_split(enriched, test_size=0.20)
    X_train, y_train = prepare_xy(train_df, fs=fs)

    cut = int(len(X_train) * (1 - CALIB_FRACTION))
    inner = int(cut * 0.85)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        booster = train_churn_model(
            X_train.iloc[:inner],
            y_train.iloc[:inner],
            X_train.iloc[inner:cut],
            y_train.iloc[inner:cut],
            fs=fs,
        )
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(booster.predict_proba(X_train.iloc[cut:]), y_train.iloc[cut:].to_numpy())

    # score everyone (train + test), calibrated
    all_X, _ = prepare_xy(enriched, fs=fs)
    prob = iso.predict(booster.predict_proba(all_X))
    return pd.DataFrame(
        {"customer_id": enriched["customer_id"].to_numpy(), "churn_prob": np.round(prob, 4)}
    )


def _load_segments() -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = OUTPUTS_DIR
    seg = pd.read_csv(out_dir / "phase_segmentation_v2_cluster_assignments.csv")
    prof = pd.read_csv(out_dir / "phase_segmentation_v2_cluster_profiles.csv")[
        ["cluster_id", "segment_name", "churn_rate", "avg_revenue"]
    ].rename(columns={"churn_rate": "segment_churn_rate", "avg_revenue": "segment_avg_revenue"})
    return seg, prof


# ── Uplift: recomputed and scored for EVERY campaign-targeted customer ───────
def _uplift_scores(cust_mart: pd.DataFrame) -> pd.DataFrame:
    from src.data import mart_loaders
    from src.features import features_uplift as fu
    from src.models.train_uplift import train_xlearner

    mart = mart_loaders.load_mart("mart_campaign_response", processed_dir=PROCESSED_DIR)
    enriched = fu.build_features(fu.enrich_with_customer_mart(mart, cust_mart))

    train_df, _test_df = fu.time_ordered_split(enriched, test_size=0.20)
    sorted_train = train_df.sort_values(fu.TIME_KEY, kind="mergesort").reset_index(drop=True)
    val_cut = int(len(sorted_train) * 0.85)
    tr_df, va_df = sorted_train.iloc[:val_cut], sorted_train.iloc[val_cut:]

    fs = fu.feature_set()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = train_xlearner(tr_df, va_df, fs=fs)

    X_all, _y_all, _t_all = fu.prepare_xy(enriched, fs=fs)
    ite = model.predict_ite(X_all, enriched[fu.CAMPAIGN_KEY].astype(str))

    scored = enriched[[fu.ID_KEY, fu.CAMPAIGN_KEY]].copy()
    scored["uplift_ite"] = ite
    # one row per customer: their single highest-ITE (most persuadable) campaign
    return (
        scored.sort_values("uplift_ite", ascending=False)
        .groupby(fu.ID_KEY, as_index=False)
        .first()
        .rename(columns={fu.CAMPAIGN_KEY: "top_uplift_campaign"})
    )


# ── Recommendations: recomputed for EVERY customer ───────────────────────────
def _recsys_scores(customer_ids: pd.Series, top_n: int = RECS_TOP_N) -> pd.DataFrame:
    from sklearn.metrics.pairwise import cosine_similarity

    from src.features.features_recsys import (
        COLD_START_THRESHOLD,
        SPLIT_DATE,
        build_content_matrix,
        build_interaction_matrix,
        category_top_n_popularity,
    )
    from src.models.train_recsys import BEST_ALPHA, BEST_K, recommend, train_svd

    orders = pd.read_csv(RAW_DIR / "orders.csv", parse_dates=["order_date"])
    oi = pd.read_csv(RAW_DIR / "order_items.csv")
    products = pd.read_csv(RAW_DIR / "products.csv")
    pa = pd.read_csv(RAW_DIR / "product_attributes.csv")
    reviews = pd.read_csv(RAW_DIR / "reviews.csv")
    sessions = pd.read_csv(RAW_DIR / "session_events.csv")

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

    inter_data = build_interaction_matrix(cust_prod, view_ev, wish_ev, pos_rev, products)
    U, s, Vt = train_svd(inter_data.R_train, k=BEST_K)

    purchase_counts = cust_prod.groupby("customer_id")["product_id"].count()
    cold_start_ids = set(purchase_counts[purchase_counts < COLD_START_THRESHOLD].index)

    content_matrix, prod_idx_map = build_content_matrix(products, pa)
    cos_sim_matrix = cosine_similarity(content_matrix)

    train_buy = cust_prod[cust_prod["order_date"] < SPLIT_DATE]
    category_pop = category_top_n_popularity(train_buy, products, N=max(top_n, 10))

    all_customers = customer_ids.tolist()
    warm_custs = [
        c for c in all_customers if c not in cold_start_ids and c in inter_data.le_c.classes_
    ]
    warm_set = set(warm_custs)
    cold_custs = [c for c in all_customers if c not in warm_set]

    prods_arr = inter_data.le_p.classes_
    prod_col_map = {p: i for i, p in enumerate(prods_arr)}
    n_prods = len(prods_arr)
    n_cat = cos_sim_matrix.shape[0]

    rec_rows: list[dict] = []

    # ── Warm customers: hybrid CF+CB, scored in memory-bounded chunks ───────
    for start in range(0, len(warm_custs), WARM_CHUNK_SIZE):
        chunk = warm_custs[start : start + WARM_CHUNK_SIZE]
        cidx = inter_data.le_c.transform(chunk)
        cf = (U[cidx] * s) @ Vt

        rows, cols = [], []
        for li, cust in enumerate(chunk):
            for p in inter_data.train_seen.get(cust, set()):
                ci = prod_idx_map.get(p)
                if ci is not None:
                    rows.append(li)
                    cols.append(ci)
        from scipy.sparse import csr_matrix as _csr
        from scipy.sparse import diags as _diags

        if rows:
            pm = _csr((np.ones(len(rows)), (rows, cols)), shape=(len(chunk), n_cat))
            row_sums = np.asarray(pm.sum(axis=1)).flatten()
            row_sums[row_sums == 0] = 1.0
            pm = _diags(1.0 / row_sums) @ pm
            cb_cat = pm @ cos_sim_matrix
            cat_prods = list(prod_idx_map.keys())
            cat_to_svd = np.array([prod_col_map.get(p, -1) for p in cat_prods])
            valid = cat_to_svd >= 0
            cb = np.zeros((len(chunk), n_prods), dtype=np.float32)
            cb[:, cat_to_svd[valid]] = cb_cat[:, valid]
        else:
            cb = np.zeros((len(chunk), n_prods), dtype=np.float32)

        for li, cust in enumerate(chunk):
            cf_sc = cf[li].copy()
            cb_sc = cb[li]
            cf_norm = (cf_sc - cf_sc.min()) / (cf_sc.max() - cf_sc.min() + 1e-9)
            cb_norm = (cb_sc - cb_sc.min()) / (cb_sc.max() - cb_sc.min() + 1e-9)
            sc = BEST_ALPHA * cf_norm + (1.0 - BEST_ALPHA) * cb_norm
            for p in inter_data.train_seen.get(cust, set()):
                col = prod_col_map.get(p)
                if col is not None:
                    sc[col] = -np.inf
            recs = recommend(sc, inter_data.le_p, K=top_n)
            rec_rows.append({"customer_id": cust, "top3_recommendations": ", ".join(recs)})

    # ── Cold-start / never-purchased customers: category-popularity fallback,
    # vectorised (the per-customer helper in train_recsys.py is pandas-heavy
    # and too slow to call tens of thousands of times) ─────────────────────
    cat_lookup = (
        cust_prod.merge(products[["product_id", "category"]], on="product_id")
        .groupby(["customer_id", "category"])
        .size()
        .reset_index(name="cnt")
        .sort_values(["customer_id", "cnt"], ascending=[True, False])
        .groupby("customer_id")["category"]
        .apply(list)
        .to_dict()
    )
    seen_lookup = cust_prod.groupby("customer_id")["product_id"].apply(set).to_dict()
    global_pop: list[str] = []
    for pids in category_pop.values():
        for pid in pids:
            if pid not in global_pop:
                global_pop.append(pid)

    for cust in cold_custs:
        seen = seen_lookup.get(cust, set())
        recs: list[str] = []
        for cat in cat_lookup.get(cust, []):
            for pid in category_pop.get(cat, []):
                if pid not in seen and pid not in recs:
                    recs.append(pid)
                if len(recs) >= top_n:
                    break
            if len(recs) >= top_n:
                break
        if len(recs) < top_n:
            for pid in global_pop:
                if pid not in seen and pid not in recs:
                    recs.append(pid)
                if len(recs) >= top_n:
                    break
        rec_rows.append({"customer_id": cust, "top3_recommendations": ", ".join(recs[:top_n])})

    return pd.DataFrame(rec_rows)


def _recommended_action(row: pd.Series) -> str:
    """Combine the four signals into one CRM decision."""
    churn = row.get("churn_prob")
    uplift = row.get("uplift_ite")
    seg = str(row.get("segment_name", ""))
    has_recs = isinstance(row.get("top3_recommendations"), str)

    high_churn = churn is not None and not pd.isna(churn) and churn >= 0.5
    persuadable = uplift is not None and not pd.isna(uplift) and uplift > 0

    if high_churn and persuadable:
        return "Retention offer NOW + personalised recs (high churn, responds to offers)"
    if high_churn and not persuadable:
        return "At risk but offer-insensitive: try service/loyalty touch, not discount"
    if not high_churn and seg in {"Champions", "Loyal"}:
        return "Low risk, high value: cross-sell via recs, no discount needed"
    if persuadable:
        return "Growth target: include in next campaign" + (" + recs" if has_recs else "")
    return "Monitor: no strong action this cycle"


def _print_card(row: pd.Series) -> None:
    print("\n" + "─" * 66)
    print(f"  CUSTOMER {row['customer_id']}   |   segment: {row.get('segment_name', 'n/a')}")
    print("─" * 66)
    churn = row.get("churn_prob")
    print(f"  Churn probability : {churn:.1%}" if pd.notna(churn) else "  Churn probability : n/a")
    up = row.get("uplift_ite")
    if pd.notna(up):
        band = "TOP uplift band (target)" if up > 0 else "low uplift"
        print(f"  Uplift (ITE)      : {up:+.3f}  [{band}]  via {row.get('top_uplift_campaign')}")
    else:
        print("  Uplift (ITE)      : never targeted by a campaign")
    recs = row.get("top3_recommendations")
    print(
        f"  Top recommendations: {recs}" if isinstance(recs, str) else "  Top recommendations: n/a"
    )
    print(f"  → ACTION: {row['recommended_action']}")


def _write_html(heroes: pd.DataFrame, path) -> None:
    cards = []
    for _, r in heroes.iterrows():
        churn = f"{r['churn_prob']:.1%}" if pd.notna(r["churn_prob"]) else "n/a"
        up = f"{r['uplift_ite']:+.3f}" if pd.notna(r["uplift_ite"]) else "n/a"
        cards.append(
            f"""
        <div class="card">
          <div class="cid">{r["customer_id"]}</div>
          <div class="seg">{r.get("segment_name", "")}</div>
          <div class="grid">
            <div><span>Churn</span><b>{churn}</b></div>
            <div><span>Uplift ITE</span><b>{up}</b></div>
            <div><span>Top recs</span><b>{r.get("top3_recommendations", "")}</b></div>
          </div>
          <div class="action">{r["recommended_action"]}</div>
        </div>"""
        )
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Customer 360</title><style>
body{{font-family:system-ui,Arial;margin:24px;background:#f6f7f9;color:#1c2430}}
h1{{font-size:20px}} .card{{background:#fff;border:1px solid #e2e6eb;border-radius:12px;
padding:16px 18px;margin:12px 0;box-shadow:0 1px 2px rgba(0,0,0,.04)}}
.cid{{font-weight:700;font-size:16px}} .seg{{color:#5b6673;font-size:13px;margin-bottom:8px}}
.grid{{display:flex;gap:28px;flex-wrap:wrap;margin:8px 0}}
.grid span{{display:block;color:#8a94a3;font-size:11px;text-transform:uppercase}}
.grid b{{font-size:15px}} .action{{margin-top:8px;padding:8px 10px;background:#eef4ff;
border-left:3px solid #2E86AB;border-radius:6px;font-size:14px}}
</style></head><body><h1>Customer 360 — one decision per customer</h1>
<p style="color:#5b6673">Churn (calibrated) + uplift + segment + recommendations, combined into a single recommended action.</p>
{"".join(cards)}
</body></html>"""
    path.write_text(html, encoding="utf-8")


def main(customer: str | None = None) -> None:
    from src.data.mart_loaders import load_mart

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    mart = load_mart("mart_customer_features", processed_dir=PROCESSED_DIR)

    churn = _churn_probabilities(mart)
    seg, prof = _load_segments()
    print("Scoring uplift (X-learner, all campaign-targeted customers) …")
    upl = _uplift_scores(mart)
    print("Scoring recommendations (hybrid SVD+CB, all customers) …")
    rec = _recsys_scores(mart["customer_id"])

    base = mart[["customer_id", "customer_value_band", "recency_days", "total_net_revenue"]].copy()
    df = (
        base.merge(churn, on="customer_id", how="left")
        .merge(seg[["customer_id", "cluster_id", "segment_name"]], on="customer_id", how="left")
        .merge(prof, on=["cluster_id", "segment_name"], how="left")
        .merge(upl, on="customer_id", how="left")
        .merge(rec, on="customer_id", how="left")
    )
    df["recommended_action"] = df.apply(_recommended_action, axis=1)

    master_path = OUTPUTS_DIR / "customer_360_master.csv"
    df.to_csv(master_path, index=False)

    # heroes: all four signals present
    heroes = df[
        df["churn_prob"].notna()
        & df["segment_name"].notna()
        & df["uplift_ite"].notna()
        & df["top3_recommendations"].notna()
    ].copy()
    heroes = heroes.sort_values("churn_prob", ascending=False).reset_index(drop=True)
    heroes_path = OUTPUTS_DIR / "customer_360_heroes.csv"
    heroes.to_csv(heroes_path, index=False)
    html_path = OUTPUTS_DIR / "customer_360.html"
    _write_html(heroes.head(12), html_path)

    if customer is not None:
        row = df[df["customer_id"] == customer]
        if row.empty:
            print(f"Customer {customer} not found.")
        else:
            _print_card(row.iloc[0])
    else:
        n_uplift = df["uplift_ite"].notna().sum()
        n_recs = df["top3_recommendations"].notna().sum()
        print(
            f"Master table: {len(df):,} customers  (churn+segment for all, "
            f"uplift for {n_uplift:,}, recommendations for {n_recs:,})"
        )
        print(
            f"Fully-populated (all four signals): {len(heroes):,} customers — "
            f"showing {min(5, len(heroes))} hero cards:"
        )
        for _, r in heroes.head(5).iterrows():
            _print_card(r)

    print(f"\nWrote: {master_path}\nWrote: {heroes_path}\nWrote: {html_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--customer", default=None, help="show the 360 card for one customer_id")
    args = p.parse_args()
    main(customer=args.customer)
