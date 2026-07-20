"""
Feature engineering for Recommendation Systems V2.

Design decisions confirmed by recsys_analysis.ipynb:
  - Signal formula: log1p(qty)×1.0 + log1p(views)×0.15 + log1p(wishes)×0.40 + pos_rev×0.5
  - Split: time-ordered at 2025-11-01
  - Cold-start threshold: <5 purchases → category-aware popularity fallback
  - Content: OHE(category, subcategory, brand_type, price_tier) + embedding group prefix + binary flags
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder

# ── Constants ─────────────────────────────────────────────────────────────────
SPLIT_DATE = pd.Timestamp("2025-11-01")
COLD_START_THRESHOLD = 5  # customers with fewer purchases get popularity fallback

# Implicit signal weights (confirmed best from §4 signal sweep)
W_BUY = 1.0
W_VIEW = 0.15
W_WISH = 0.40
W_POS_REV = 0.5


# ── Dataclass ─────────────────────────────────────────────────────────────────
@dataclass
class InteractionData:
    """Holds train/test splits and encoders for the interaction matrix."""

    R_train: csr_matrix  # (n_customers × n_products) implicit score matrix
    le_c: LabelEncoder  # customer label encoder
    le_p: LabelEncoder  # product label encoder
    train_seen: dict[str, set]  # {customer_id: set of product_ids seen in train}
    test_novel: dict[str, set]  # {customer_id: set of novel test product_ids}
    n_train: int
    n_test: int
    n_eval_customers: int  # customers that have novel test products


# ── Signal construction ────────────────────────────────────────────────────────


def _implicit_score(df: pd.DataFrame) -> pd.Series:
    """Compute implicit score from multi-signal interaction frame."""
    qty = df.get("quantity", pd.Series(0, index=df.index)).fillna(0)
    views = df.get("view_count", pd.Series(0, index=df.index)).fillna(0)
    wishes = df.get("wish_count", pd.Series(0, index=df.index)).fillna(0)
    pos_rev = df.get("pos_review", pd.Series(0, index=df.index)).fillna(0)
    return (
        np.log1p(qty) * W_BUY
        + np.log1p(views) * W_VIEW
        + np.log1p(wishes) * W_WISH
        + pos_rev * W_POS_REV
    )


def build_interaction_matrix(
    cust_prod: pd.DataFrame,  # merged orders × order_items with order_date
    view_ev: pd.DataFrame,  # session events filtered to view_product with known customer/product
    wish_ev: pd.DataFrame,  # session events filtered to wishlist_add with known customer/product
    pos_rev: pd.DataFrame,  # reviews filtered to rating >= 4
    all_products: pd.DataFrame,  # products table (for full product universe)
) -> InteractionData:
    """
    Build time-ordered train/test interaction matrices.

    Train period : order_date < SPLIT_DATE
    Test  period : order_date >= SPLIT_DATE
    """
    # ── Time split ──────────────────────────────────────────────────────────
    train_buy = cust_prod[cust_prod["order_date"] < SPLIT_DATE].copy()
    test_buy = cust_prod[cust_prod["order_date"] >= SPLIT_DATE].copy()

    # Customers that appear in both halves
    overlap = set(train_buy.customer_id.unique()) & set(test_buy.customer_id.unique())

    # ── Aggregate train signals ──────────────────────────────────────────────
    buy_agg = (
        train_buy[train_buy.customer_id.isin(overlap)]
        .groupby(["customer_id", "product_id"])["quantity"]
        .sum()
        .reset_index()
    )

    # Views — filter to train period if event_time is available
    view_df = view_ev.copy()
    if "event_time" in view_df.columns:
        view_df = view_df[pd.to_datetime(view_df["event_time"]) < SPLIT_DATE]
    view_agg = view_df.groupby(["customer_id", "product_id"]).size().reset_index(name="view_count")

    wish_agg = wish_ev.groupby(["customer_id", "product_id"]).size().reset_index(name="wish_count")

    rev_agg = pos_rev.groupby(["customer_id", "product_id"]).size().reset_index(name="pos_review")

    # Merge all signals
    inter = reduce(
        lambda a, b: pd.merge(a, b, on=["customer_id", "product_id"], how="outer"),
        [buy_agg, view_agg, wish_agg, rev_agg],
    ).fillna(0)
    inter = inter[inter.customer_id.isin(overlap)]
    inter["score"] = _implicit_score(inter)
    inter = inter[inter["score"] > 0]

    # ── Encoders (full product + customer universe) ──────────────────────────
    all_custs = sorted(overlap)
    all_prods = sorted(all_products.product_id.unique())
    le_c = LabelEncoder().fit(all_custs)
    le_p = LabelEncoder().fit(all_prods)

    valid = inter[inter.customer_id.isin(le_c.classes_) & inter.product_id.isin(le_p.classes_)]
    ci = le_c.transform(valid.customer_id)
    pi = le_p.transform(valid.product_id)
    R_train = csr_matrix(
        (valid.score.values, (ci, pi)),
        shape=(len(all_custs), len(all_prods)),
    )

    # ── Holdout sets ─────────────────────────────────────────────────────────
    train_seen = (
        train_buy[train_buy.customer_id.isin(overlap)]
        .groupby("customer_id")["product_id"]
        .apply(set)
        .to_dict()
    )
    test_gt = (
        test_buy[test_buy.customer_id.isin(overlap)]
        .groupby("customer_id")["product_id"]
        .apply(set)
        .to_dict()
    )
    test_novel = {
        c: test_gt[c] - train_seen.get(c, set())
        for c in test_gt
        if (test_gt[c] - train_seen.get(c, set()))
    }

    return InteractionData(
        R_train=R_train,
        le_c=le_c,
        le_p=le_p,
        train_seen=train_seen,
        test_novel=test_novel,
        n_train=int(train_buy[train_buy.customer_id.isin(overlap)].shape[0]),
        n_test=int(test_buy[test_buy.customer_id.isin(overlap)].shape[0]),
        n_eval_customers=len(test_novel),
    )


# ── Content features ──────────────────────────────────────────────────────────


def build_content_matrix(
    products: pd.DataFrame,
    pa: pd.DataFrame,
) -> tuple[np.ndarray, dict]:
    """
    Build a product content feature matrix for hybrid scoring.

    Returns
    -------
    content_matrix : np.ndarray  shape (n_products, n_features)
    prod_idx_map   : dict {product_id: row_index}
    """
    prod_full = products.merge(
        pa[["product_id", "recommendation_embedding_group", "eco_flag", "bundle_candidate_flag"]],
        on="product_id",
    )

    cat_ohe = pd.get_dummies(
        prod_full[["category", "subcategory", "brand_type", "price_tier"]],
        prefix=["cat", "sub", "brand", "price"],
    ).astype(float)

    # Use first 2 chars of embedding group as coarse cluster indicator
    emb_grp = pd.get_dummies(
        prod_full["recommendation_embedding_group"].astype(str).str[:2],
        prefix="emb",
    ).astype(float)

    bin_feats = prod_full[["eco_flag", "bundle_candidate_flag"]].astype(float)

    content_matrix = np.hstack([cat_ohe.values, emb_grp.values, bin_feats.values])
    prod_idx_map = {p: i for i, p in enumerate(prod_full.product_id)}

    return content_matrix, prod_idx_map


# ── Cold-start helpers ────────────────────────────────────────────────────────


def category_top_n_popularity(
    cust_prod: pd.DataFrame,
    products: pd.DataFrame,
    N: int = 10,
) -> dict:
    """
    Return {category: [top-N product_ids by purchase count]} for cold-start fallback.
    Uses the full purchase history (train only; caller should filter if needed).
    """
    cat_pop = (
        cust_prod.merge(products[["product_id", "category"]], on="product_id")
        .groupby(["category", "product_id"])
        .size()
        .reset_index(name="pop")
        .sort_values("pop", ascending=False)
    )
    return cat_pop.groupby("category")["product_id"].apply(lambda x: list(x[:N])).to_dict()


def get_customer_categories(
    customer_id: str,
    cust_prod: pd.DataFrame,
    products: pd.DataFrame,
    top_n: int = 3,
) -> list[str]:
    """Return a customer's most purchased categories (for cold-start routing)."""
    bought = cust_prod[cust_prod.customer_id == customer_id]
    if bought.empty:
        return []
    cat_counts = (
        bought.merge(products[["product_id", "category"]], on="product_id")
        .groupby("category")
        .size()
        .sort_values(ascending=False)
    )
    return list(cat_counts.index[:top_n])
