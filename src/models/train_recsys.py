"""
Training, scoring, and evaluation for Recommendation Systems V2.

Design decisions confirmed by recsys_analysis.ipynb:
  - Algorithm: SVD on multi-signal implicit matrix (scipy.sparse.linalg.svds — no densification)
  - Best k: 200 (confirmed by §5 k-sweep; hit@10=0.1043 CF-only)
  - Best hybrid: alpha=0.8 CF + 0.2 CB (confirmed by §7; hit@10=0.1225)
  - Eval: hit_rate@10 and MRR@10 on novel test products only
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

from src.features.features_recsys import (
    InteractionData,
    COLD_START_THRESHOLD,
    build_content_matrix,
    category_top_n_popularity,
    get_customer_categories,
)

# Best hyperparameters from analysis notebook
BEST_K     = 200
BEST_ALPHA = 0.8    # CF × alpha + CB × (1 - alpha)


# ── Model training ────────────────────────────────────────────────────────────

def train_svd(R: csr_matrix, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit truncated SVD on a sparse implicit interaction matrix.

    Parameters
    ----------
    R : csr_matrix  (n_customers × n_products)
    k : int         number of latent factors

    Returns
    -------
    U  : (n_customers, k)
    s  : (k,)
    Vt : (k, n_products)
    """
    U, s, Vt = svds(R, k=k)
    return U, s, Vt


# ── Scoring ───────────────────────────────────────────────────────────────────

def predict_scores(
    U: np.ndarray,
    s: np.ndarray,
    Vt: np.ndarray,
    customer_idx: int,
    seen_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute raw CF recommendation scores for one customer.

    Parameters
    ----------
    seen_mask : boolean array; True positions are masked to -inf (already purchased)

    Returns
    -------
    scores : (n_products,)
    """
    scores = (U[customer_idx] * s) @ Vt
    if seen_mask is not None:
        scores[seen_mask] = -np.inf
    return scores


def hybrid_scores(
    cf_scores: np.ndarray,
    cos_sim_matrix: np.ndarray,
    seen_product_indices: List[int],
    prods_arr: np.ndarray,
    prod_idx_map: dict,
    alpha: float = BEST_ALPHA,
) -> np.ndarray:
    """
    Blend CF scores with content-based scores using a precomputed similarity matrix.

    Content score for a customer = average cosine similarity to their seen products.
    Both CF and CB vectors are min-max normalised before blending.

    Parameters
    ----------
    cf_scores            : (n_products,) raw SVD scores (seen items already masked to -inf)
    cos_sim_matrix       : (n_catalog_products, n_catalog_products) precomputed cosine similarity
    seen_product_indices : row indices in cos_sim_matrix for seen products
    prods_arr            : array of product_ids in le_p order (length = n_products in SVD)
    prod_idx_map         : {product_id: cos_sim_matrix row index}
    alpha                : weight on CF (1-alpha goes to CB)

    Returns
    -------
    blended : (n_products,)
    """
    if not seen_product_indices or alpha >= 1.0:
        return cf_scores

    # Average similarity to seen products (fast: index precomputed matrix)
    cb_full = cos_sim_matrix[seen_product_indices].mean(axis=0)   # (n_catalog,)

    # Align to prods_arr order (le_p product ordering)
    cb = np.array([cb_full[prod_idx_map[p]] if p in prod_idx_map else 0.0 for p in prods_arr])

    # Normalise (excluding -inf masked entries)
    finite_mask = np.isfinite(cf_scores)
    cf_norm = np.where(finite_mask,
                       (cf_scores - cf_scores[finite_mask].min()) /
                       (cf_scores[finite_mask].max() - cf_scores[finite_mask].min() + 1e-9),
                       -np.inf)
    cb_norm = (cb - cb.min()) / (cb.max() - cb.min() + 1e-9)

    blended = np.where(finite_mask, alpha * cf_norm + (1.0 - alpha) * cb_norm, -np.inf)
    return blended


def recommend(
    scores: np.ndarray,
    le_p: LabelEncoder,
    K: int = 10,
    exclude_set: Optional[Set] = None,
) -> List[str]:
    """
    Return top-K product IDs from a score vector.

    Parameters
    ----------
    exclude_set : additional product_ids to exclude (e.g. train purchases)
    """
    prods = le_p.classes_
    sc = scores.copy()
    if exclude_set:
        mask = np.array([p in exclude_set for p in prods])
        sc[mask] = -np.inf
    top_idx = np.argsort(sc)[::-1][:K]
    return list(prods[top_idx])


# ── Cold-start recommendations ────────────────────────────────────────────────

def cold_start_recs(
    customer_id: str,
    cust_prod: pd.DataFrame,
    products: pd.DataFrame,
    category_pop: dict,
    K: int = 10,
) -> List[str]:
    """
    Produce top-K recommendations for cold-start customers (<5 purchases)
    using category-aware popularity fallback.

    Strategy:
      - Find the customer's top purchased categories (if any purchases exist)
      - Return the top-K most popular products from those categories
      - Fall back to global top-K if no category signal
    """
    preferred_cats = get_customer_categories(customer_id, cust_prod, products, top_n=3)
    seen = set(cust_prod[cust_prod.customer_id == customer_id]["product_id"])

    recs: List[str] = []
    for cat in preferred_cats:
        for pid in category_pop.get(cat, []):
            if pid not in seen and pid not in recs:
                recs.append(pid)
            if len(recs) >= K:
                return recs

    # Fallback: global popularity across all categories
    if len(recs) < K:
        for cat_pids in category_pop.values():
            for pid in cat_pids:
                if pid not in seen and pid not in recs:
                    recs.append(pid)
                if len(recs) >= K:
                    return recs

    return recs[:K]


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_hitrate_mrr(
    U: np.ndarray,
    s: np.ndarray,
    Vt: np.ndarray,
    inter_data: InteractionData,
    cos_sim_matrix: Optional[np.ndarray],
    prod_idx_map: Optional[dict],
    alpha: float = BEST_ALPHA,
    n_eval: int = 5000,
    K: int = 10,
) -> dict:
    """
    Evaluate hit_rate@K and MRR@K on novel test products.

    Only customers with at least one novel test product (never bought in train)
    are included in the denominator.

    Parameters
    ----------
    cos_sim_matrix : (n_catalog, n_catalog) precomputed product cosine similarity matrix
    n_eval         : cap on number of eval customers (for speed; use None for full eval)

    Returns
    -------
    dict with keys: hit_rate, mrr, n_evaluated
    """
    le_c = inter_data.le_c
    le_p = inter_data.le_p
    prods_arr = le_p.classes_
    n_prods = len(prods_arr)
    train_seen = inter_data.train_seen
    test_novel = inter_data.test_novel

    eval_custs = list(test_novel.keys())
    if n_eval is not None:
        eval_custs = eval_custs[:n_eval]

    # Only keep customers in the encoder
    eval_custs = [c for c in eval_custs if c in le_c.classes_]
    if not eval_custs:
        return {"hit_rate": 0.0, "mrr": 0.0, "n_evaluated": 0}

    eval_cidx = le_c.transform(eval_custs)          # (n_eval,)
    prod_col = {p: i for i, p in enumerate(prods_arr)}

    # ── CF scores (batch) ────────────────────────────────────────────────
    all_cf_eval = (U[eval_cidx] * s) @ Vt           # (n_eval, n_prods) — one BLAS call

    # ── CB scores (batch via sparse × dense) ────────────────────────────
    if cos_sim_matrix is not None and prod_idx_map is not None and alpha < 1.0:
        # Build binary purchase indicator rows for eval customers (sparse)
        rows_list, cols_list = [], []
        for local_i, cust in enumerate(eval_custs):
            seen = train_seen.get(cust, set())
            for p in seen:
                ci = prod_idx_map.get(p)
                if ci is not None:
                    rows_list.append(local_i)
                    cols_list.append(ci)
        n_cat = cos_sim_matrix.shape[0]
        if rows_list:
            data = np.ones(len(rows_list))
            purchase_mask = csr_matrix(
                (data, (rows_list, cols_list)),
                shape=(len(eval_custs), n_cat),
            )
            # Row-normalise (L1) so each customer row = average CB similarity
            row_sums = np.asarray(purchase_mask.sum(axis=1)).flatten()
            row_sums[row_sums == 0] = 1.0
            from scipy.sparse import diags as sp_diags
            norm_mask = sp_diags(1.0 / row_sums) @ purchase_mask
            # CB scores aligned to cos_sim ordering (n_eval, n_cat_products)
            cb_cat = norm_mask @ cos_sim_matrix         # (n_eval, 5000) dense

            # Align cb_cat columns to prods_arr ordering
            cat_prods = list(prod_idx_map.keys())       # same order as cos_sim_matrix rows
            cat_prod_arr = np.array(cat_prods)
            # Map cos_sim_matrix product_id -> prods_arr column
            cat_to_svd = np.array([prod_col.get(p, -1) for p in cat_prods])
            valid = cat_to_svd >= 0
            # Initialise cb scores in SVD ordering
            cb_svd = np.zeros((len(eval_custs), n_prods), dtype=np.float32)
            cb_svd[:, cat_to_svd[valid]] = cb_cat[:, valid]
        else:
            cb_svd = np.zeros((len(eval_custs), n_prods), dtype=np.float32)
        use_hybrid = True
    else:
        cb_svd = None
        use_hybrid = False

    # ── Seen masks (batch) ───────────────────────────────────────────────
    # Build a sparse binary seen matrix (n_eval, n_prods) for masking
    seen_rows, seen_cols = [], []
    for local_i, cust in enumerate(eval_custs):
        for p in train_seen.get(cust, set()):
            col = prod_col.get(p)
            if col is not None:
                seen_rows.append(local_i)
                seen_cols.append(col)

    # ── Compute final scores and evaluate ────────────────────────────────
    hits = 0
    rr = 0.0
    n = 0

    # Build seen set per customer for fast top-k filtering
    seen_by_local = {}
    for local_i, cust in enumerate(eval_custs):
        seen_by_local[local_i] = {prod_col[p] for p in train_seen.get(cust, set()) if p in prod_col}

    for local_i, cust in enumerate(eval_custs):
        cf_sc = all_cf_eval[local_i].copy()

        if use_hybrid:
            cb_sc = cb_svd[local_i]
            # Normalise CF to [0,1]
            cf_min, cf_max = cf_sc.min(), cf_sc.max()
            cf_norm = (cf_sc - cf_min) / (cf_max - cf_min + 1e-9)
            # Normalise CB to [0,1]
            cb_min, cb_max = cb_sc.min(), cb_sc.max()
            cb_norm = (cb_sc - cb_min) / (cb_max - cb_min + 1e-9)
            sc = alpha * cf_norm + (1.0 - alpha) * cb_norm
        else:
            sc = cf_sc

        # Mask seen products
        for col in seen_by_local[local_i]:
            sc[col] = -np.inf

        top_k_idx = np.argpartition(sc, -K)[-K:]
        top_k_idx = top_k_idx[np.argsort(sc[top_k_idx])[::-1]]
        top_k = prods_arr[top_k_idx]

        gt = test_novel.get(cust, set())
        if any(p in gt for p in top_k):
            hits += 1
            for rank, p in enumerate(top_k, 1):
                if p in gt:
                    rr += 1.0 / rank
                    break
        n += 1

    return {
        "hit_rate": hits / n if n else 0.0,
        "mrr":      rr   / n if n else 0.0,
        "n_evaluated": n,
    }


def hr_at_k_curve(
    U: np.ndarray,
    s: np.ndarray,
    Vt: np.ndarray,
    inter_data: InteractionData,
    cos_sim_matrix: Optional[np.ndarray],
    prod_idx_map: Optional[dict],
    alpha: float = BEST_ALPHA,
    k_values: List[int] = None,
    n_eval: int = 3000,
) -> pd.DataFrame:
    """Return a DataFrame of hit_rate and MRR at multiple K values for chart generation."""
    if k_values is None:
        k_values = [1, 5, 10, 20, 50]
    rows = []
    for K in k_values:
        metrics = evaluate_hitrate_mrr(
            U, s, Vt, inter_data, cos_sim_matrix, prod_idx_map,
            alpha=alpha, n_eval=n_eval, K=K
        )
        rows.append({"K": K, "hit_rate": metrics["hit_rate"], "mrr": metrics["mrr"]})
    return pd.DataFrame(rows)
