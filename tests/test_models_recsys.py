"""Smoke tests for src/models/train_recsys.py — tiny synthetic fits."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder

from src.features.features_recsys import InteractionData
from src.models import train_recsys as tr


def _tiny_interaction_data() -> InteractionData:
    # 6 customers x 8 products, sparse implicit scores
    rng = np.random.default_rng(0)
    n_c, n_p = 6, 8
    density_idx = rng.choice(n_c * n_p, size=15, replace=False)
    rows, cols, data = [], [], []
    for idx in density_idx:
        rows.append(idx // n_p)
        cols.append(idx % n_p)
        data.append(rng.uniform(0.5, 3.0))
    R = csr_matrix((data, (rows, cols)), shape=(n_c, n_p))

    le_c = LabelEncoder().fit([f"C{i}" for i in range(n_c)])
    le_p = LabelEncoder().fit([f"P{i}" for i in range(n_p)])

    train_seen = {"C0": {"P0", "P1"}, "C1": {"P2"}}
    test_novel = {"C0": {"P3"}, "C2": {"P4"}}

    return InteractionData(
        R_train=R,
        le_c=le_c,
        le_p=le_p,
        train_seen=train_seen,
        test_novel=test_novel,
        n_train=15,
        n_test=5,
        n_eval_customers=len(test_novel),
    )


def test_train_svd_shapes():
    inter = _tiny_interaction_data()
    U, s, Vt = tr.train_svd(inter.R_train, k=3)
    assert U.shape == (6, 3)
    assert s.shape == (3,)
    assert Vt.shape == (3, 8)


def test_predict_scores_masks_seen_items():
    inter = _tiny_interaction_data()
    U, s, Vt = tr.train_svd(inter.R_train, k=3)
    seen_mask = np.array([True, False, False, False, False, False, False, False])
    scores = tr.predict_scores(U, s, Vt, customer_idx=0, seen_mask=seen_mask)
    assert scores.shape == (8,)
    assert scores[0] == -np.inf
    assert np.isfinite(scores[1:]).all()


def test_recommend_returns_top_k_excluding_masked():
    scores = np.array([5.0, -np.inf, 3.0, 1.0, 4.0])
    le_p = LabelEncoder().fit(["P0", "P1", "P2", "P3", "P4"])
    recs = tr.recommend(scores, le_p, K=3)
    assert recs == ["P0", "P4", "P2"]  # ranked descending, P1 excluded (-inf)


def test_recommend_respects_exclude_set():
    scores = np.array([5.0, 4.0, 3.0])
    le_p = LabelEncoder().fit(["P0", "P1", "P2"])
    # K=2 leaves room for exclusion to actually change which items surface;
    # K=3 with only 3 candidates would be forced to include the excluded one anyway.
    recs = tr.recommend(scores, le_p, K=2, exclude_set={"P0"})
    assert "P0" not in recs
    assert recs[0] == "P1"


def test_hybrid_scores_returns_cf_only_when_no_seen_products():
    cf_scores = np.array([1.0, 2.0, 3.0])
    cos_sim = np.eye(3)
    out = tr.hybrid_scores(cf_scores, cos_sim, [], np.array(["P0", "P1", "P2"]), {}, alpha=0.8)
    np.testing.assert_array_equal(out, cf_scores)


def test_hybrid_scores_blends_cf_and_cb():
    cf_scores = np.array([1.0, 2.0, 3.0, -np.inf])
    cos_sim = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.5],
            [0.0, 0.5, 1.0],
        ]
    )
    prod_idx_map = {"P0": 0, "P1": 1, "P2": 2}
    prods_arr = np.array(["P0", "P1", "P2", "P3"])
    out = tr.hybrid_scores(cf_scores, cos_sim, [0], prods_arr, prod_idx_map, alpha=0.5)
    assert out.shape == (4,)
    assert out[3] == -np.inf  # masked entries stay masked


def test_cold_start_recs_uses_preferred_categories_then_falls_back():
    cust_prod = pd.DataFrame({"customer_id": ["C1", "C1"], "product_id": ["P1", "P1"]})
    products = pd.DataFrame(
        {"product_id": ["P1", "P2", "P3"], "category": ["fashion", "fashion", "beauty"]}
    )
    category_pop = {"fashion": ["P1", "P2"], "beauty": ["P3"]}
    recs = tr.cold_start_recs("C1", cust_prod, products, category_pop, K=2)
    # P1 already seen by C1, so category fallback should skip it -> P2, then global fallback -> P3
    assert recs == ["P2", "P3"]


def test_cold_start_recs_unknown_customer_uses_global_fallback():
    cust_prod = pd.DataFrame({"customer_id": [], "product_id": []})
    products = pd.DataFrame({"product_id": ["P1"], "category": ["fashion"]})
    category_pop = {"fashion": ["P1"]}
    recs = tr.cold_start_recs("C_UNKNOWN", cust_prod, products, category_pop, K=1)
    assert recs == ["P1"]


def test_evaluate_hitrate_mrr_cf_only():
    inter = _tiny_interaction_data()
    U, s, Vt = tr.train_svd(inter.R_train, k=3)
    out = tr.evaluate_hitrate_mrr(U, s, Vt, inter, cos_sim_matrix=None, prod_idx_map=None, K=5)
    assert 0.0 <= out["hit_rate"] <= 1.0
    assert 0.0 <= out["mrr"] <= 1.0
    assert out["n_evaluated"] == len(inter.test_novel)


def test_hr_at_k_curve_returns_one_row_per_k():
    inter = _tiny_interaction_data()
    U, s, Vt = tr.train_svd(inter.R_train, k=3)
    curve = tr.hr_at_k_curve(
        U, s, Vt, inter, cos_sim_matrix=None, prod_idx_map=None, k_values=[1, 3]
    )
    assert list(curve["K"]) == [1, 3]
    assert set(curve.columns) == {"K", "hit_rate", "mrr"}
