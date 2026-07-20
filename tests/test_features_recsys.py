"""Unit tests for src/features/features_recsys.py — data-independent, in-memory frames."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import features_recsys as fr


def test_implicit_score_formula():
    df = pd.DataFrame(
        {
            "quantity": [2, 0],
            "view_count": [3, 0],
            "wish_count": [1, 0],
            "pos_review": [1, 0],
        }
    )
    score = fr._implicit_score(df)
    expected0 = (
        np.log1p(2) * fr.W_BUY
        + np.log1p(3) * fr.W_VIEW
        + np.log1p(1) * fr.W_WISH
        + 1 * fr.W_POS_REV
    )
    assert score.iloc[0] == expected0
    assert score.iloc[1] == 0.0


def test_implicit_score_handles_missing_columns():
    df = pd.DataFrame({"quantity": [1]})  # views/wishes/pos_review absent
    score = fr._implicit_score(df)
    assert score.iloc[0] == np.log1p(1) * fr.W_BUY


def _products_frame() -> pd.DataFrame:
    return pd.DataFrame({"product_id": ["P1", "P2", "P3"]})


def test_build_interaction_matrix_restricts_to_train_test_overlap():
    cust_prod = pd.DataFrame(
        {
            "customer_id": ["C1", "C1", "C2"],  # C2 only has a train purchase, no test purchase
            "product_id": ["P1", "P1", "P2"],
            "order_date": pd.to_datetime(["2025-10-01", "2025-11-15", "2025-10-05"]),
            "quantity": [1, 1, 1],
        }
    )
    empty_events = pd.DataFrame(columns=["customer_id", "product_id"])
    empty_reviews = pd.DataFrame(columns=["customer_id", "product_id"])
    products = _products_frame()

    inter = fr.build_interaction_matrix(
        cust_prod, empty_events, empty_events, empty_reviews, products
    )

    # C1 bought in both train (<SPLIT_DATE) and test (>=SPLIT_DATE); C2 only in train.
    assert "C1" in inter.le_c.classes_
    assert "C2" not in inter.le_c.classes_
    assert inter.R_train.shape == (1, len(products))


def test_build_content_matrix_shape_and_index_map():
    products = pd.DataFrame(
        {
            "product_id": ["P1", "P2"],
            "category": ["fashion", "beauty"],
            "subcategory": ["fas_0", "bea_1"],
            "brand_type": ["national", "owned_brand"],
            "price_tier": ["mid", "entry"],
        }
    )
    pa = pd.DataFrame(
        {
            "product_id": ["P1", "P2"],
            "recommendation_embedding_group": ["FA_0", "BE_1"],
            "eco_flag": [True, False],
            "bundle_candidate_flag": [False, True],
        }
    )
    matrix, idx_map = fr.build_content_matrix(products, pa)
    assert matrix.shape[0] == 2
    assert idx_map == {"P1": 0, "P2": 1}


def test_category_top_n_popularity_ranks_by_purchase_count():
    cust_prod = pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3"],
            "product_id": ["P1", "P1", "P2"],
        }
    )
    products = pd.DataFrame({"product_id": ["P1", "P2"], "category": ["fashion", "fashion"]})
    pop = fr.category_top_n_popularity(cust_prod, products, N=1)
    assert pop["fashion"] == ["P1"]  # P1 bought twice, P2 once


def test_get_customer_categories_returns_top_purchased_categories():
    cust_prod = pd.DataFrame(
        {
            "customer_id": ["C1", "C1", "C1", "C2"],
            "product_id": ["P1", "P1", "P2", "P3"],
        }
    )
    products = pd.DataFrame(
        {"product_id": ["P1", "P2", "P3"], "category": ["fashion", "beauty", "home"]}
    )
    cats = fr.get_customer_categories("C1", cust_prod, products, top_n=2)
    assert cats[0] == "fashion"  # bought twice, most frequent
    assert len(cats) == 2


def test_get_customer_categories_empty_for_unknown_customer():
    cust_prod = pd.DataFrame({"customer_id": ["C1"], "product_id": ["P1"]})
    products = pd.DataFrame({"product_id": ["P1"], "category": ["fashion"]})
    assert fr.get_customer_categories("C_UNKNOWN", cust_prod, products) == []
