"""Segmentation V2 model training — nested k-means clustering.

Key design decisions (findings from segmentation_analysis.ipynb):

  Algorithm    : Nested k-means.
                 Stage 1 — isolate non-purchasers (is_non_purchaser=1) as a
                 dedicated segment (Finding 6.B: cluster 0 in k=2 run was 88 %
                 non-purchasers, confirming the flag cleanly separates them).
                 Stage 2 — apply k-means (k=3, n_init=20) on the buyer
                 sub-population (44 432 rows) using V2 log+ratio features
                 reduced by PCA-5.  k=3 on buyers yields sil≈0.38
                 (Finding 3.A / 5.A) with stable assignments (std ≈ 0.000).

  Why nested   : k=2 overall achieves the highest silhouette (0.551) but the
                 two clusters are merely buyers vs non-buyers (9.8 pp churn
                 spread, Finding 6.A).  Nesting preserves the natural stratum
                 boundary while generating actionable buyer sub-segments with
                 distinctly different churn rates and AOV profiles.

  Features     : 13 V2 buyer features — log-transformed volume metrics,
                 behavioural ratios (purchase_rate, order_rate_per_month),
                 session-depth metrics.  ``loyalty_tier`` excluded (50 % null,
                 Finding 1.B).  PCA-5 covers 80 % of variance (Finding 5.C).

  Naming       : Segment names assigned post-hoc by descending revenue so
                 Segment 0 → Non-Purchasers is always cluster_id=3 (highest
                 label), buyers ranked by avg_revenue.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.features.features_segmentation import (
    BUYER_FEATURES,
    SegmentationPipeline,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Number of buyer segments (Finding 3.A / 5.A)
N_BUYER_CLUSTERS: int = 3

# Cluster names assigned by descending avg_revenue among buyers
# Non-purchasers always get the dedicated label below.
BUYER_SEGMENT_NAMES: list[str] = ["Champions", "Mid-Tier Active", "Dormant Buyers"]
NON_PURCHASER_SEGMENT_NAME: str = "Non-Purchasers"

KMEANS_N_INIT: int = 20
KMEANS_RANDOM_STATE: int = 42


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class SegmentationModel:
    """Container for a fitted Segmentation V2 model.

    Attributes
    ----------
    pipeline : SegmentationPipeline
        Fitted scaler + PCA objects.
    kmeans : KMeans
        Fitted k-means on buyer PCA-reduced features.
    n_buyer_clusters : int
        Number of buyer segments (excluding non-purchasers).
    revenue_order : list[int]
        Maps raw k-means label → revenue rank (0 = highest revenue).
    labels_ : np.ndarray
        Final integer cluster labels (0..n_buyer_clusters for buyers,
        n_buyer_clusters for non-purchasers) for the training population.
    silhouette_buyers : float
        Silhouette score computed on buyer PCA features.
    davies_bouldin_buyers : float
        Davies–Bouldin score computed on buyer PCA features.
    buyer_features : list[str]
        Feature names used for clustering.
    """

    pipeline: SegmentationPipeline = field(default_factory=SegmentationPipeline)
    kmeans: KMeans = field(
        default_factory=lambda: KMeans(
            n_clusters=N_BUYER_CLUSTERS,
            n_init=KMEANS_N_INIT,
            random_state=KMEANS_RANDOM_STATE,
        )
    )
    n_buyer_clusters: int = N_BUYER_CLUSTERS
    revenue_order: list[int] = field(default_factory=list)
    labels_: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    silhouette_buyers: float = float("nan")
    davies_bouldin_buyers: float = float("nan")
    buyer_features: list[str] = field(default_factory=lambda: list(BUYER_FEATURES))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_segmentation(
    fe: pd.DataFrame,
    n_buyer_clusters: int = N_BUYER_CLUSTERS,
    n_init: int = KMEANS_N_INIT,
    random_state: int = KMEANS_RANDOM_STATE,
) -> SegmentationModel:
    """Fit the two-stage nested segmentation model.

    Parameters
    ----------
    fe:
        Engineered feature DataFrame (output of ``engineer_features``).
        Must contain ``customer_id``, ``is_non_purchaser``, all
        ``BUYER_FEATURES``, and ``total_net_revenue``.
    n_buyer_clusters:
        Number of k-means clusters within the buyer population.
    n_init:
        Number of k-means initialisations (passed to KMeans).
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    SegmentationModel
        Fitted model with ``labels_`` aligned to ``fe`` row order; non-
        purchasers receive label ``n_buyer_clusters``.
    """
    non_purch_mask = fe["is_non_purchaser"].astype(bool)
    buyer_fe = fe.loc[~non_purch_mask].copy()
    n_total = len(fe)

    # --- Stage 1: fit pipeline on buyers -----------------------------------
    n_comp = pipeline_n_components(buyer_fe)
    pipeline = SegmentationPipeline(pca=PCA(n_components=n_comp, random_state=random_state))
    X_pca = pipeline.fit_transform(buyer_fe)

    # --- Stage 2: k-means on buyer PCA space --------------------------------
    km = KMeans(
        n_clusters=n_buyer_clusters,
        n_init=n_init,
        random_state=random_state,
    )
    buyer_raw_labels = km.fit_predict(X_pca)

    # Compute quality metrics on buyer PCA space
    sil = float(silhouette_score(X_pca, buyer_raw_labels, sample_size=10_000, random_state=0))
    db = float(davies_bouldin_score(X_pca, buyer_raw_labels))

    # --- Revenue-rank relabelling  (Champion=0, Mid-Tier=1, Dormant=2) ------
    buyer_fe = buyer_fe.copy()
    buyer_fe["_raw_label"] = buyer_raw_labels
    mean_revenue = (
        buyer_fe.groupby("_raw_label")["total_net_revenue"].mean().sort_values(ascending=False)
    )
    # revenue_order[i] = new_label for raw_label i
    revenue_order: list[int] = [0] * n_buyer_clusters
    for new_rank, raw_lbl in enumerate(mean_revenue.index):
        revenue_order[int(raw_lbl)] = new_rank
    buyer_renamed = np.array([revenue_order[r] for r in buyer_raw_labels], dtype=int)

    # --- Assemble full-population label array --------------------------------
    # non-purchasers → label = n_buyer_clusters  (e.g. 3 when k_buyer=3)
    labels = np.full(n_total, n_buyer_clusters, dtype=int)
    buyer_idx = np.where(non_purch_mask.values == 0)[0]
    labels[buyer_idx] = buyer_renamed

    model = SegmentationModel(
        pipeline=pipeline,
        kmeans=km,
        n_buyer_clusters=n_buyer_clusters,
        revenue_order=revenue_order,
        labels_=labels,
        silhouette_buyers=sil,
        davies_bouldin_buyers=db,
        buyer_features=list(BUYER_FEATURES),
    )
    return model


def pipeline_n_components(buyer_fe: pd.DataFrame) -> int:
    """Determine PCA n_components to capture 80 % of variance (Finding 5.C).

    Returns at least 2 and at most len(BUYER_FEATURES).
    """
    X = buyer_fe[BUYER_FEATURES].fillna(0).values
    X_scaled = StandardScaler().fit_transform(X)
    pca_full = PCA(random_state=42).fit(X_scaled)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n = int(np.searchsorted(cum_var, 0.80) + 1)
    return max(2, min(n, len(BUYER_FEATURES)))


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def cluster_profiles(
    fe: pd.DataFrame,
    labels: np.ndarray,
    n_buyer_clusters: int = N_BUYER_CLUSTERS,
) -> pd.DataFrame:
    """Return a per-cluster summary DataFrame.

    Parameters
    ----------
    fe:
        Engineered feature DataFrame (must also contain ``churn_flag_90d``
        and ``customer_segment_seed``).
    labels:
        Integer label array aligned to ``fe`` rows.
    n_buyer_clusters:
        Number of buyer clusters (non-purchasers have label
        ``n_buyer_clusters``).

    Returns
    -------
    pd.DataFrame
        Columns: cluster_id, segment_name, n, pct_of_total,
        churn_rate, pct_non_purchaser, median_aov, avg_revenue,
        median_recency, median_order_rate.
    """
    tmp = fe.copy()
    tmp["cluster_id"] = labels

    # Build segment name map
    name_map: dict[int, str] = {i: BUYER_SEGMENT_NAMES[i] for i in range(n_buyer_clusters)}
    name_map[n_buyer_clusters] = NON_PURCHASER_SEGMENT_NAME

    agg = (
        tmp.groupby("cluster_id")
        .agg(
            n=("customer_id", "count"),
            pct_non_purchaser=("is_non_purchaser", "mean"),
            churn_rate=("churn_flag_90d", "mean"),
            median_aov=("avg_order_value", "median"),
            avg_revenue=("total_net_revenue", "mean"),
            median_recency=("recency_days", "median"),
            median_order_rate=("order_rate_per_month", "median"),
        )
        .reset_index()
    )

    n_total = len(fe)
    agg["pct_of_total"] = agg["n"] / n_total
    agg["segment_name"] = agg["cluster_id"].map(name_map)

    col_order = [
        "cluster_id",
        "segment_name",
        "n",
        "pct_of_total",
        "churn_rate",
        "pct_non_purchaser",
        "median_aov",
        "avg_revenue",
        "median_recency",
        "median_order_rate",
    ]
    return agg[col_order].sort_values("cluster_id").reset_index(drop=True)


def segment_metrics(
    model: SegmentationModel,
    profiles: pd.DataFrame,
) -> dict:
    """Return a dict of scalar quality metrics for logging / JSON output."""
    churn_spread = float(profiles["churn_rate"].max() - profiles["churn_rate"].min())
    return {
        "n_total": int(profiles["n"].sum()),
        "n_buyer_clusters": model.n_buyer_clusters,
        "n_total_segments": model.n_buyer_clusters + 1,
        "silhouette_buyers": round(model.silhouette_buyers, 4),
        "davies_bouldin_buyers": round(model.davies_bouldin_buyers, 4),
        "churn_spread_pp": round(churn_spread * 100, 2),
        "cluster_sizes": profiles.set_index("cluster_id")["n"].to_dict(),
        "segment_names": profiles.set_index("cluster_id")["segment_name"].to_dict(),
    }
