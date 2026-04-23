"""Segmentation V2 feature engineering.

Decisions encoded here are documented and justified in
``analysis_notebooks/segmentation_analysis.ipynb``. Each choice has a
finding number tying it back to the analysis.

V2 improvements over phase9:
  Finding 1.A  Non-purchaser flag — 11.1% of customers have no purchase history.
               Phase9 silently imputed 0; V2 adds an explicit ``is_non_purchaser``
               binary feature so k-means can treat them as a coherent stratum.
  Finding 1.B  Loyalty-tier excluded — 50% null (non-enrolled ≠ low loyalty).
               Column retained in raw data for post-hoc profiling only.
  Finding 1.C  Log1p transforms — total_orders skew=0.82, total_net_revenue
               skew=1.85, recency_days skew=3.20. Standardising raw values
               lets high-volume outliers dominate centroid updates.
  Finding 2.A  Volume collinearity — 11 pairs |r|>0.80 (total_sessions ≈
               sessions_with_purchase r=0.95). Log-transform brings skew down;
               PCA decorrelates after standardisation.
  Finding 4.B  Behavioural ratios — ``purchase_rate`` (session CVR) and
               ``order_rate_per_month`` are orthogonal to raw volume and
               capture browse-to-buy efficiency and purchase velocity.
  Finding 5.C  PCA components — 5 components capture 80 % of variance on V2
               features (phase9 needed 6 with raw skewed features).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Feature lists
# ---------------------------------------------------------------------------

# Columns that will be log1p-transformed before scaling (Finding 1.C / 4.A)
LOG_COLS: list[str] = [
    "total_orders",
    "total_net_revenue",
    "avg_order_value",
    "total_units",
    "total_sessions",
    "recency_days",
    "tenure_days",
    "total_returns",
]

# Columns where nulls mean "never purchased" — impute to 0 (Finding 1.A)
PURCHASE_COLS: list[str] = [
    "total_orders",
    "total_net_revenue",
    "avg_order_value",
    "avg_basket_size",
    "revenue_per_order",
    "total_units",
    "avg_item_discount_pct",
    "total_returns",
    "return_rate_per_unit",
    "online_order_share",
    "store_order_share",
    "avg_item_margin",
    "total_discount_amount",
    "total_refund_amount",
]

# Session columns — tiny null rate (0.4%), fill with median
SESSION_COLS: list[str] = [
    "total_sessions",
    "avg_session_minutes",
    "avg_pages_viewed",
    "sessions_add_to_cart",
    "sessions_with_purchase",
]

# V2 buyer feature set used for clustering the buyer sub-population
# (Applied AFTER non-purchasers are split off — Finding 6.B / nested approach)
BUYER_FEATURES: list[str] = [
    "log_total_orders",        # F1.C / F4.A: volume, log-transformed
    "log_total_net_revenue",   # revenue, log-transformed
    "avg_order_value",         # basket value (moderate skew, kept raw)
    "log_recency_days",        # F4.A: recency, log-transformed
    "log_tenure_days",         # customer age, log-transformed
    "avg_item_discount_pct",   # discount affinity
    "return_rate_per_unit",    # returns behaviour
    "log_total_sessions",      # browse engagement, log-transformed
    "avg_session_minutes",     # session depth
    "avg_pages_viewed",        # browse intensity
    "purchase_rate",           # F4.B: sessions_with_purchase / total_sessions
    "online_order_share",      # channel preference (web vs in-store)
    "order_rate_per_month",    # F4.B: purchase velocity
]

# PCA settings (Finding 5.C)
PCA_N_COMPONENTS: int = 5   # 80 % variance on V2 log-feature set


# ---------------------------------------------------------------------------
# Engineering helpers
# ---------------------------------------------------------------------------

def engineer_features(cust: pd.DataFrame) -> pd.DataFrame:
    """Return an enriched copy of *cust* with V2 derived columns.

    Parameters
    ----------
    cust:
        Raw ``mart_customer_features`` DataFrame (50 000 rows × ~40 cols).

    Returns
    -------
    pd.DataFrame
        Input plus derived columns:
        ``is_non_purchaser``, ``log_*``, ``purchase_rate``,
        ``order_rate_per_month``, ``campaign_cvr``.
    """
    fe = cust.copy()

    # 1. Non-purchaser flag — before any imputation (Finding 1.A / 4.C)
    fe["is_non_purchaser"] = fe["total_orders"].isnull().astype("int8")

    # 2. Purchase-column null → 0 (non-purchasers legitimately have 0)
    for col in PURCHASE_COLS:
        fe[col] = fe[col].fillna(0)

    # 3. recency_days for true non-purchasers → tenure_days (maximally dormant)
    fe["recency_days"] = fe["recency_days"].fillna(fe["tenure_days"])

    # 4. Session columns — fill with median
    for col in SESSION_COLS:
        fe[col] = fe[col].fillna(fe[col].median())

    # 5. Log1p transforms (Finding 1.C / 4.A)
    for col in LOG_COLS:
        fe[f"log_{col}"] = np.log1p(fe[col])

    # 6. Behavioural ratios (Finding 4.B)
    fe["purchase_rate"] = np.where(
        fe["total_sessions"] > 0,
        fe["sessions_with_purchase"] / fe["total_sessions"],
        0.0,
    )
    fe["order_rate_per_month"] = np.where(
        fe["tenure_days"] > 0,
        fe["total_orders"] / (fe["tenure_days"] / 30.0),
        0.0,
    )

    return fe


# ---------------------------------------------------------------------------
# Dataclass for fitted pipeline
# ---------------------------------------------------------------------------

@dataclass
class SegmentationPipeline:
    """Holds the fitted scaler + PCA objects for the buyer sub-population.

    Attributes
    ----------
    scaler : StandardScaler
        Fitted on buyer rows using ``BUYER_FEATURES``.
    pca : PCA
        Fitted PCA (``PCA_N_COMPONENTS`` components).
    buyer_features : list[str]
        Feature names passed to the scaler.
    pca_n_components : int
        Number of PCA components.
    """

    scaler: StandardScaler = field(default_factory=StandardScaler)
    pca: PCA = field(default_factory=lambda: PCA(
        n_components=PCA_N_COMPONENTS, random_state=42
    ))
    buyer_features: list[str] = field(default_factory=lambda: list(BUYER_FEATURES))
    pca_n_components: int = PCA_N_COMPONENTS

    def fit_transform(self, buyer_fe: pd.DataFrame) -> np.ndarray:
        """Fit scaler + PCA on buyer rows and return PCA-reduced matrix.

        Parameters
        ----------
        buyer_fe:
            Rows from engineered feature DataFrame where ``is_non_purchaser == 0``.

        Returns
        -------
        np.ndarray, shape (n_buyers, pca_n_components)
        """
        X = buyer_fe[self.buyer_features].fillna(0).values
        X_scaled = self.scaler.fit_transform(X)
        return self.pca.fit_transform(X_scaled)

    def transform(self, buyer_fe: pd.DataFrame) -> np.ndarray:
        """Apply fitted pipeline to new buyer rows (inference path).

        Parameters
        ----------
        buyer_fe:
            Rows from engineered feature DataFrame where ``is_non_purchaser == 0``.

        Returns
        -------
        np.ndarray, shape (n_rows, pca_n_components)
        """
        X = buyer_fe[self.buyer_features].fillna(0).values
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)
