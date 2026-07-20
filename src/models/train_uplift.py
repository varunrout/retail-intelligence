"""Uplift V2 model training — X-learner with LightGBM.

Algorithm (Finding 4.C / §4):
  Stage 1 — fit separate outcome models for treated and control arms,
             stratified per campaign (Finding 2.C / §2) to prevent
             cross-campaign confounding.
  Stage 2 — regress pseudo-outcomes onto all examples to produce ITE scores.
  Final   — blend Stage 2 scores using the per-campaign propensity weight g_c.

Full derivation in analysis_notebooks/uplift_analysis.ipynb §7 design.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.features.features_uplift import CAMPAIGN_KEY, FeatureSet, feature_set

# ── Stage 1: outcome model hypers ─────────────────────────────────────────
STAGE1_PARAMS: dict = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
    "deterministic": True,
}

# Stage 2 regresses a continuous pseudo-outcome (CATE), so use regression.
STAGE2_PARAMS: dict = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
    "deterministic": True,
}

DEFAULT_NUM_BOOST_ROUND = 1000
DEFAULT_EARLY_STOPPING_ROUNDS = 60


@dataclass
class XLearnerModel:
    """Container for a fitted X-learner.

    Attributes
    ----------
    tau1 : lgb.Booster
        Stage-2 regressor trained on treated pseudo-outcomes.
    tau0 : lgb.Booster
        Stage-2 regressor trained on control pseudo-outcomes.
    campaign_propensity : dict[str, float]
        Per-campaign treatment fraction g_c.  Used as blending weight.
    feature_set : FeatureSet
        The feature set used during training.
    stage1_best_iters : dict
        Best iteration counts from Stage 1 models, keyed by
        ``"mu1_{campaign_id}"`` / ``"mu0_{campaign_id}"``.
    stage2_best_iters : dict
        Best iteration counts from Stage 2 models, keyed by
        ``"tau1"`` / ``"tau0"``.
    """

    tau1: lgb.Booster
    tau0: lgb.Booster
    campaign_propensity: dict[str, float]
    feature_set: FeatureSet
    stage1_best_iters: dict = field(default_factory=dict)
    stage2_best_iters: dict = field(default_factory=dict)

    def predict_ite(self, X: pd.DataFrame, campaigns: pd.Series) -> np.ndarray:
        """Return individual treatment effect estimates for each row.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (same feature set used at training).
        campaigns : pd.Series
            Campaign identifier for each row (used to look up blending weight).
        """
        X = _apply_categorical_dtypes(X, self.feature_set)
        tau1_scores = self.tau1.predict(X, num_iteration=self.stage2_best_iters.get("tau1"))
        tau0_scores = self.tau0.predict(X, num_iteration=self.stage2_best_iters.get("tau0"))

        # g_c = treatment fraction within campaign; fallback to 0.5 if unseen
        g_c = campaigns.map(self.campaign_propensity).fillna(0.5).values
        ite = g_c * tau1_scores + (1.0 - g_c) * tau0_scores
        return ite.astype(np.float32)


def _apply_categorical_dtypes(X: pd.DataFrame, fs: FeatureSet) -> pd.DataFrame:
    X = X.copy()
    for col in fs.categorical_columns:
        if col in X.columns:
            X[col] = X[col].astype("category")
    return X


def _lgb_dataset(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    cat_cols: list[str],
    *,
    reference: lgb.Dataset | None = None,
    weight: np.ndarray | None = None,
) -> lgb.Dataset:
    return lgb.Dataset(
        X,
        label=y,
        categorical_feature=cat_cols,
        weight=weight,
        reference=reference,
        free_raw_data=False,
    )


def _train_lgb(
    train_ds: lgb.Dataset,
    valid_ds: lgb.Dataset,
    params: dict,
    *,
    num_boost_round: int = DEFAULT_NUM_BOOST_ROUND,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
) -> lgb.Booster:
    return lgb.train(
        params,
        train_ds,
        num_boost_round=num_boost_round,
        valid_sets=[train_ds, valid_ds],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )


def train_xlearner(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    *,
    fs: FeatureSet | None = None,
    stage1_params: dict | None = None,
    stage2_params: dict | None = None,
    num_boost_round: int = DEFAULT_NUM_BOOST_ROUND,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
    tc_weight_cap: float = 10.0,
) -> XLearnerModel:
    """Fit a full X-learner.

    Parameters
    ----------
    train_df, valid_df
        DataFrames that include features, label (``response_flag_30d``),
        ``treatment_flag``, and ``campaign_id``.  The validation split is
        used only for early-stopping in Stage 1 and Stage 2.
    tc_weight_cap
        Maximum up-weight factor applied to control rows in Stage 1 μ₀ fit
        (Finding 1.B — T/C ratio up to 12.4:1).  Capped to avoid extreme
        weights destabilising the loss.
    """
    from src.features.features_uplift import LABEL, TREATMENT_KEY

    fs = fs or feature_set()
    s1_params = {**STAGE1_PARAMS, **(stage1_params or {})}
    s2_params = {**STAGE2_PARAMS, **(stage2_params or {})}

    cat_cols_present = [c for c in fs.categorical_columns if c in train_df.columns]
    avail_feats = [c for c in fs.feature_columns if c in train_df.columns]

    X_tr = _apply_categorical_dtypes(train_df[avail_feats], fs)
    y_tr = train_df[LABEL].astype(float).values
    t_tr = train_df[TREATMENT_KEY].astype(int).values
    c_tr = train_df[CAMPAIGN_KEY].astype(str)

    X_va = _apply_categorical_dtypes(valid_df[avail_feats], fs)
    y_va = valid_df[LABEL].astype(float).values
    t_va = valid_df[TREATMENT_KEY].astype(int).values

    # ── Per-campaign propensity (treatment fraction) ────────────────────────
    propensity: dict[str, float] = train_df.groupby(CAMPAIGN_KEY)[TREATMENT_KEY].mean().to_dict()

    # ── Stage 1: fit μ₁ and μ₀ per campaign ────────────────────────────────
    print("Stage 1: fitting per-campaign outcome models …")

    # Predictions on the full training set from Stage 1 cross-fitting
    mu1_hat_tr = np.zeros(len(X_tr), dtype=np.float64)
    mu0_hat_tr = np.zeros(len(X_tr), dtype=np.float64)
    # Also build validation predictions so Stage 2 validation pseudo-outcomes are correct
    mu1_hat_va = np.zeros(len(X_va), dtype=np.float64)
    mu0_hat_va = np.zeros(len(X_va), dtype=np.float64)
    stage1_best_iters: dict[str, int] = {}
    # Store Stage 1 boosters so we can cross-predict on validation set
    _mu1_boosters: dict[str, lgb.Booster] = {}
    _mu0_boosters: dict[str, lgb.Booster] = {}

    campaigns_in_train = c_tr.unique().tolist()

    for cmp in sorted(campaigns_in_train):
        mask_tr = (c_tr == cmp).values
        mask_va = (valid_df[CAMPAIGN_KEY].astype(str) == cmp).values

        # Keep a full-campaign snapshot for cross-predicting at end of Stage 1
        X_cmp_full = X_tr[mask_tr]
        y_cmp_all = y_tr[mask_tr]
        t_cmp_all = t_tr[mask_tr]

        if mask_va.sum() == 0:
            # No validation rows for this campaign — use small hold-out from train
            n_hold = max(1, int(mask_tr.sum() * 0.15))
            X_v_cmp = X_cmp_full.iloc[-n_hold:]
            y_v_cmp = y_cmp_all[-n_hold:]
            t_v_cmp = t_cmp_all[-n_hold:]
            X_cmp_all = X_cmp_full.iloc[:-n_hold]
            y_cmp_all = y_cmp_all[:-n_hold]
            t_cmp_all = t_cmp_all[:-n_hold]
        else:
            X_cmp_all = X_cmp_full
            X_v_cmp = X_va[mask_va]
            y_v_cmp = y_va[mask_va]
            t_v_cmp = t_va[mask_va]

        treat_mask_tr = t_cmp_all == 1
        ctrl_mask_tr = t_cmp_all == 0
        treat_mask_va = t_v_cmp == 1
        ctrl_mask_va = t_v_cmp == 0

        # ── μ₁: treatment arm ──────────────────────────────────────────────
        if treat_mask_tr.sum() >= 20 and treat_mask_va.sum() >= 5:
            ds1_tr = _lgb_dataset(
                X_cmp_all[treat_mask_tr], y_cmp_all[treat_mask_tr], cat_cols_present
            )
            ds1_va = _lgb_dataset(
                X_v_cmp[treat_mask_va], y_v_cmp[treat_mask_va], cat_cols_present, reference=ds1_tr
            )
            mu1_bst = _train_lgb(
                ds1_tr,
                ds1_va,
                s1_params,
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds,
            )
            stage1_best_iters[f"mu1_{cmp}"] = mu1_bst.best_iteration or num_boost_round
            # Score ALL campaign rows with μ₁ (use full snapshot, not trimmed fit set)
            mu1_hat_tr[mask_tr] = mu1_bst.predict(
                X_cmp_full, num_iteration=stage1_best_iters[f"mu1_{cmp}"]
            )
            _mu1_boosters[cmp] = mu1_bst
        else:
            # Fallback: use global treatment mean for this campaign
            mu1_hat_tr[mask_tr] = float(y_tr[t_tr == 1].mean()) if (t_tr == 1).any() else 0.5

        # ── μ₀: control arm (up-weighted for T/C imbalance) ───────────────
        if ctrl_mask_tr.sum() >= 10 and ctrl_mask_va.sum() >= 3:
            g_c = float(propensity.get(cmp, 0.5))
            # weight = (1-g_c)/g_c capped at tc_weight_cap
            tc_ratio = (1.0 - g_c) / g_c if g_c > 0 else 1.0
            ctrl_weight = min(tc_ratio, tc_weight_cap)
            w_ctrl = np.full(ctrl_mask_tr.sum(), ctrl_weight)

            ds0_tr = _lgb_dataset(
                X_cmp_all[ctrl_mask_tr], y_cmp_all[ctrl_mask_tr], cat_cols_present, weight=w_ctrl
            )
            ds0_va = _lgb_dataset(
                X_v_cmp[ctrl_mask_va], y_v_cmp[ctrl_mask_va], cat_cols_present, reference=ds0_tr
            )
            mu0_bst = _train_lgb(
                ds0_tr,
                ds0_va,
                s1_params,
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds,
            )
            stage1_best_iters[f"mu0_{cmp}"] = mu0_bst.best_iteration or num_boost_round
            mu0_hat_tr[mask_tr] = mu0_bst.predict(
                X_cmp_full, num_iteration=stage1_best_iters[f"mu0_{cmp}"]
            )
            _mu0_boosters[cmp] = mu0_bst
        else:
            mu0_hat_tr[mask_tr] = float(y_tr[t_tr == 0].mean()) if (t_tr == 0).any() else 0.5

        n_cmp = mask_tr.sum()
        print(f"  {cmp}: n={n_cmp:,}  T={treat_mask_tr.sum():,}  C={ctrl_mask_tr.sum():,}")

    # ── Stage 2: pseudo-outcome regression ─────────────────────────────────
    print("Stage 2: pseudo-outcome regression …")

    # Treated rows: D̃ = Y - μ̂₀(X)
    treat_idx = t_tr == 1
    ctrl_idx = t_tr == 0

    D_treated = y_tr[treat_idx] - mu0_hat_tr[treat_idx]
    D_control = mu1_hat_tr[ctrl_idx] - y_tr[ctrl_idx]

    # Validation pseudo-outcomes for Stage 2 early stopping
    # Use stored Stage 1 boosters for accurate predictions on the validation set.
    t_va_arr = t_va.astype(int)
    c_va = valid_df[CAMPAIGN_KEY].astype(str)

    for cmp in sorted(campaigns_in_train):
        mask = (c_va == cmp).values
        if mask.sum() == 0:
            continue
        X_v_cmp_all = X_va[mask]
        g_c = float(propensity.get(cmp, 0.5))
        if cmp in _mu1_boosters:
            mu1_hat_va[mask] = _mu1_boosters[cmp].predict(
                X_v_cmp_all, num_iteration=stage1_best_iters.get(f"mu1_{cmp}")
            )
        else:
            mu1_hat_va[mask] = g_c
        if cmp in _mu0_boosters:
            mu0_hat_va[mask] = _mu0_boosters[cmp].predict(
                X_v_cmp_all, num_iteration=stage1_best_iters.get(f"mu0_{cmp}")
            )
        else:
            mu0_hat_va[mask] = g_c

    treat_va_idx = t_va_arr == 1
    ctrl_va_idx = t_va_arr == 0
    D_treated_va = y_va[treat_va_idx] - mu0_hat_va[treat_va_idx]
    D_control_va = mu1_hat_va[ctrl_va_idx] - y_va[ctrl_va_idx]

    X_tr_treat = X_tr[treat_idx]
    X_tr_ctrl = X_tr[ctrl_idx]
    X_va_treat = X_va[treat_va_idx]
    X_va_ctrl = X_va[ctrl_va_idx]

    # τ₁: trained on treated pseudo-outcomes
    ds2_tau1_tr = _lgb_dataset(X_tr_treat, D_treated, cat_cols_present)
    ds2_tau1_va = _lgb_dataset(X_va_treat, D_treated_va, cat_cols_present, reference=ds2_tau1_tr)
    tau1_bst = _train_lgb(
        ds2_tau1_tr,
        ds2_tau1_va,
        s2_params,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
    )

    # τ₀: trained on control pseudo-outcomes
    ds2_tau0_tr = _lgb_dataset(X_tr_ctrl, D_control, cat_cols_present)
    ds2_tau0_va = _lgb_dataset(X_va_ctrl, D_control_va, cat_cols_present, reference=ds2_tau0_tr)
    tau0_bst = _train_lgb(
        ds2_tau0_tr,
        ds2_tau0_va,
        s2_params,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
    )

    stage2_best_iters = {
        "tau1": tau1_bst.best_iteration or num_boost_round,
        "tau0": tau0_bst.best_iteration or num_boost_round,
    }
    print(f"  τ₁ best iter={stage2_best_iters['tau1']}  τ₀ best iter={stage2_best_iters['tau0']}")

    return XLearnerModel(
        tau1=tau1_bst,
        tau0=tau0_bst,
        campaign_propensity=propensity,
        feature_set=fs,
        stage1_best_iters=stage1_best_iters,
        stage2_best_iters=stage2_best_iters,
    )


# ── Evaluation helpers ─────────────────────────────────────────────────────


def evaluate_uplift(
    df: pd.DataFrame,
    ite_scores: np.ndarray,
    *,
    n_deciles: int = 10,
    label: str = "xlearner_v2",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(summary_row, decile_table)``.

    Summary row mirrors ``phase7_uplift_model_comparison.csv`` schema.
    Decile table mirrors ``phase7_uplift_decile_summary.csv`` schema.
    """
    from src.features.features_uplift import LABEL, TREATMENT_KEY

    df = df.copy()
    df["_ite"] = ite_scores
    t = df[TREATMENT_KEY].astype(int)
    y = df[LABEL].astype(int)

    # ── Overall ATE on test set ─────────────────────────────────────────────
    ate_test = float(y[t == 1].mean() - y[t == 0].mean())

    # ── Decile evaluation ──────────────────────────────────────────────────
    # Sort by ITE descending, split into n_deciles buckets, measure observed
    # uplift (mean(Y|T=1) - mean(Y|T=0)) within each bucket.
    df = df.sort_values("_ite", ascending=False).reset_index(drop=True)
    df["_decile"] = pd.qcut(df.index, q=n_deciles, labels=False) + 1

    decile_rows = []
    cumulative_treat = 0
    cumulative_ctrl = 0
    cumulative_lift = 0.0
    for dec in range(1, n_deciles + 1):
        mask = df["_decile"] == dec
        sub = df[mask]
        n_t = int((sub[TREATMENT_KEY] == 1).sum())
        n_c = int((sub[TREATMENT_KEY] == 0).sum())
        r_t = float(sub.loc[sub[TREATMENT_KEY] == 1, LABEL].mean()) if n_t else np.nan
        r_c = float(sub.loc[sub[TREATMENT_KEY] == 0, LABEL].mean()) if n_c else np.nan
        obs_uplift = (r_t - r_c) if (n_t and n_c) else np.nan
        mean_ite = float(sub["_ite"].mean())
        cumulative_treat += n_t
        cumulative_ctrl += n_c
        if not np.isnan(obs_uplift):
            cumulative_lift += obs_uplift
        decile_rows.append(
            {
                "model": label,
                "decile": dec,
                "n_treatment": n_t,
                "n_control": n_c,
                "response_rate_treatment": r_t,
                "response_rate_control": r_c,
                "observed_uplift": obs_uplift,
                "mean_ite_score": mean_ite,
                "cumulative_observed_uplift": cumulative_lift / dec,
            }
        )
    decile_df = pd.DataFrame(decile_rows)

    # ── Qini-like area (cumulative incremental responses vs random) ─────────
    # Count incremental responses above random in each decile bucket
    incr_responses = []
    for dec in range(1, n_deciles + 1):
        mask = df["_decile"] == dec
        sub_t = df[mask & (df[TREATMENT_KEY] == 1)]
        sub_c = df[mask & (df[TREATMENT_KEY] == 0)]
        if len(sub_t) and len(sub_c):
            incr = float(sub_t[LABEL].mean() - sub_c[LABEL].mean()) * len(sub_t)
        else:
            incr = 0.0
        incr_responses.append(incr)
    # Cumulative normalised (Qini-like area)
    cumulative_incr = np.cumsum(incr_responses)
    random_cumulative = np.linspace(0, sum(incr_responses), n_deciles)
    qini_area = float(np.trapz(cumulative_incr - random_cumulative))

    # ── Top-decile and top-half observed uplift ────────────────────────────
    top1 = float(decile_df.loc[decile_df["decile"] == 1, "observed_uplift"].iloc[0])
    top3 = float(decile_df[decile_df["decile"] <= 3]["observed_uplift"].mean())
    top5 = float(decile_df[decile_df["decile"] <= 5]["observed_uplift"].mean())

    # ── Spearman rank-correlation (ITE vs observed outcome gap) ───────────
    # Bin by ITE decile and correlate mean ITE with observed uplift
    valid_rows = decile_df.dropna(subset=["observed_uplift"])
    if len(valid_rows) >= 3:
        spearman_r, _ = spearmanr(valid_rows["mean_ite_score"], valid_rows["observed_uplift"])
    else:
        spearman_r = np.nan

    summary = pd.DataFrame(
        [
            {
                "model": label,
                "split_strategy": "time_ordered_80_20_by_assignment_datetime",
                "approved_feature_count": len([c for c in feature_set().feature_columns]),
                "test_rows": int(len(df)),
                "overall_ate_test": ate_test,
                "top1_decile_observed_uplift": top1,
                "top3_decile_observed_uplift": top3,
                "top5_decile_observed_uplift": top5,
                "qini_like_area": qini_area,
                "spearman_rank_corr": spearman_r,
            }
        ]
    )
    return summary, decile_df


def feature_importance_table(model: XLearnerModel, *, top_n: int = 30) -> pd.DataFrame:
    """Return Stage-2 feature importances (τ₁ and τ₀ averaged)."""
    rows = []
    for stage, bst in [("tau1", model.tau1), ("tau0", model.tau0)]:
        imp = pd.DataFrame(
            {
                "feature": bst.feature_name(),
                "importance_gain": bst.feature_importance(importance_type="gain"),
                "importance_split": bst.feature_importance(importance_type="split"),
                "stage": stage,
            }
        )
        rows.append(imp)
    combined = pd.concat(rows)
    agg = (
        combined.groupby("feature")[["importance_gain", "importance_split"]]
        .mean()
        .sort_values("importance_gain", ascending=False)
        .head(top_n)
        .reset_index()
    )
    return agg
