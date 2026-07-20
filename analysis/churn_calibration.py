"""Churn — probability calibration and reliability before threshold selection.

The committed churn threshold artefacts (`phase_churn_v2_threshold_selection.csv`)
gate a retention-offer decision at 0.50/0.70 probability, but nothing checks that
the LightGBM scores are calibrated probabilities. A boosted model's raw output is
often over-confident, so a "0.70" cut may not mean a 70% churn risk. This script
reports the reliability of the churn scores and the effect of isotonic
calibration, on the same time-ordered split used everywhere else.

Models on the identical held-out rows:
  - lightgbm_v2_raw        : booster output, uncalibrated
  - lightgbm_v2_isotonic   : booster + isotonic fit on a held-out calibration slice
  - logreg_balanced        : the model the lift analysis says should ship

Outputs:
  outputs/churn_calibration_brier.csv      Brier + log-loss per model
  outputs/churn_calibration_reliability.csv reliability-curve points
  outputs/churn_calibration_reliability.png reliability diagram

Run:
    python -m analysis.churn_calibration
    python -m analysis.churn_calibration --smoke
"""

from __future__ import annotations

import argparse
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

from src.config import OUTPUTS_DIR, PROCESSED_DIR
from src.features.features_churn import (
    build_features,
    feature_set,
    filter_active_population,
    filter_mature_cohort,
    prepare_xy,
    time_ordered_split,
)

PREFIX = "churn_calibration"
MATURITY_DAYS = 90
TEST_SIZE = 0.20
CALIB_FRACTION = 0.15  # tail of the train block held out to fit the calibrator
N_BINS = 10
SEED = 42


def _brier_row(name: str, y_true: np.ndarray, p: np.ndarray) -> dict:
    return {
        "model": name,
        "brier_score": float(brier_score_loss(y_true, p)),
        "log_loss": float(log_loss(y_true, np.clip(p, 1e-6, 1 - 1e-6))),
        "mean_pred": float(p.mean()),
        "observed_rate": float(y_true.mean()),
    }


def main(smoke: bool = False) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    if smoke:
        from analysis.churn_incremental_lift import _make_smoke_frame

        enriched = build_features(_make_smoke_frame())
        print(f"[smoke] synthetic rows: {len(enriched):,}")
    else:
        from src.data.mart_loaders import load_mart

        mart = load_mart("mart_customer_features", processed_dir=PROCESSED_DIR)
        mart = filter_active_population(mart)
        mature = filter_mature_cohort(mart, maturity_days=MATURITY_DAYS)
        enriched = build_features(mature)
        print(f"mature rows: {len(enriched):,}")

    fs = feature_set()
    train_df, test_df = time_ordered_split(enriched, test_size=TEST_SIZE)
    X_train, y_train = prepare_xy(train_df, fs=fs)
    X_test, y_test = prepare_xy(test_df, fs=fs)
    y_test_arr = y_test.to_numpy()

    # Held-out calibration slice = the time tail of the train block.
    cut = int(len(X_train) * (1 - CALIB_FRACTION))
    X_fit, y_fit = X_train.iloc[:cut], y_train.iloc[:cut]
    X_cal, y_cal = X_train.iloc[cut:], y_train.iloc[cut:]

    # ── LightGBM: fit, then isotonic-calibrate on the held-out slice ────────
    from src.models.train_churn import train_churn_model

    inner = int(len(X_fit) * 0.85)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        booster = train_churn_model(
            X_fit.iloc[:inner], y_fit.iloc[:inner], X_fit.iloc[inner:], y_fit.iloc[inner:], fs=fs
        )
    raw_cal = booster.predict_proba(X_cal)
    raw_test = booster.predict_proba(X_test)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_cal, y_cal.to_numpy())
    iso_test = iso.predict(raw_test)

    # ── Logistic-regression reference (fair pipeline, same features) ────────
    from sklearn.linear_model import LogisticRegression

    from analysis.churn_incremental_lift import _sklearn_pipeline

    numeric = [c for c in fs.feature_columns if c not in fs.categorical_columns]
    logreg = _sklearn_pipeline(
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED),
        numeric,
        list(fs.categorical_columns),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logreg.fit(X_train, y_train)
    logreg_test = logreg.predict_proba(X_test)[:, 1]

    preds = {
        "lightgbm_v2_raw": raw_test,
        "lightgbm_v2_isotonic": iso_test,
        "logreg_balanced": logreg_test,
    }

    # ── Brier / log-loss table ──────────────────────────────────────────────
    brier = pd.DataFrame([_brier_row(n, y_test_arr, p) for n, p in preds.items()])

    # ── Reliability-curve points ─────────────────────────────────────────────
    rel_frames = []
    for name, p in preds.items():
        frac_pos, mean_pred = calibration_curve(y_test_arr, p, n_bins=N_BINS, strategy="quantile")
        rel_frames.append(
            pd.DataFrame(
                {"model": name, "mean_predicted": mean_pred, "observed_fraction": frac_pos}
            )
        )
    reliability = pd.concat(rel_frames, ignore_index=True)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfectly calibrated")
    colours = {
        "lightgbm_v2_raw": "#C0392B",
        "lightgbm_v2_isotonic": "#2E86AB",
        "logreg_balanced": "#27AE60",
    }
    for name, p in preds.items():
        sub = reliability[reliability["model"] == name]
        ax.plot(
            sub["mean_predicted"],
            sub["observed_fraction"],
            marker="o",
            color=colours[name],
            label=f"{name} (Brier {brier_score_loss(y_test_arr, p):.3f})",
        )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed churn fraction")
    ax.set_title("Churn — reliability diagram (time-ordered test)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if not smoke:
        brier_path = OUTPUTS_DIR / f"{PREFIX}_brier.csv"
        rel_path = OUTPUTS_DIR / f"{PREFIX}_reliability.csv"
        png_path = OUTPUTS_DIR / f"{PREFIX}_reliability.png"
        brier.to_csv(brier_path, index=False)
        reliability.to_csv(rel_path, index=False)
        fig.savefig(png_path, dpi=130)
        print(f"Wrote: {brier_path}\nWrote: {rel_path}\nWrote: {png_path}")
    plt.close(fig)

    print("\nBrier / log-loss on the time-ordered test set:")
    print(brier.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run on synthetic data, no marts")
    args = parser.parse_args()
    main(smoke=args.smoke)
