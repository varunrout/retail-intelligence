"""Forecast — naive benchmarks the V2 comparison never included.

`phase_forecast_v2_vs_baseline.csv` reports LightGBM-Tweedie V2 beating the phase8
Random Forest by ~1.4% MAE and calls it a win. Two things are missing: (1) a naive
benchmark to show whether either tree model has real signal, and (2) an honest read
that a 1.4% gap over RF is not a meaningful improvement.

This scores two naive forecasts on the SAME week-ordered test split (weeks >=
2025-08-04) as the V2 model:
  - naive_last_week   : this week = last week's units (lag_1w_units_sold)
  - naive_seasonal_4w : this week = units 4 weeks ago (lag_4w_units_sold)

Note rolling_4w_avg_units is NOT used as a naive forecast: it includes the current
week and is flagged as leakage in features_forecast.FORBIDDEN_FEATURES.

Run:
    python -m analysis.forecast_naive_baseline
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import OUTPUTS_DIR, PROCESSED_DIR
from src.features import features_forecast as ff

# Committed V2 headline (outputs/phase_forecast_v2_model_comparison.csv)
V2_MAE = 1.2336
V2_SMAPE = 0.4562
# Committed RF baseline (outputs/phase_forecast_v2_vs_baseline.csv)
RF_MAE = 1.2512
RF_SMAPE = 0.4677


def _mae(a: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(a - p)))


def _smape(a: np.ndarray, p: np.ndarray) -> float:
    denom = np.abs(a) + np.abs(p)
    m = denom != 0
    return float(np.mean(2 * np.abs(a - p)[m] / denom[m]))


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # lag_1w / lag_4w already exist in the mart, so no feature build is needed.
    df = pd.read_csv(PROCESSED_DIR / "mart_product_demand.csv")
    df[ff.TIME_KEY] = pd.to_datetime(df[ff.TIME_KEY])
    _, test_df = ff.week_time_split(df)
    y = test_df[ff.LABEL].to_numpy(dtype=float)

    rows = [
        {"model": "lgbm_tweedie_v2", "mae": V2_MAE, "smape": V2_SMAPE, "source": "committed"},
        {"model": "rf_baseline", "mae": RF_MAE, "smape": RF_SMAPE, "source": "committed"},
    ]
    for col, name in [
        ("lag_1w_units_sold", "naive_last_week"),
        ("lag_4w_units_sold", "naive_seasonal_4w"),
    ]:
        p = test_df[col].to_numpy(dtype=float)
        ok = ~np.isnan(p)
        rows.append(
            {
                "model": name,
                "mae": round(_mae(y[ok], p[ok]), 4),
                "smape": round(_smape(y[ok], p[ok]), 4),
                "source": f"computed (n={int(ok.sum())})",
            }
        )

    table = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    best_naive = table[table["model"].str.startswith("naive")]["mae"].min()
    table["v2_vs_this_pct_mae"] = ((table["mae"] - V2_MAE) / table["mae"] * 100).round(1)

    out = OUTPUTS_DIR / "forecast_naive_baseline.csv"
    table.to_csv(out, index=False)
    print(table.to_string(index=False))
    print(
        f"\nV2 beats the best naive ({best_naive:.3f} MAE) by "
        f"{(best_naive - V2_MAE) / best_naive * 100:.1f}% — real signal. "
        f"V2 beats the RF baseline by only {(RF_MAE - V2_MAE) / RF_MAE * 100:.1f}% — "
        f"RF is effectively the ceiling."
    )
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
