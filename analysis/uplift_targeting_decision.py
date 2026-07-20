"""Uplift — turn the decile table into an actual targeting decision.

The uplift workstream ends at a decile table and a Qini number but never states
who gets an offer. This script reads the committed V2 decile summary and computes,
for each cumulative top-k band, the observed incremental conversion rate and the
expected incremental conversions per 1,000 treated customers, then picks the band
to target (where incremental uplift stays above the population ATE).

Run:
    python -m analysis.uplift_targeting_decision
"""

from __future__ import annotations

import pandas as pd

from src.config import OUTPUTS_DIR

DECILE_CSV = OUTPUTS_DIR / "phase_uplift_v2_decile_summary.csv"


def main() -> None:
    dec = pd.read_csv(DECILE_CSV).sort_values("decile").reset_index(drop=True)
    ate = float(
        (dec["response_rate_treatment"] * dec["n_treatment"]).sum() / dec["n_treatment"].sum()
        - (dec["response_rate_control"] * dec["n_control"]).sum() / dec["n_control"].sum()
    )

    rows = []
    cum_incr = 0.0
    cum_treated = 0
    for _, r in dec.iterrows():
        marginal = float(r["observed_uplift"])  # this decile's own observed uplift
        incr = marginal * r["n_treatment"]
        cum_incr += incr
        cum_treated += int(r["n_treatment"])
        band_rate = cum_incr / cum_treated
        rows.append(
            {
                "decile": int(r["decile"]),
                "marginal_uplift": round(marginal, 4),
                "marginal_above_ate": bool(marginal > ate),
                "cum_treated": cum_treated,
                "cum_incremental_conversions": round(cum_incr, 1),
                "cum_incremental_rate": round(band_rate, 4),
                "cum_incr_per_1000_treated": round(band_rate * 1000, 1),
            }
        )
    table = pd.DataFrame(rows)

    # Decision: the deepest CONTIGUOUS run of top deciles whose OWN (marginal)
    # uplift clears the population ATE. Beyond that, an offer buys incremental
    # conversions at a rate below what treating everyone would give.
    target_k = 0
    for _, r in table.iterrows():
        if r["marginal_above_ate"]:
            target_k = int(r["decile"])
        else:
            break
    target_k = max(target_k, 1)

    out_path = OUTPUTS_DIR / "uplift_targeting_decision.csv"
    table.to_csv(out_path, index=False)

    print(f"Population ATE (treat everyone): {ate * 100:.2f}pp")
    print(table.to_string(index=False))
    band = table[table["decile"] == target_k].iloc[0]
    print(
        f"\nDECISION: target the top {target_k} deciles "
        f"({band['cum_treated']:,} treated in the test sample).\n"
        f"  Incremental conversion rate in band: {band['cum_incremental_rate'] * 100:.2f}pp "
        f"(vs {ate * 100:.2f}pp untargeted)\n"
        f"  Expected incremental conversions per 1,000 treated: "
        f"{band['cum_incr_per_1000_treated']:.0f} (vs {ate * 1000:.0f} untargeted)"
    )
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
