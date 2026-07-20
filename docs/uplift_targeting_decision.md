# Uplift — the targeting decision

_The uplift workstream produced a decile table and a Qini number but never said
who gets an offer. This closes that gap and fills the missing top-3-decile figure._

## What V2 actually buys over the baseline

`outputs/phase_uplift_v2_vs_baseline.csv` (top-3 was previously blank; now filled):

| metric | baseline (T-learner) | X-learner V2 | delta |
|---|---|---|---|
| overall ATE (test) | 0.0424 | 0.0444 | +0.0020 |
| top-3-decile uplift | 0.0398 | **0.0625** | **+0.0227** |
| top-5-decile uplift | 0.0537 | 0.0450 | -0.0086 |
| Qini-like area | 502 | **745** | **+242** |

The overall ATE is a tie — the campaigns lift conversion ~4.4pp either way. What
V2 adds is **concentration**: it ranks the most persuadable customers higher, so
the top-3 band and the Qini area both improve markedly. It is slightly worse at
the top-5 band, and the per-decile ordering is noisy in the middle (deciles 6-8
bounce back above the ATE), so the ranking is useful at the extreme top, not as a
clean monotone score.

## The decision

`analysis/uplift_targeting_decision.py` (writes `outputs/uplift_targeting_decision.csv`)
walks the deciles by predicted uplift and compares each decile's **own** observed
uplift to the population ATE (treat-everyone = 4.44pp):

| decile | marginal uplift | above ATE? |
|---|---|---|
| 1 | 7.54pp | yes |
| 2 | 6.36pp | yes |
| 3 | 4.83pp | yes |
| 4 | 1.47pp | no |
| 5 | 2.30pp | no |

**Target the top 3 deciles.** In that band the observed incremental conversion
rate is **6.24pp vs 4.44pp** for untreated targeting — about **62 incremental
conversions per 1,000 treated, against 44** if you treat everyone. At decile 4 the
marginal uplift collapses to 1.5pp, well below the ATE, so extending the offer
past decile 3 spends budget on customers who convert incrementally at less than the
untargeted rate. The deciles 6-8 rebound is not a reason to target them: with the
model's weak rank correlation it is most likely noise, and acting on it would mean
skipping deciles 4-5, which no coherent offer policy does.

## Reproduce

```bash
python -m src.data.run_phase_uplift_v2          # regenerates the decile + vs_baseline tables
python -m analysis.uplift_targeting_decision    # emits the decision table
```
