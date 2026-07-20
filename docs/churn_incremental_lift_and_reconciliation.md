# Churn — incremental lift and baseline reconciliation

_Read this before trusting any churn headline number._

## The problem with the committed comparison

`outputs/phase_churn_v2_vs_baseline.csv` reports the LightGBM V2 model as
-0.050 ROC-AUC and -0.137 PR-AUC against the phase6 Random Forest, with the note
"regimes differ". That note is the whole problem: the two models were never
measured on the same footing.

| | phase6 baseline (RF) | phase_churn_v2 (LightGBM) |
|---|---|---|
| Split | stratified random 80/20 | time-ordered 80/20 |
| Population | all customers | active + 90-day mature cohort only |
| Features | include recency-derived | recency excluded |
| Leakage | `customer_segment_seed` included | `customer_segment_seed` included |

Three things change at once — algorithm, split, and feature policy — so the delta
cannot tell you whether LightGBM is better or worse than a simple model. It only
tells you the two setups score differently, which is unsurprising.

## The one matched comparison

`analysis/churn_incremental_lift.py` fixes this. Every model is trained and scored
on the **identical** time-ordered split, the **same** active + 90-day mature
population, and the **same** feature policy (recency excluded, and
`customer_segment_seed` removed — see the leakage note below). The baselines get a
fair standard pipeline (median-impute numerics, standard-scale, one-hot the same
categoricals); LightGBM keeps its native categorical/null handling. The only thing
that varies is the algorithm. A paired bootstrap over the shared held-out rows
(n=8,388, 2,000 resamples) gives a 95% CI on the lift.

Result (`outputs/churn_incremental_lift_metrics.csv`):

| model | ROC-AUC | PR-AUC |
|---|---|---|
| **logreg_balanced** | **0.844** | **0.497** |
| rf_balanced | 0.814 | 0.482 |
| lightgbm_v2 | 0.812 | 0.448 |

Paired lift, LightGBM V2 vs the best baseline (logistic regression)
(`outputs/churn_incremental_lift_paired_bootstrap.csv`):

| metric | lift | 95% CI | verdict |
|---|---|---|---|
| ROC-AUC | -0.033 | [-0.038, -0.027] | does not beat baseline |
| PR-AUC | -0.050 | [-0.065, -0.034] | does not beat baseline |

## Verdict

**LightGBM V2 does not beat a plain logistic regression on this problem.** On a
like-for-like footing the gradient-boosted model is the *worst* of the three, and
the paired bootstrap CI on the lift excludes zero in the negative direction for
both ROC-AUC and PR-AUC — the gap is real, not noise. A regularised logistic
regression is the model that should ship for churn.

This reverses the impression the V2 layer gives. The earlier -0.050 headline was
read as "the time-ordered regime is just harder"; the matched test shows the
harder regime is real *and* the boosted model adds nothing over a linear baseline
inside it. Reporting this cleanly is the point: a rigorous negative result is more
useful than a comparison rigged across three moving parts.

## Leakage note: customer_segment_seed

`customer_segment_seed` is a synthetic data-generating seed, not an attribute that
exists for a real customer at scoring time. It was in both the phase6 and V2
feature sets. It has been removed from the churn feature set
(`src/features/features_churn.py`, `_EXCLUDED_LEAKAGE_FEATURES`). Removing it drops
LightGBM ROC-AUC from ~0.815 to 0.812 on the matched split — small, but the point
is correctness: a feature that cannot exist in production does not belong in the
comparison.

## Calibration — the thresholds gate on miscalibrated probabilities

`phase_churn_v2_threshold_selection.csv` picks retention-offer cuts at 0.50 and
0.70 probability, but the raw LightGBM scores are not calibrated probabilities.
On the time-ordered test set (`outputs/churn_calibration_brier.csv`):

| model | Brier | mean predicted | observed rate |
|---|---|---|---|
| lightgbm_v2_raw | 0.197 | **0.441** | 0.150 |
| lightgbm_v2_isotonic | **0.141** | 0.323 | 0.150 |
| logreg_balanced | 0.237 | 0.479 | 0.150 |

The raw model predicts a mean churn probability of **0.44 against an actual rate of
0.15** — it overstates risk roughly threefold, so a "0.70" cut does not mean a 70%
churn risk and the threshold memo is built on sand. Isotonic calibration on a
held-out slice fixes most of it (Brier 0.197 → 0.141). Note the tension worth
stating plainly: logistic regression ranks best (Section above) but is the worst
*calibrated* because `class_weight="balanced"` inflates its probabilities. The
practical recommendation: rank with a model chosen on AUC/PR-AUC, but **calibrate
before applying any probability threshold**, and report the reliability diagram
(`outputs/churn_calibration_reliability.png`) alongside the threshold table.

## Reproduce

```bash
python -m analysis.churn_incremental_lift          # matched baseline comparison
python -m analysis.churn_incremental_lift --smoke  # synthetic, no data needed
python -m analysis.churn_calibration               # reliability + calibrated Brier
python -m analysis.churn_calibration --smoke
```
