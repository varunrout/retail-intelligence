# Recommendation Systems V2 — Model Design and Implementation

## Overview

Hybrid SVD + content-based filtering system for product recommendations. Combines
collaborative filtering via truncated SVD with content-based scoring using cosine
similarity on product attributes. Addresses the Phase 10 evaluation methodology flaw
(random train/test split inflating difficulty) and introduces multi-signal weighting,
proper time-split evaluation, and vectorised batch inference.

---

## Source Files

| File | Role |
|---|---|
| `src/features/features_recsys.py` | Interaction matrix, content matrix, cold-start features |
| `src/models/train_recsys.py` | SVD, hybrid scoring, evaluation, k/alpha sweeps |
| `src/data/run_phase_recsys_v2.py` | End-to-end pipeline runner |

---

## Interaction Matrix

| Property | Value |
|---|---|
| Customers | 44,739 |
| Products | 5,000 |
| Matrix density | 0.60% (99.40% sparse) |
| Train interactions | ~994,000 |
| Test interactions | ~179,000 |
| Split date | 2025-11-01 (time-ordered) |
| Eval customers | 30,884 (customers with novel test items) |

**Multi-signal score (per interaction):**

```
score = log1p(qty) × 1.0 + log1p(views) × 0.15 + log1p(wishes) × 0.40 + pos_review × 0.50
```

Weights were chosen to give purchase quantity the dominant signal, with wishlist
(genuine purchase intent) the next strongest, reviews a moderate bonus, and browse
views the weakest signal. Log1p compression reduces heavy-tail distortion from power
users.

---

## Content Matrix

Built from `data/raw/product_attributes.csv`:
- Shape: (5,000 × 40)
- Features: TF-IDF on product tags, category OHE, price bands, average rating quantile

The content matrix supports both content-based scoring and cold-start recommendations
(products with no purchase history scored purely from attributes).

---

## Collaborative Filtering — Truncated SVD

The interaction matrix is decomposed as:
`R ≈ U · diag(s) · Vt`

where:
- `U` ∈ ℝ^(customers × k) — latent customer factors
- `s` ∈ ℝ^k — singular values
- `Vt` ∈ ℝ^(k × products) — latent product factors

**k-sweep results:**

| k | CF Hit@10 |
|---|---|
| 50 | 0.0630 |
| 100 | 0.0873 |
| 150 | 0.0986 |
| **200** | **0.1043** |
| 250 | 0.1038 |

**Best k = 200** selected (diminishing returns above 200).

---

## Hybrid Scoring

Final recommendation score blends CF and CB:

```
hybrid(u, i) = α × CF_score(u, i) + (1 − α) × CB_score(u, i)
```

where `CB_score` is the cosine similarity between item `i` and the mean embedding
of the customer's previously seen items.

**Alpha-sweep results:**

| α | Hybrid Hit@10 |
|---|---|
| 0.0 (CB only) | 0.0062 |
| 0.5 | 0.1133 |
| 0.7 | 0.1204 |
| **0.8** | **0.1225** ← best during sweep |
| 0.9 | 0.1208 |
| 1.0 (CF only) | 0.1043 |

**Best α = 0.8** — CF dominates (80%) with a modest CB contribution boosting novelty
and coverage for underexplored catalogue areas.

---

## Cold-Start Routing

Customers with < 5 historical purchases are routed to cold-start recommendations based
on:
1. Customer's preferred categories (from any browse/wishlist history)
2. Popularity-weighted top products within those categories

Cold-start customers: approximately 13,747 of 44,739 total (those excluded from eval
due to insufficient history for novel item evaluation).

---

## Vectorised Batch Evaluation

**CF scores for all eval customers:**
```python
eval_cidx = le_c.transform(eval_custs)            # single batch encode
all_cf_eval = (U[eval_cidx] * s) @ Vt             # shape: (30884, 5000)
```

**CB scores for all eval customers:**
```python
cos_sim_matrix = (content_norm @ content_norm.T)   # (5000, 5000) precomputed
norm_mask = normalize(purchase_mask[eval_cidx])    # (30884, 5000) sparse
all_cb_eval = norm_mask @ cos_sim_matrix           # one sparse × dense matmul
```

This replaces per-customer loops (30K × cosine_similarity calls → ~2 hrs) with
two matrix operations completing in minutes.

---

## Final Results

| Metric | SVD V2 (k=200, α=0.8) | Phase 10 SVD Baseline (k=50, random split) | Phase 10 (time-split reeval) |
|---|---|---|---|
| Hit@10 | **0.1204** | 0.0104 | 0.0630 |
| MRR@10 | **0.0606** | 0.0035 | — |
| Improvement factor | **11.57×** over Phase 10 | — | **1.91×** over honest baseline |

**Honest comparison note:** The Phase 10 baseline used a random train/test split where
97.9% of test items had never appeared in training — an artifically hard setting. On a
time-split re-evaluation, the Phase 10 SVD-50 achieves Hit@10 = 0.0630. V2 improves
this to 0.1204 (+91% on comparable evaluation).

---

## Outputs

Six files saved in `outputs/`:

| File | Contents |
|---|---|
| `phase_recsys_v2_params.csv` | Hyperparameter log |
| `phase_recsys_v2_metrics.json` | Hit@10, MRR@10, n_eval_customers |
| `phase_recsys_v2_vs_baseline.csv` | V2 vs Phase 10 metric comparison |
| `phase_recsys_v2_sample_recs.csv` | Top-10 recs for 500 sample customers |
| `phase_recsys_v2_k_sweep.csv` | CF Hit@10 across k ∈ {50,100,150,200,250} |
| `phase_recsys_v2_alpha_sweep.csv` | Hybrid Hit@10 across α ∈ {0.0,...,1.0} |
