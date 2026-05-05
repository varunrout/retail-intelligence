# Phase 10: Recommendation System — Baseline Report

## Objective

Produce personalised top-10 product recommendations for each customer using
collaborative filtering and content-based methods. Establish a hit-rate baseline
against which V2 can be measured.

---

## Scope and Method

| Step | Detail |
|---|---|
| Interaction signal | Purchase quantities only (`order_items` + `orders`) |
| Matrix | Implicit buy-only customer × product |
| CF method | SVD-50 via `scipy.sparse.linalg.svds` on CSR matrix |
| CB method | Cosine similarity on TF-IDF product text features |
| Evaluation | hit_rate@10, MRR@10 |
| Eval methodology | **Random hold-out** (Phase 10 methodology — see caveats) |

---

## Baseline Methods Compared

### 1. Global Popularity

Recommend the top-10 globally best-selling products to every customer regardless of
their purchase history.

- hit_rate@10: 0.0036
- MRR@10: 0.0016

A floor benchmark. Captures the most purchased products, but makes no attempt at
personalisation.

### 2. Content-Based (Cosine Similarity)

Build a product feature matrix from text and categorical attributes; recommend products
most similar to each customer's past purchases using cosine similarity.

- hit_rate@10: 0.0062
- MRR@10: 0.0022

Modest improvement over popularity. Content similarity captures product category and
type proximity, but cannot learn latent preferences from purchase patterns.

### 3. Collaborative Filter — SVD-50 (buy-only)

Matrix factorisation on the implicit buy-only customer × product matrix. SVD decomposed
to k=50 latent factors using `scipy.sparse.linalg.svds`.

- hit_rate@10: **0.0104**
- MRR@10: **0.0035**

Best of the Phase 10 baselines. Latent factor model captures user-item co-occurrence
patterns that neither popularity nor content-based methods can.

---

## Phase 10 Summary

| Method | hit_rate@10 | MRR@10 |
|---|---|---|
| Popularity (global) | 0.0036 | 0.0016 |
| Content-Based (cosine) | 0.0062 | 0.0022 |
| SVD-50 (buy-only) | **0.0104** | **0.0035** |

---

## Critical Evaluation Caveat

The Phase 10 metrics above were produced with a **random hold-out** split rather than
a time-ordered split. This significantly understates model performance because:

1. The test set contains items the customer bought *before* the training cutoff,
   which are much easier to predict (the model has already seen them implicitly).
2. Novel items — products a customer has never purchased — are the true value of a
   recommender. Random splits do not specifically measure this.

When SVD-50 (buy-only) is re-evaluated with a proper **time-ordered split** at 2025-11-01,
measuring only on novel test purchases (items never seen in train), the same model
scores hit_rate@10=**0.0630** — 6× higher than the Phase 10 reported figure. The true
baseline for improvement comparison is therefore 0.0630.

---

## Interpretation

- SVD-50 at k=50 is severely underfit for 5,000 products. More latent factors are needed.
- Buy-only signals discard meaningful implicit intent: views and wishlists that precede
  purchases are excluded, leaving 63% of available interaction signal unused.
- The 99.40% matrix sparsity means higher-rank SVD and richer signals are both necessary
  to produce meaningful recommendations.
- Cold-start customers (few purchases) receive globally popular recommendations, which is
  a poor fallback with no category awareness.

---

## Caveats

- Hit rate is a conservative metric on a 5,000-product catalogue; even a good model will
  miss frequently because there are many valid products the customer would have liked but
  didn't happen to buy in the test window.
- 97.9% of test purchases are products the customer never bought in train (true novel
  items), so evaluation is genuinely hard and the 0.0104 figure should not be taken as
  evidence the problem is nearly unsolvable.
- Phase 10 did not implement cold-start routing; all customers received SVD scores
  regardless of their interaction history depth.

---

## Artifacts

| File | Description |
|---|---|
| `outputs/phase10_recommendation_eval.csv` | hit_rate@10 and MRR@10 for all three baselines |
| `outputs/phase10_recommendation_eval.png` | Bar chart comparison of baseline methods |
| `outputs/phase10_global_popularity_ranking.csv` | Global top-N product popularity ranking |

---

## Next Phase

**Phase 10 → V2 (Multi-Signal Hybrid SVD):** Enrich interaction signals with views,
wishlists and positive reviews; increase SVD rank to k=200; blend collaborative filtering
with content-based scores (hybrid α=0.8); add cold-start routing; evaluate on novel test
items only using a time-ordered split.
