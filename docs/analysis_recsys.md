# Recommendation Systems Analysis — Evidence Audit Before V2

## Purpose

Evidence-first analysis of the interaction data before designing Recommendation Systems
V2. Covers data sparsity, multi-signal enrichment, baseline diagnosis, feature sweep,
rank selection, content features, and hybrid blending.

---

## §1 Data Profile

| Metric | Value |
|---|---|
| Customers | 44,739 |
| Products | 5,000 |
| Matrix sparsity | **99.40%** |
| Median purchases per customer | 23 |
| Customers buying across 3+ categories | 73% |

**Finding 1.A:** 99.40% sparse interaction matrix — a classic implicit collaborative
filtering setting. Virtually every customer-product pair is unobserved.

**Finding 1.B:** Four interaction signal types are available: purchases (`order_items`),
`view_product` events, `wishlist_add` events, and positive reviews (rating ≥ 4). All
5,000 products have at least one purchase.

**Finding 1.C:** 73% of customers buy across 3+ categories — no strong single-category
loyalty. Cross-category recommendations are behaviorally appropriate.

---

## §2 Interaction Signals

**Finding 2.A:** Multi-signal graph = 2.19M interaction pairs vs 1.34M purchase-only
(+63%). Views (813K events) and wishlists (45K) substantially enrich the interaction
graph.

**Finding 2.B:** 97.9% of test purchases are products **never** purchased by the customer
in train — virtually the entire evaluation is on cold/novel items. Pure repeat-purchase
logic (recommend what you bought before) fails completely; discovery-oriented CF is
essential.

**Finding 2.C:** Reviews are bimodal (peaks at 3 and 4 stars). 63% are positive (≥ 4).
Negative reviews indicate dislike and must be **excluded** from implicit signal — they
would pull the model towards products the customer did not enjoy.

**Implicit score formula:**
```
score = log1p(qty) × 1.0 + log1p(views) × 0.15 + log1p(wishes) × 0.40 + pos_reviews × 0.5
```

Signal weights rationale: buy = strongest intent; wish = moderate intent (saved, not bought);
view = weak intent (explored, not converted); positive review = post-purchase confirmation.

---

## §3 Baseline Diagnosis

**Finding 3.A — Evaluation methodology failure:**
Phase 10 SVD-50 hit@10 = **0.0104** (Phase 10 eval) vs **0.0630** (time-split).
The same model scores 6× higher under a proper time-ordered hold-out. Phase 10's random
split allowed test items to appear in training history, making the evaluation trivially
easy — but the reported 0.0104 reflects a broken evaluation harness, not the model's
true difficulty.

**Finding 3.B:** Phase 10 used purchase signal only. 813K view events, 45K wishlist
events, and review signals were not used.

**Finding 3.C:** Phase 10 had no cold-start strategy. All customers — including those
with zero or one purchase — received SVD scores, producing meaningless recommendations
for the 11% of customers with insufficient purchase history.

---

## §4 Signal Scheme Sweep

Results at SVD rank k=50 across signal combinations:

| Scheme | hit@10 |
|---|---|
| buy only | 0.0630 |
| buy + view | 0.0731 |
| buy + wish | 0.0748 |
| **buy + view + wish + review** | **0.0783** |

**Finding 4.A:** Best scheme = buy+view+wish+pos_review at hit@10 = 0.0783. Each
additional signal type provides incremental lift.

**Finding 4.B:** View signal (813K events) adds the largest volume of new pairs (+62%
of multi-signal pairs are view-only). Weight `w_view = 0.15` (1/7 of purchase weight)
to prevent views from dominating the implicit score.

---

## §5 SVD Rank Sweep (k Selection)

Multi-signal matrix, time-ordered evaluation:

| k | hit@10 |
|---|---|
| 20 | 0.0883 |
| 50 | 0.0783 |
| 100 | 0.0887 |
| 150 | 0.0950 |
| **200** | **0.1043** |

**Finding 5.A:** k = 200 is optimal (hit@10 = 0.1043). Diminishing returns beyond k = 100
but meaningful gain from 150 → 200.

**Finding 5.B:** `scipy.sparse.linalg.svds` on the CSR matrix scales to the full
(30,992 × 5,000) matrix without densification. Memory and runtime are acceptable.

---

## §6 Content Features

**Finding 6.A:** Content similarity is strongly category-discriminative:
- Intra-category cosine similarity: 0.548
- Cross-category cosine similarity: 0.167

OHE over `category`, `subcategory`, `brand_type`, `price_tier` plus embedding group
prefix and binary attribute flags → (5,000 × 40) product feature matrix.

**Finding 6.B:** 630 `recommendation_embedding_groups` provide pre-computed product
clusters. These can serve as efficient cold-start recommendation groups.

**Finding 6.C:** Content-based alone achieves hit@10 ≈ 0.035 — weaker than CF but
provides orthogonal signal that helps for cold-start users and new products.

---

## §7 Hybrid Blending and Cold-Start

**Finding 7.A — Hybrid improvement:**
Blending CF and CB scores at α = 0.8 (80% CF + 20% CB) achieves hit@10 = **0.1225**
vs CF-only 0.1043 — a further +17% relative gain from content smoothing.

**Finding 7.B — Cold-start customers:**
5,017 customers (11%) have < 5 purchases. SVD embeddings are unreliable for these
customers due to insufficient interaction history. Category-aware top-N popularity
fallback provides a personalised baseline without requiring CF.

**Finding 7.C — V2 routing strategy:**
1. Customers with ≥ 5 purchases → Multi-signal SVD (k=200) + CB hybrid (α = 0.8)
2. Customers with < 5 purchases → Category-aware top-10 popularity fallback
3. Products never seen in training → CB similarity only

---

## §8 V2 Design Decisions

| Dimension | Phase 10 Baseline | V2 Decision |
|---|---|---|
| Interaction signal | Buy only | Buy + view (×0.15) + wish (×0.40) + pos_review (×0.50) |
| SVD rank | k=50 | k=200 |
| Hybrid blending | None | α=0.8 CF + 0.2 CB |
| Cold-start | Global popularity | Category-aware top-N popularity |
| Evaluation split | Random hold-out | Time-ordered at 2025-11-01 |
| Eval scope | All test items | Novel items only (never purchased in train) |
| Eval customers | All | 30,884 customers with novel test purchases |
