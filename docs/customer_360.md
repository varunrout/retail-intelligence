# Customer 360 — one decision per customer

`analysis/customer_360.py` is the project's "tangible business output": it takes
the four customer-level signals produced across the six workstreams — churn
probability, uplift (persuadability), segment, and product recommendations —
and combines them into a single recommended action per customer, the shape a
CRM or a marketer would actually consume.

```
  churn probability  +  uplift (persuadability)  +  segment  +  recommendations
        │                      │                     │              │
        └──────────────────────┴─────────┬───────────┴──────────────┘
                                          ▼
                              one recommended action
```

## Run it

```bash
export PYTHONPATH=.
python -m analysis.customer_360                # full run, all customers
python -m analysis.customer_360 --customer C012345   # print one card
```

A full run retrains and rescores the uplift X-learner and the recommender SVD
on the full dataset (it does not read the committed top-500 sample artefacts),
so it takes a few minutes — dominated by the uplift model, which fits a
LightGBM booster per campaign.

## Outputs

- `outputs/customer_360_master.csv` — every customer, with each signal
  populated where available.
- `outputs/customer_360_heroes.csv` — customers who have all four signals.
- `outputs/customer_360.html` — printable one-pager of the top 12 hero cards.

## Recommended-action logic

| Condition | Action |
|---|---|
| High churn risk (≥50%) + persuadable (positive uplift) | Retention offer now + personalised recs |
| High churn risk, not persuadable | Service/loyalty touch, not a discount |
| Low risk, Champion/Loyal segment | Cross-sell via recs, no discount needed |
| Persuadable, not high risk | Include in next campaign |
| None of the above | Monitor, no strong action this cycle |

## Coverage — read this before trusting the numbers

- **Churn and segment are computed for every customer.** Churn is recomputed
  in this script (LightGBM, isotonic-calibrated, same feature set as
  `src/models/train_churn.py`) so every customer in `mart_customer_features`
  gets a probability. Segment comes from the committed
  `outputs/phase_segmentation_v2_cluster_assignments.csv`, which already
  covers the full population.
- **Uplift is recomputed for every campaign-targeted customer, not everyone.**
  The script retrains the X-learner from `src/models/train_uplift.py` on
  `mart_campaign_response` and scores every row, but a customer who was never
  targeted by any campaign has no uplift row to score — that's a real
  population gap (campaigns target ~35% of customers each, over ~8 campaigns,
  so most but not all customers end up with a score), not a sampling
  artefact left over from an earlier version of this script.
- **Recommendations are computed for every customer.** Warm customers (≥5
  purchases, present in the SVD's customer set) get the hybrid CF+content
  model from `src/models/train_recsys.py`; everyone else gets a
  category-popularity fallback, so recommendation coverage is complete.
- **"Hero" cards require all four signals**, so the hero count is bounded by
  uplift coverage even though churn, segment and recs are fully populated.

An earlier version of this script read the committed top-500 sample artefacts
for uplift and recommendations (`outputs/phase_uplift_v2_scored_sample_top500.csv`,
`outputs/phase_recsys_v2_recommendations_sample.csv`). Those samples barely
overlap, so almost no customer had all four signals. Recomputing both against
the full population, as described above, is what makes the hero-card view
meaningful.
