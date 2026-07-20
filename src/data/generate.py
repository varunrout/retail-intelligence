"""Deterministic synthetic raw-data generator for retail-intelligence.

Emits the 16 raw tables under ``data/raw/`` (plus ``returns_hidden_labels.csv``
under ``data/processed/``) that the existing ``sql/*.sql`` marts consume. The
data is internally consistent (keys join across tables) and carries planted
signal so the models find something real rather than noise:

- churn: engagement drives order recency, so low-engagement customers are the
  ones whose last order falls past the 90-day churn cutoff.
- uplift: each customer has a latent persuadability; the treatment lifts
  conversion most for persuadable customers, giving a positive, heterogeneous ATE.
- forecast: weekly demand has product base rates, seasonality and trend, so lag
  features are predictive.
- returns anomaly: a small fraction of returns are abuse, driven by a latent
  per-order fraud seed and serial-returner behaviour (~0.7% prevalence), written
  to returns_hidden_labels.csv.

Everything is seeded, so a given ``--seed`` and ``--scale`` reproduce byte-for-byte.

Run:
    python -m src.data.generate                      # default scale
    python -m src.data.generate --scale sample       # tiny, for CI/tests
    python -m src.data.generate --n-customers 20000  # custom
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR

# ── Scale presets ───────────────────────────────────────────────────────────
SCALES: dict[str, dict] = {
    "sample": {"n_customers": 400, "n_products": 60, "n_stores": 4, "months": 12},
    "default": {"n_customers": 4000, "n_products": 300, "n_stores": 12, "months": 16},
    "full": {"n_customers": 50000, "n_products": 5000, "n_stores": 60, "months": 20},
}

REGIONS = ["North", "South", "East", "West", "Central"]
CATEGORIES = ["fashion", "home", "beauty", "grocery_light", "electronics", "sports"]
INCOME_BANDS = ["low", "lower_mid", "mid", "upper_mid", "high"]
SEGMENTS = ["new", "steady", "loyal", "at_risk", "lapsing"]


@dataclass
class Config:
    n_customers: int
    n_products: int
    n_stores: int
    months: int
    seed: int


def _dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series)


def _date_str(arr) -> np.ndarray:
    return pd.DatetimeIndex(pd.to_datetime(arr)).strftime("%Y-%m-%d").to_numpy()


def _ts_str(arr) -> np.ndarray:
    return pd.DatetimeIndex(pd.to_datetime(arr)).strftime("%Y-%m-%d %H:%M:%S").to_numpy()


# ── Calendar ────────────────────────────────────────────────────────────────
def gen_calendar(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="D")
    df = pd.DataFrame({"date": dates})
    df["day_of_week"] = dates.day_name()
    iso = dates.isocalendar()
    df["week_of_year"] = iso.week.to_numpy()
    df["month"] = dates.month
    df["quarter"] = dates.quarter
    df["year"] = dates.year
    df["is_weekend"] = dates.dayofweek >= 5
    df["is_month_end"] = dates.is_month_end
    df["is_payday_period"] = dates.day.isin([1, 2, 3, 15, 16, 28, 29, 30, 31])
    holidays = {
        "01-01": "New Year Day",
        "12-25": "Christmas",
        "12-26": "Boxing Day",
        "11-29": "Black Friday",
        "07-04": "Summer Sale",
    }
    md = dates.strftime("%m-%d")
    df["holiday_name"] = [holidays.get(x) for x in md]
    df["is_public_holiday"] = df["holiday_name"].notna()
    season = np.select(
        [dates.month.isin([12, 1, 2]), dates.month.isin([3, 4, 5]), dates.month.isin([6, 7, 8])],
        ["winter_reset", "spring_launch", "summer_sale"],
        default="autumn_peak",
    )
    df["retail_season"] = season
    df["promo_season_flag"] = dates.month.isin([1, 6, 7, 11, 12])
    df["date"] = _date_str(df["date"])
    return df


# ── Stores ──────────────────────────────────────────────────────────────────
def gen_stores(cfg: Config, rng: np.random.Generator) -> pd.DataFrame:
    ids = [f"S{i:03d}" for i in range(1, cfg.n_stores + 1)]
    cities = [f"City{i}" for i in range(cfg.n_stores)]
    return pd.DataFrame(
        {
            "store_id": ids,
            "region": rng.choice(REGIONS, cfg.n_stores),
            "city": cities,
            "store_type": rng.choice(["mall", "neighborhood", "flagship", "outlet"], cfg.n_stores),
            "opening_date": _date_str(
                pd.Timestamp("2020-01-01")
                + pd.to_timedelta(rng.integers(0, 1400, cfg.n_stores), "D")
            ),
            "store_size_band": rng.choice(["small", "medium", "large"], cfg.n_stores),
            "store_format": rng.choice(["convenience", "high_traffic", "standard"], cfg.n_stores),
            "latitude_fake": np.round(rng.uniform(30, 55, cfg.n_stores), 4),
            "longitude_fake": np.round(rng.uniform(-120, -70, cfg.n_stores), 4),
        }
    )


# ── Products (+ attributes) ─────────────────────────────────────────────────
def gen_products(cfg: Config, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = cfg.n_products
    ids = [f"P{i:05d}" for i in range(1, n + 1)]
    category = rng.choice(CATEGORIES, n)
    base_price = np.round(rng.gamma(2.0, 20.0, n) + 5, 2)
    cost = np.round(base_price * rng.uniform(0.45, 0.75, n), 2)
    price_tier = np.select([base_price < 20, base_price < 60], ["entry", "mid"], default="premium")
    products = pd.DataFrame(
        {
            "product_id": ids,
            "category": category,
            "subcategory": [f"{c[:3]}_{rng.integers(0, 5)}" for c in category],
            "brand_type": rng.choice(["owned_brand", "emerging_label", "national"], n),
            "price_tier": price_tier,
            "base_price": base_price,
            "cost": cost,
            "launch_date": _date_str(
                pd.Timestamp("2021-01-01") + pd.to_timedelta(rng.integers(0, 1600, n), "D")
            ),
            "product_style": rng.choice(["minimal", "modern", "classic", "bold"], n),
            "seasonal_flag": rng.random(n) < 0.3,
            "premium_flag": price_tier == "premium",
            # latent return propensity per product (also drives real returns)
            "return_risk_seed": np.round(rng.beta(1.5, 12, n), 4),
            "margin_band": rng.choice(["low", "mid", "high"], n),
        }
    )
    attrs = pd.DataFrame(
        {
            "product_id": ids,
            "color": rng.choice(["black", "white", "sand", "silver", "navy", "red"], n),
            "size_group": rng.choice(["single", "standard", "family"], n),
            "material_type": rng.choice(["cotton", "ceramic", "plastic", "metal", "blend"], n),
            "eco_flag": rng.random(n) < 0.2,
            "bundle_candidate_flag": rng.random(n) < 0.35,
            "recommendation_embedding_group": [
                f"{c[:3].upper()}_{rng.integers(0, 4)}" for c in category
            ],
            "description_keywords": [
                f"{c}, {s}" for c, s in zip(category, products["product_style"], strict=False)
            ],
        }
    )
    return products, attrs


# ── Customers (with latent traits used across workstreams) ──────────────────
def gen_customers(cfg: Config, rng: np.random.Generator, anchor: pd.Timestamp, start: pd.Timestamp):
    n = cfg.n_customers
    ids = [f"C{i:06d}" for i in range(1, n + 1)]
    signup = start + pd.to_timedelta(rng.integers(0, (anchor - start).days - 5, n), "D")
    engagement = rng.beta(2.0, 2.0, n)  # latent: drives orders, sessions, recency
    persuadability = rng.beta(2.0, 3.0, n)  # latent: drives uplift
    base_return = rng.beta(1.5, 10, n)  # latent: drives returns
    loyalty_enrolled = rng.random(n) < 0.45
    latent = pd.DataFrame(
        {
            "customer_id": ids,
            "engagement": engagement,
            "persuadability": persuadability,
            "base_return": base_return,
        }
    )
    customers = pd.DataFrame(
        {
            "customer_id": ids,
            "signup_date": _date_str(signup),
            "birth_year": rng.integers(1955, 2005, n).astype(float),
            "gender": rng.choice(["male", "female", "other"], n, p=[0.48, 0.48, 0.04]),
            "region": rng.choice(REGIONS, n),
            "city_tier": rng.choice(["Tier_1", "Tier_2", "Tier_3"], n),
            "acquisition_channel": rng.choice(
                ["paid_social", "organic", "referral", "app", "email"], n
            ),
            "preferred_channel": rng.choice(["online", "store", "app"], n),
            "loyalty_tier": np.where(
                loyalty_enrolled, rng.choice(["silver", "gold", "platinum"], n), None
            ),
            "loyalty_signup_date": np.where(
                loyalty_enrolled,
                _date_str(signup + pd.to_timedelta(rng.integers(0, 200, n), "D")),
                None,
            ),
            "income_band": rng.choice(INCOME_BANDS, n),
            # synthetic seed column (deliberately correlated with engagement)
            "customer_segment_seed": np.select(
                [engagement < 0.25, engagement < 0.5, engagement < 0.75],
                ["lapsing", "at_risk", "steady"],
                default="loyal",
            ),
            "is_marketing_opt_in": rng.random(n) < 0.7,
        }
    )
    return customers, latent


# ── Orders + order_items ────────────────────────────────────────────────────
def gen_orders(cfg, rng, customers, latent, products, stores, anchor, start):
    n = cfg.n_customers
    eng = latent["engagement"].to_numpy()
    signup = _dt(customers["signup_date"]).to_numpy()

    # order count grows with engagement
    n_orders = rng.poisson(1 + 16 * eng) + 1

    # churn: low-engagement customers stop earlier. gap = days between last order
    # and the anchor. churn_flag (recency > 90) therefore anti-correlates with eng.
    churn_prob = 1 / (1 + np.exp(8 * (eng - 0.30)))
    is_churned = rng.random(n) < churn_prob
    gap = np.where(
        is_churned,
        rng.integers(95, 260, n),
        rng.integers(0, 80, n),
    )
    last_order = anchor - pd.to_timedelta(gap, "D")
    last_order = pd.to_datetime(
        np.maximum(last_order.values, (pd.to_datetime(signup) + pd.Timedelta(days=3)).values)
    )

    # Build order rows customer by customer (vectorised per customer via repeat).
    cust_idx = np.repeat(np.arange(n), n_orders)
    total_orders = len(cust_idx)
    # order dates: uniform between signup and the customer's last-order date
    su = pd.to_datetime(signup)[cust_idx].astype("int64")
    lo = pd.to_datetime(last_order)[cust_idx].astype("int64")
    frac = rng.random(total_orders)
    od_int = su + (frac * (lo - su)).astype("int64")
    order_ts = pd.to_datetime(od_int)
    # ensure each customer's last order lands exactly on last_order (recency truth)
    first_of_cust = np.concatenate([[True], cust_idx[1:] != cust_idx[:-1]])
    # place the max at the true last_order by overwriting the first row per customer
    order_ts = order_ts.to_numpy()
    order_ts[first_of_cust] = pd.to_datetime(last_order).to_numpy()

    hours = rng.integers(8, 22, total_orders)
    mins = rng.integers(0, 60, total_orders)
    order_dt = pd.to_datetime(order_ts) + pd.to_timedelta(hours, "h") + pd.to_timedelta(mins, "m")

    order_ids = [f"O{i:08d}" for i in range(1, total_orders + 1)]
    channel = rng.choice(["online", "store", "app"], total_orders, p=[0.5, 0.3, 0.2])
    store_id = np.where(
        channel == "store", rng.choice(stores["store_id"].to_numpy(), total_orders), None
    )
    status = rng.choice(
        ["completed", "canceled", "returned", "pending"], total_orders, p=[0.86, 0.06, 0.05, 0.03]
    )
    fraud_seed = np.round(rng.beta(1.2, 20, total_orders), 4)

    orders = pd.DataFrame(
        {
            "order_id": order_ids,
            "customer_id": customers["customer_id"].to_numpy()[cust_idx],
            "order_datetime": _ts_str(order_dt),
            "order_date": _date_str(order_dt),
            "channel": channel,
            "store_id_nullable": store_id,
            "payment_type": rng.choice(
                ["debit_card", "credit_card", "wallet", "bnpl"], total_orders
            ),
            "order_status": status,
            "device_type": rng.choice(["desktop", "mobile", "pos", "tablet"], total_orders),
            "used_promo_code": rng.random(total_orders) < 0.25,
            "campaign_attributed_flag": rng.random(total_orders) < 0.15,
            "fraud_risk_seed": fraud_seed,
        }
    )
    orders["promo_code_nullable"] = np.where(
        orders["used_promo_code"],
        [f"SAVE{v}" for v in rng.integers(5, 60, total_orders)],
        None,
    )
    orders["session_id_nullable"] = None  # linked later for a subset

    # ── order_items ──────────────────────────────────────────────────────────
    basket = rng.integers(1, 6, total_orders)
    orders["basket_size"] = basket
    item_order_idx = np.repeat(np.arange(total_orders), basket)
    n_items = len(item_order_idx)
    prod_idx = rng.integers(0, cfg.n_products, n_items)
    qty = rng.integers(1, 4, n_items)
    list_price = products["base_price"].to_numpy()[prod_idx]
    disc = np.round(rng.beta(1.5, 6, n_items), 4)
    unit_net = list_price * (1 - disc)
    line_net = np.round(unit_net * qty, 2)
    cost = products["cost"].to_numpy()[prod_idx]
    completed = (orders["order_status"].to_numpy() == "completed")[item_order_idx]
    # canceled/pending lines carry zero realised revenue
    line_net = np.where(completed, line_net, 0.0)
    safe_net = np.where(line_net > 0, line_net, 1.0)
    margin = np.round(np.where(line_net > 0, (line_net - cost * qty) / safe_net, 0.0), 4)

    order_items = pd.DataFrame(
        {
            "order_item_id": [f"OI{i:09d}" for i in range(1, n_items + 1)],
            "order_id": np.asarray(order_ids)[item_order_idx],
            "product_id": products["product_id"].to_numpy()[prod_idx],
            "quantity": qty,
            "item_list_price": np.round(list_price, 2),
            "item_discount_pct": disc,
            "item_net_price": line_net,
            "item_cost": np.round(cost, 2),
            "item_margin": margin,
            "fulfillment_type": rng.choice(
                ["ship_from_dc", "ship_from_store", "in_store_takeaway"], n_items
            ),
        }
    )

    # order money fields from line totals
    gross = np.round(list_price * qty, 2)
    line_df = pd.DataFrame({"order_id": order_items["order_id"], "gross": gross, "net": line_net})
    agg = line_df.groupby("order_id", sort=False).sum()
    orders = orders.merge(
        agg.rename(columns={"gross": "gross_amount", "net": "net_amount"}),
        on="order_id",
        how="left",
    )
    orders["gross_amount"] = orders["gross_amount"].fillna(0.0).round(2)
    orders["net_amount"] = orders["net_amount"].fillna(0.0).round(2)
    orders["discount_amount"] = (orders["gross_amount"] - orders["net_amount"]).round(2)
    orders["shipping_fee"] = np.where(orders["channel"].to_numpy() == "online", 0.0, 0.0)

    # column order to match the original schema
    orders = orders[
        [
            "order_id",
            "customer_id",
            "order_datetime",
            "order_date",
            "channel",
            "store_id_nullable",
            "payment_type",
            "order_status",
            "gross_amount",
            "discount_amount",
            "net_amount",
            "shipping_fee",
            "device_type",
            "basket_size",
            "used_promo_code",
            "promo_code_nullable",
            "campaign_attributed_flag",
            "session_id_nullable",
            "fraud_risk_seed",
        ]
    ]
    return orders, order_items, cust_idx, is_churned


# ── Returns (+ hidden abuse label) ──────────────────────────────────────────
def gen_returns(cfg, rng, orders, order_items, products, customers, latent):
    # only completed-order items can be returned
    completed_ids = set(orders.loc[orders["order_status"] == "completed", "order_id"])
    items = order_items[order_items["order_id"].isin(completed_ids)].copy()
    items = items.merge(
        orders[["order_id", "customer_id", "order_date"]], on="order_id", how="left"
    )
    prod_ret = products.set_index("product_id")["return_risk_seed"]
    cust_ret = latent.set_index("customer_id")["base_return"]
    p_return = (
        0.6 * items["product_id"].map(prod_ret).to_numpy()
        + 0.6 * items["customer_id"].map(cust_ret).to_numpy()
        + 0.15 * items["item_discount_pct"].to_numpy()
    )
    returned = rng.random(len(items)) < np.clip(p_return, 0, 0.9)
    ret_items = items[returned].reset_index(drop=True)
    m = len(ret_items)
    req_gap = rng.integers(2, 40, m)
    req_date = _dt(ret_items["order_date"]) + pd.to_timedelta(req_gap, "D")
    proc_date = req_date + pd.to_timedelta(rng.integers(1, 10, m), "D")
    refund = np.round(ret_items["item_net_price"].to_numpy() * rng.uniform(0.8, 1.0, m), 2)

    returns = pd.DataFrame(
        {
            "return_id": [f"R{i:08d}" for i in range(1, m + 1)],
            "order_item_id": ret_items["order_item_id"].to_numpy(),
            "return_request_date": _date_str(req_date),
            "return_processed_date": _date_str(proc_date),
            "return_reason": rng.choice(
                ["not_as_expected", "wrong_size", "damaged", "changed_mind", "late_delivery"], m
            ),
            "refund_amount": refund,
            "refund_method": rng.choice(["original_payment", "store_credit"], m, p=[0.8, 0.2]),
            "return_status": rng.choice(["processed", "approved", "pending"], m, p=[0.7, 0.2, 0.1]),
            "return_channel": rng.choice(["online", "store", "courier"], m),
        }
    )

    # ── hidden abuse label ────────────────────────────────────────────────────
    # abuse: high per-order fraud seed AND (fast return OR high refund). Rare.
    order_fraud = orders.set_index("order_id")["fraud_risk_seed"]
    ret_fraud = ret_items["order_id"].map(order_fraud).to_numpy()
    fast = req_gap <= 5
    high_refund = refund > np.quantile(refund, 0.85) if m else np.zeros(m, bool)
    abuse_score = ret_fraud + 0.5 * fast + 0.3 * high_refund
    abuse = abuse_score > np.quantile(abuse_score, 0.993) if m else np.zeros(m, bool)
    labels = pd.DataFrame(
        {"return_id": returns["return_id"], "abuse_flag_hidden_for_validation": abuse}
    )
    return returns, labels


# ── Web sessions + session events ───────────────────────────────────────────
def gen_sessions(cfg, rng, customers, latent, orders, products, anchor, start):
    eng = latent["engagement"].to_numpy()
    n_sessions = rng.poisson(2 + 20 * eng)
    cust_idx = np.repeat(np.arange(cfg.n_customers), n_sessions)
    total = len(cust_idx)
    signup = _dt(customers["signup_date"]).to_numpy()
    su = pd.to_datetime(signup)[cust_idx].astype("int64")
    an = np.int64(anchor.value)
    frac = rng.random(total)
    start_int = su + (frac * (an - su)).astype("int64")
    s_start = pd.to_datetime(start_int)
    dur = rng.integers(2, 60, total)
    s_end = s_start + pd.to_timedelta(dur, "m")
    purchase = rng.random(total) < (0.05 + 0.3 * eng[cust_idx])
    add_cart = purchase | (rng.random(total) < 0.3)
    sess_ids = [f"WS{i:09d}" for i in range(1, total + 1)]
    sessions = pd.DataFrame(
        {
            "session_id": sess_ids,
            "customer_id_nullable": customers["customer_id"].to_numpy()[cust_idx],
            "session_start": _ts_str(s_start),
            "session_end": _ts_str(s_end),
            "traffic_source": rng.choice(["search", "direct", "paid_social", "email"], total),
            "device_type": rng.choice(["desktop", "mobile", "tablet"], total),
            "pages_viewed": rng.integers(1, 20, total),
            "product_views": rng.integers(0, 15, total),
            "category_views": rng.integers(0, 8, total),
            "add_to_cart_flag": add_cart,
            "checkout_flag": purchase,
            "purchase_flag": purchase,
            "bounce_flag": rng.random(total) < 0.2,
        }
    )
    # a light session_events table (one row per session, product view)
    prod_idx = rng.integers(0, cfg.n_products, total)
    events = pd.DataFrame(
        {
            "event_id": [f"E{i:010d}" for i in range(1, total + 1)],
            "session_id": sess_ids,
            "customer_id_nullable": sessions["customer_id_nullable"].to_numpy(),
            "event_time": sessions["session_start"].to_numpy(),
            "event_type": rng.choice(["view_product", "search", "add_to_cart"], total),
            "product_id_nullable": products["product_id"].to_numpy()[prod_idx],
            "category_nullable": products["category"].to_numpy()[prod_idx],
        }
    )
    return sessions, events


# ── Reviews ─────────────────────────────────────────────────────────────────
def gen_reviews(cfg, rng, customers, products, orders, order_items, anchor, start):
    # reviewers = a sample of completed order_items
    completed_ids = set(orders.loc[orders["order_status"] == "completed", "order_id"])
    items = order_items[order_items["order_id"].isin(completed_ids)]
    items = items.merge(
        orders[["order_id", "customer_id", "order_date"]], on="order_id", how="left"
    )
    take = rng.random(len(items)) < 0.15
    r = items[take].reset_index(drop=True)
    m = len(r)
    rating = rng.choice([1, 2, 3, 4, 5], m, p=[0.05, 0.08, 0.15, 0.32, 0.4])
    return pd.DataFrame(
        {
            "review_id": [f"RV{i:08d}" for i in range(1, m + 1)],
            "customer_id": r["customer_id"].to_numpy(),
            "product_id": r["product_id"].to_numpy(),
            "review_date": _date_str(
                _dt(r["order_date"]) + pd.to_timedelta(rng.integers(3, 40, m), "D")
            ),
            "rating": rating,
            "review_length": rng.integers(10, 300, m),
            "sentiment_seed": np.round(np.clip(rating / 5 + rng.normal(0, 0.1, m), 0, 1), 4),
            "helpful_votes": rng.poisson(1.5, m),
        }
    )


# ── Campaigns / targets / events (uplift signal) ────────────────────────────
def gen_campaigns(cfg, rng, customers, latent, anchor, start):
    n_campaigns = max(6, cfg.months // 2)
    cids = [f"CMP{i:03d}" for i in range(1, n_campaigns + 1)]
    starts = start + pd.to_timedelta(
        np.sort(rng.integers(10, (anchor - start).days - 40, n_campaigns)), "D"
    )
    campaigns = pd.DataFrame(
        {
            "campaign_id": cids,
            "campaign_name": [f"Campaign {i}" for i in range(1, n_campaigns + 1)],
            "campaign_type": rng.choice(["retention", "seasonal_promo", "winback"], n_campaigns),
            "start_date": _date_str(starts),
            "end_date": _date_str(starts + pd.to_timedelta(14, "D")),
            "channel": rng.choice(["email", "push", "sms"], n_campaigns),
            "offer_type": rng.choice(["discount_pct", "bundle_offer", "free_ship"], n_campaigns),
            "offer_strength": np.round(rng.uniform(0.08, 0.25, n_campaigns), 2),
            "target_rule_summary": rng.choice(
                ["inactive_90d", "high_value", "fashion_affinity"], n_campaigns
            ),
            "control_group_pct": rng.choice([0.1, 0.15, 0.2], n_campaigns),
        }
    )

    persuade = latent["persuadability"].to_numpy()
    eng = latent["engagement"].to_numpy()
    target_rows = []
    event_rows = []
    for ci, cid in enumerate(cids):
        # ~35% of customers targeted per campaign
        mask = rng.random(cfg.n_customers) < 0.35
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        ctrl_pct = float(campaigns.loc[ci, "control_group_pct"])
        is_control = rng.random(len(idx)) < ctrl_pct
        assign = pd.Timestamp(campaigns.loc[ci, "start_date"]) + pd.to_timedelta(
            rng.integers(0, 14 * 24 * 60, len(idx)), "m"
        )
        base_conv = 0.08 + 0.12 * eng[idx]  # organic conversion
        tau = 0.20 * persuade[idx]  # treatment effect, heterogeneous
        treated = ~is_control
        p_conv = base_conv + treated * tau
        conv30 = rng.random(len(idx)) < np.clip(p_conv, 0, 0.95)
        conv7 = conv30 & (rng.random(len(idx)) < 0.6)
        seg = np.select([eng[idx] < 0.33, eng[idx] < 0.66], ["at_risk", "steady"], default="loyal")
        target_rows.append(
            pd.DataFrame(
                {
                    "campaign_id": cid,
                    "customer_id": customers["customer_id"].to_numpy()[idx],
                    "assignment_datetime": _ts_str(assign),
                    "treatment_flag": treated,
                    "control_flag": is_control,
                    "eligibility_flag": True,
                    "predicted_business_segment_at_send": seg,
                    "targeting_rule_source": "merch_rule_v2",
                }
            )
        )
        delivered = treated & (rng.random(len(idx)) < 0.97)
        opened = delivered & (rng.random(len(idx)) < 0.5)
        clicked = opened & (rng.random(len(idx)) < 0.4)
        rev30 = np.where(conv30, np.round(rng.gamma(2, 30, len(idx)), 2), 0.0)
        event_rows.append(
            pd.DataFrame(
                {
                    "campaign_id": cid,
                    "customer_id": customers["customer_id"].to_numpy()[idx],
                    "delivered_flag": delivered,
                    "open_flag": opened,
                    "click_flag": clicked,
                    "unsubscribe_flag": rng.random(len(idx)) < 0.01,
                    "conversion_within_7d": conv7,
                    "conversion_within_30d": conv30,
                    "revenue_within_7d": np.where(conv7, rev30, 0.0),
                    "revenue_within_30d": rev30,
                }
            )
        )
    targets = pd.concat(target_rows, ignore_index=True)
    events = pd.concat(event_rows, ignore_index=True)
    return campaigns, targets, events


# ── Daily prices + inventory (forecast features) ────────────────────────────
def gen_daily(cfg, rng, orders, order_items, products, start, anchor):
    # active (product, location) pairs come from completed orders
    completed = orders[orders["order_status"] == "completed"][
        ["order_id", "channel", "store_id_nullable"]
    ].copy()
    completed["store_id_or_online"] = np.where(
        completed["channel"] == "store", completed["store_id_nullable"], "ONLINE"
    )
    it = order_items.merge(completed, on="order_id", how="inner")
    pairs = it[["product_id", "store_id_or_online"]].drop_duplicates().reset_index(drop=True)
    pairs = pairs.dropna()

    days = pd.date_range(start, anchor, freq="D")
    n_days = len(days)
    n_pairs = len(pairs)
    prod_ids = np.repeat(pairs["product_id"].to_numpy(), n_days)
    loc_ids = np.repeat(pairs["store_id_or_online"].to_numpy(), n_days)
    date_col = np.tile(_date_str(days), n_pairs)

    base = products.set_index("product_id")["base_price"]
    listed = base.reindex(pairs["product_id"]).to_numpy()
    listed = np.repeat(listed, n_days) * (1 + rng.normal(0, 0.05, n_pairs * n_days))
    listed = np.round(np.clip(listed, 1, None), 2)
    disc = np.round(np.clip(rng.beta(1.2, 8, n_pairs * n_days), 0, 0.6), 4)
    promo = rng.random(n_pairs * n_days) < 0.12
    prices = pd.DataFrame(
        {
            "date": date_col,
            "product_id": prod_ids,
            "store_id_or_online": loc_ids,
            "listed_price": listed,
            "discount_pct": disc,
            "promo_flag": promo,
            "markdown_flag": rng.random(n_pairs * n_days) < 0.06,
            "bundle_flag": rng.random(n_pairs * n_days) < 0.05,
        }
    )
    start_inv = rng.integers(5, 80, n_pairs * n_days)
    inv = pd.DataFrame(
        {
            "date": date_col,
            "product_id": prod_ids,
            "store_id_or_online": loc_ids,
            "starting_inventory": start_inv,
            "ending_inventory": np.clip(start_inv - rng.integers(0, 10, n_pairs * n_days), 0, None),
            "stock_received": rng.integers(0, 20, n_pairs * n_days),
            "stockout_flag": rng.random(n_pairs * n_days) < 0.04,
            "backorder_flag": rng.random(n_pairs * n_days) < 0.02,
        }
    )
    return prices, inv


# ── Orchestration ───────────────────────────────────────────────────────────
def generate(cfg: Config) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    anchor = pd.Timestamp("2025-09-30")
    start = (anchor - pd.DateOffset(months=cfg.months)).normalize()

    print(
        f"Generating scale: customers={cfg.n_customers} products={cfg.n_products} "
        f"stores={cfg.n_stores} range={start.date()}..{anchor.date()} seed={cfg.seed}"
    )

    calendar = gen_calendar(start, anchor)
    stores = gen_stores(cfg, rng)
    products, attrs = gen_products(cfg, rng)
    customers, latent = gen_customers(cfg, rng, anchor, start)
    orders, order_items, _, _ = gen_orders(
        cfg, rng, customers, latent, products, stores, anchor, start
    )
    returns, hidden = gen_returns(cfg, rng, orders, order_items, products, customers, latent)
    sessions, events = gen_sessions(cfg, rng, customers, latent, orders, products, anchor, start)
    # link a subset of online orders to sessions
    reviews = gen_reviews(cfg, rng, customers, products, orders, order_items, anchor, start)
    campaigns, targets, camp_events = gen_campaigns(cfg, rng, customers, latent, anchor, start)
    prices, inventory = gen_daily(cfg, rng, orders, order_items, products, start, anchor)

    tables = {
        "calendar": calendar,
        "stores": stores,
        "products": products,
        "product_attributes": attrs,
        "customers": customers,
        "orders": orders,
        "order_items": order_items,
        "returns": returns,
        "reviews": reviews,
        "web_sessions": sessions,
        "session_events": events,
        "campaigns": campaigns,
        "campaign_targets": targets,
        "campaign_events": camp_events,
        "daily_prices": prices,
        "daily_inventory": inventory,
    }
    for name, df in tables.items():
        path = RAW_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"  wrote {name:22s} {len(df):>10,} rows")

    hidden.to_csv(PROCESSED_DIR / "returns_hidden_labels.csv", index=False)
    print(
        f"  wrote returns_hidden_labels {len(hidden):>6,} rows "
        f"({hidden['abuse_flag_hidden_for_validation'].mean() * 100:.2f}% abuse)"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic retail raw data")
    p.add_argument("--scale", choices=list(SCALES), default="default")
    p.add_argument("--n-customers", type=int, default=None)
    p.add_argument("--n-products", type=int, default=None)
    p.add_argument("--n-stores", type=int, default=None)
    p.add_argument("--months", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    preset = SCALES[args.scale]
    cfg = Config(
        n_customers=args.n_customers or preset["n_customers"],
        n_products=args.n_products or preset["n_products"],
        n_stores=args.n_stores or preset["n_stores"],
        months=args.months or preset["months"],
        seed=args.seed,
    )
    generate(cfg)


if __name__ == "__main__":
    main()
