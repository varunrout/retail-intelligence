from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import duckdb
import pandas as pd

from src.config import OUTPUTS_DIR, RAW_DIR


@dataclass(frozen=True)
class TableSpec:
    name: str
    key_columns: list[str]
    date_columns: list[str]


TABLE_SPECS: list[TableSpec] = [
    TableSpec("calendar", ["date"], ["date"]),
    TableSpec("campaign_events", ["campaign_id", "customer_id"], ["event_date"]),
    TableSpec("campaign_targets", ["campaign_id", "customer_id"], ["assignment_datetime"]),
    TableSpec("campaigns", ["campaign_id"], ["start_date", "end_date"]),
    TableSpec("customers", ["customer_id"], ["signup_date", "last_purchase_date"]),
    TableSpec("daily_inventory", ["date", "product_id", "store_id_or_online"], ["date"]),
    TableSpec("daily_prices", ["date", "product_id", "store_id_or_online"], ["date"]),
    TableSpec("order_items", ["order_item_id"], ["item_fulfillment_date"]),
    TableSpec("orders", ["order_id"], ["order_datetime", "order_date"]),
    TableSpec("product_attributes", ["product_id"], []),
    TableSpec("products", ["product_id"], ["launch_date", "discontinued_date"]),
    TableSpec("returns", ["return_id"], ["return_request_date", "return_processed_date"]),
    TableSpec("reviews", ["review_id"], ["review_date"]),
    TableSpec("session_events", ["event_id"], ["event_time"]),
    TableSpec("stores", ["store_id"], ["opening_date"]),
    TableSpec("web_sessions", ["session_id"], ["session_start", "session_end"]),
]


def _quote_ident(name: str) -> str:
    return f'"{name}"'


def _csv_path(table_name: str) -> str:
    return str((RAW_DIR / f"{table_name}.csv").resolve()).replace("\\", "/")


def _existing_columns(
    con: duckdb.DuckDBPyConnection, table_name: str, columns: Iterable[str]
) -> list[str]:
    table_info = con.execute(f"PRAGMA table_info({_quote_ident(table_name)})").fetchdf()
    existing = set(table_info["name"].tolist())
    return [c for c in columns if c in existing]


def _scalar_count(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    row = con.execute(sql).fetchone()
    if row is None:
        return 0
    return int(row[0])


def _load_raw_views(con: duckdb.DuckDBPyConnection) -> None:
    for spec in TABLE_SPECS:
        path = _csv_path(spec.name)
        con.execute(
            f"""
            CREATE OR REPLACE VIEW {_quote_ident(spec.name)} AS
            SELECT * FROM read_csv_auto('{path}', header=true, sample_size=-1)
            """
        )


def build_table_profile(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for spec in TABLE_SPECS:
        row_count = _scalar_count(con, f"SELECT COUNT(*) FROM {_quote_ident(spec.name)}")
        table_info = con.execute(f"PRAGMA table_info({_quote_ident(spec.name)})").fetchdf()
        cols = table_info["name"].tolist()

        key_cols = [c for c in spec.key_columns if c in cols]
        date_cols = [c for c in spec.date_columns if c in cols]

        date_min = None
        date_max = None
        if date_cols:
            # Use the first declared date column as the canonical range tracker.
            dcol = date_cols[0]
            date_row = con.execute(
                f"SELECT MIN(TRY_CAST({_quote_ident(dcol)} AS TIMESTAMP)), MAX(TRY_CAST({_quote_ident(dcol)} AS TIMESTAMP)) FROM {_quote_ident(spec.name)}"
            ).fetchone()
            if date_row is None:
                date_min, date_max = None, None
            else:
                date_min, date_max = date_row

        records.append(
            {
                "table_name": spec.name,
                "row_count": int(row_count),
                "column_count": int(len(cols)),
                "candidate_key": ",".join(key_cols) if key_cols else "",
                "canonical_date_column": date_cols[0] if date_cols else "",
                "date_min": str(date_min) if date_min is not None else "",
                "date_max": str(date_max) if date_max is not None else "",
            }
        )
    return pd.DataFrame(records).sort_values("table_name")


def build_key_checks(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for spec in TABLE_SPECS:
        existing_keys = _existing_columns(con, spec.name, spec.key_columns)
        if not existing_keys:
            records.append(
                {
                    "table_name": spec.name,
                    "key_columns": "",
                    "row_count": _scalar_count(
                        con, f"SELECT COUNT(*) FROM {_quote_ident(spec.name)}"
                    ),
                    "distinct_key_count": None,
                    "duplicate_rows_on_key": None,
                    "key_null_rows": None,
                    "key_null_rate": None,
                }
            )
            continue

        key_expr = ", ".join(_quote_ident(c) for c in existing_keys)
        null_pred = " OR ".join(f"{_quote_ident(c)} IS NULL" for c in existing_keys)
        row_count = _scalar_count(con, f"SELECT COUNT(*) FROM {_quote_ident(spec.name)}")
        distinct_count = _scalar_count(
            con,
            f"SELECT COUNT(*) FROM (SELECT DISTINCT {key_expr} FROM {_quote_ident(spec.name)})",
        )
        null_rows = _scalar_count(
            con,
            f"SELECT COUNT(*) FROM {_quote_ident(spec.name)} WHERE {null_pred}",
        )
        records.append(
            {
                "table_name": spec.name,
                "key_columns": ",".join(existing_keys),
                "row_count": row_count,
                "distinct_key_count": distinct_count,
                "duplicate_rows_on_key": row_count - distinct_count,
                "key_null_rows": null_rows,
                "key_null_rate": (null_rows / row_count) if row_count else 0.0,
            }
        )

    return pd.DataFrame(records).sort_values("table_name")


def build_join_validations(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    checks = [
        {
            "check_name": "orders_without_customer",
            "sql": """
                SELECT COUNT(*)
                FROM orders o
                LEFT JOIN customers c ON o.customer_id = c.customer_id
                WHERE c.customer_id IS NULL
            """,
            "expected": "0",
            "notes": "Order facts should resolve to customer dimension.",
        },
        {
            "check_name": "order_items_without_order",
            "sql": """
                SELECT COUNT(*)
                FROM order_items oi
                LEFT JOIN orders o ON oi.order_id = o.order_id
                WHERE o.order_id IS NULL
            """,
            "expected": "0",
            "notes": "Order item facts should resolve to order header.",
        },
        {
            "check_name": "order_items_without_product",
            "sql": """
                SELECT COUNT(*)
                FROM order_items oi
                LEFT JOIN products p ON oi.product_id = p.product_id
                WHERE p.product_id IS NULL
            """,
            "expected": "0",
            "notes": "Order item facts should resolve to product dimension.",
        },
        {
            "check_name": "returns_without_order_item",
            "sql": """
                SELECT COUNT(*)
                FROM returns r
                LEFT JOIN order_items oi ON r.order_item_id = oi.order_item_id
                WHERE oi.order_item_id IS NULL
            """,
            "expected": "0",
            "notes": "Returns should map back to transacted order items.",
        },
        {
            "check_name": "campaign_targets_without_campaign",
            "sql": """
                SELECT COUNT(*)
                FROM campaign_targets ct
                LEFT JOIN campaigns c ON ct.campaign_id = c.campaign_id
                WHERE c.campaign_id IS NULL
            """,
            "expected": "0",
            "notes": "Campaign targeting should resolve to campaign metadata.",
        },
        {
            "check_name": "campaign_events_without_target",
            "sql": """
                SELECT COUNT(*)
                FROM campaign_events ce
                LEFT JOIN campaign_targets ct
                  ON ce.campaign_id = ct.campaign_id
                 AND ce.customer_id = ct.customer_id
                WHERE ct.customer_id IS NULL
            """,
            "expected": "0",
            "notes": "Events should map to targeted campaign assignments.",
        },
        {
            "check_name": "store_orders_without_store_id",
            "sql": """
                SELECT COUNT(*)
                FROM orders
                WHERE LOWER(channel) = 'store' AND store_id_nullable IS NULL
            """,
            "expected": "0",
            "notes": "Store channel orders should have a store id.",
        },
        {
            "check_name": "store_orders_store_not_found",
            "sql": """
                SELECT COUNT(*)
                FROM orders o
                LEFT JOIN stores s ON o.store_id_nullable = s.store_id
                WHERE LOWER(o.channel) = 'store'
                  AND o.store_id_nullable IS NOT NULL
                  AND s.store_id IS NULL
            """,
            "expected": "0",
            "notes": "Store ids in orders should resolve to store dimension.",
        },
        {
            "check_name": "inventory_without_product",
            "sql": """
                SELECT COUNT(*)
                FROM daily_inventory di
                LEFT JOIN products p ON di.product_id = p.product_id
                WHERE p.product_id IS NULL
            """,
            "expected": "0",
            "notes": "Inventory snapshots should resolve to products.",
        },
        {
            "check_name": "prices_without_product",
            "sql": """
                SELECT COUNT(*)
                FROM daily_prices dp
                LEFT JOIN products p ON dp.product_id = p.product_id
                WHERE p.product_id IS NULL
            """,
            "expected": "0",
            "notes": "Price snapshots should resolve to products.",
        },
    ]

    rows: list[dict[str, object]] = []
    for check in checks:
        value = _scalar_count(con, check["sql"])
        rows.append(
            {
                "check_name": check["check_name"],
                "failed_rows": value,
                "expected_failed_rows": check["expected"],
                "status": "PASS" if value == 0 else "FAIL",
                "notes": check["notes"],
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    try:
        _load_raw_views(con)
        table_profile = build_table_profile(con)
        key_checks = build_key_checks(con)
        join_checks = build_join_validations(con)

        table_profile.to_csv(OUTPUTS_DIR / "phase2_table_profile_summary.csv", index=False)
        key_checks.to_csv(OUTPUTS_DIR / "phase2_key_checks_summary.csv", index=False)
        join_checks.to_csv(OUTPUTS_DIR / "phase2_join_validation_summary.csv", index=False)

        print("Wrote:")
        print(OUTPUTS_DIR / "phase2_table_profile_summary.csv")
        print(OUTPUTS_DIR / "phase2_key_checks_summary.csv")
        print(OUTPUTS_DIR / "phase2_join_validation_summary.csv")
    finally:
        con.close()


if __name__ == "__main__":
    main()
