from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR, SQL_DIR

MARTS = [
    "mart_customer_features",
    "mart_product_demand",
    "mart_campaign_response",
    "mart_returns_risk",
    "mart_recommendation_interactions",
    "mart_store_week_performance",
]

RAW_TABLES = [
    "calendar",
    "campaign_events",
    "campaign_targets",
    "campaigns",
    "customers",
    "daily_inventory",
    "daily_prices",
    "order_items",
    "orders",
    "product_attributes",
    "products",
    "returns",
    "reviews",
    "session_events",
    "stores",
    "web_sessions",
]


def _read_sql(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_raw_views(con: duckdb.DuckDBPyConnection) -> None:
    for table_name in RAW_TABLES:
        csv_path = (RAW_DIR / f"{table_name}.csv").resolve().as_posix()
        con.execute(
            f"""
            CREATE OR REPLACE VIEW {table_name} AS
            SELECT * FROM read_csv_auto('{csv_path}', header=true, sample_size=-1)
            """
        )


def _run_sql_file(con: duckdb.DuckDBPyConnection, filename: str) -> None:
    sql_path = SQL_DIR / filename
    con.execute(_read_sql(sql_path))


def _table_exists(con: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    row = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
        [table_name],
    ).fetchone()
    return bool(row and row[0] > 0)


def _scalar_int(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    row = con.execute(sql).fetchone()
    if row is None:
        return 0
    return int(row[0])


def _build_row_counts(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    rows: list[dict[str, int | str]] = []
    for mart in MARTS:
        count = _scalar_int(con, f"SELECT COUNT(*) FROM {mart}")
        rows.append({"mart_name": mart, "row_count": count})
    return pd.DataFrame(rows).sort_values("mart_name")


def _build_quality_checks(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    checks = [
        (
            "mart_customer_features_duplicate_customer_id",
            "SELECT COUNT(*) - COUNT(DISTINCT customer_id) FROM mart_customer_features",
            0,
        ),
        (
            "mart_product_demand_duplicate_grain",
            """
            SELECT COALESCE(SUM(cnt - 1), 0)
            FROM (
                SELECT COUNT(*) AS cnt
                FROM mart_product_demand
                GROUP BY week_start_date, product_id, store_id_or_online
                HAVING COUNT(*) > 1
            ) d
            """,
            0,
        ),
        (
            "mart_campaign_response_duplicate_grain",
            """
            SELECT COALESCE(SUM(cnt - 1), 0)
            FROM (
                SELECT COUNT(*) AS cnt
                FROM mart_campaign_response
                GROUP BY campaign_id, customer_id
                HAVING COUNT(*) > 1
            ) d
            """,
            0,
        ),
        (
            "mart_returns_risk_duplicate_order_item",
            "SELECT COUNT(*) - COUNT(DISTINCT order_item_id) FROM mart_returns_risk",
            0,
        ),
        (
            "mart_reco_duplicate_customer_product",
            """
            SELECT COALESCE(SUM(cnt - 1), 0)
            FROM (
                SELECT COUNT(*) AS cnt
                FROM mart_recommendation_interactions
                GROUP BY customer_id, product_id
                HAVING COUNT(*) > 1
            ) d
            """,
            0,
        ),
        (
            "mart_store_week_duplicate_grain",
            """
            SELECT COALESCE(SUM(cnt - 1), 0)
            FROM (
                SELECT COUNT(*) AS cnt
                FROM mart_store_week_performance
                GROUP BY week_start_date, store_id_or_online
                HAVING COUNT(*) > 1
            ) d
            """,
            0,
        ),
        (
            "mart_customer_features_null_customer_id",
            "SELECT COUNT(*) FROM mart_customer_features WHERE customer_id IS NULL",
            0,
        ),
        (
            "mart_campaign_response_null_campaign_or_customer",
            "SELECT COUNT(*) FROM mart_campaign_response WHERE campaign_id IS NULL OR customer_id IS NULL",
            0,
        ),
    ]

    rows: list[dict[str, int | str]] = []
    for name, sql, expected in checks:
        value = _scalar_int(con, sql)
        rows.append(
            {
                "check_name": name,
                "failed_rows": value,
                "expected_failed_rows": int(expected),
                "status": "PASS" if value == int(expected) else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    db_path = PROCESSED_DIR / "retail_intelligence.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        _load_raw_views(con)

        _run_sql_file(con, "staging_shared.sql")
        for mart in MARTS:
            _run_sql_file(con, f"{mart}.sql")

        for mart in MARTS:
            if not _table_exists(con, mart):
                raise RuntimeError(f"Expected mart table was not created: {mart}")

        for mart in MARTS:
            out_csv = (PROCESSED_DIR / f"{mart}.csv").resolve().as_posix()
            con.execute(f"COPY (SELECT * FROM {mart}) TO '{out_csv}' (HEADER, DELIMITER ',')")

        row_counts = _build_row_counts(con)
        quality = _build_quality_checks(con)

        row_counts.to_csv(PROCESSED_DIR / "phase3_mart_row_counts.csv", index=False)
        quality.to_csv(PROCESSED_DIR / "phase3_mart_quality_checks.csv", index=False)

        print(f"DuckDB mart database: {db_path}")
        print("Exported marts and validation summaries to data/processed")
    finally:
        con.close()


if __name__ == "__main__":
    main()
