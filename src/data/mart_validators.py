from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.mart_schemas import MART_CONTRACTS, MartContract


def _append_result(
    results: list[dict[str, Any]],
    mart_name: str,
    check_name: str,
    failed_rows: int,
    details: str = "",
) -> None:
    results.append(
        {
            "mart_name": mart_name,
            "check_name": check_name,
            "failed_rows": int(failed_rows),
            "status": "PASS" if int(failed_rows) == 0 else "FAIL",
            "details": details,
        }
    )


def validate_mart(df: pd.DataFrame, contract: MartContract) -> pd.DataFrame:
    results: list[dict[str, Any]] = []

    missing_required = [col for col in contract.required_columns if col not in df.columns]
    _append_result(
        results,
        contract.mart_name,
        "missing_required_columns",
        0 if not missing_required else len(missing_required),
        details=",".join(missing_required),
    )
    if missing_required:
        return pd.DataFrame(results)

    missing_key_cols = [col for col in contract.key_columns if col not in df.columns]
    _append_result(
        results,
        contract.mart_name,
        "missing_key_columns",
        0 if not missing_key_cols else len(missing_key_cols),
        details=",".join(missing_key_cols),
    )

    if not missing_key_cols:
        duplicate_rows = int(df.duplicated(subset=contract.key_columns).sum())
        _append_result(results, contract.mart_name, "duplicate_key_rows", duplicate_rows)

        key_null_rows = int(df[contract.key_columns].isna().any(axis=1).sum())
        _append_result(results, contract.mart_name, "null_rows_in_key_columns", key_null_rows)

    for col in contract.non_negative_columns:
        if col not in df.columns:
            _append_result(results, contract.mart_name, f"missing_non_negative_column::{col}", 1)
            continue
        invalid_count = int((df[col] < 0).fillna(False).sum())
        _append_result(results, contract.mart_name, f"negative_values::{col}", invalid_count)

    for col in contract.zero_to_one_columns:
        if col not in df.columns:
            _append_result(results, contract.mart_name, f"missing_zero_to_one_column::{col}", 1)
            continue
        invalid_count = int(((df[col] < 0) | (df[col] > 1)).fillna(False).sum())
        _append_result(results, contract.mart_name, f"outside_0_1_range::{col}", invalid_count)

    for col in contract.date_columns:
        if col not in df.columns:
            _append_result(results, contract.mart_name, f"missing_date_column::{col}", 1)
            continue
        null_date_rows = int(df[col].isna().sum())
        _append_result(results, contract.mart_name, f"null_date_rows::{col}", null_date_rows)

    return pd.DataFrame(results)


def validate_all_marts(marts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for mart_name, df in marts.items():
        contract = MART_CONTRACTS[mart_name]
        frames.append(validate_mart(df, contract))
    if not frames:
        return pd.DataFrame(columns=["mart_name", "check_name", "failed_rows", "status", "details"])
    return pd.concat(frames, ignore_index=True)


def summarize_validation_results(validation_df: pd.DataFrame) -> pd.DataFrame:
    if validation_df.empty:
        return pd.DataFrame(columns=["mart_name", "total_checks", "failed_checks", "status"])

    grouped = validation_df.groupby("mart_name", as_index=False).agg(
        total_checks=("check_name", "count"),
        failed_checks=("status", lambda s: int((s == "FAIL").sum())),
    )
    grouped["status"] = grouped["failed_checks"].apply(lambda x: "PASS" if x == 0 else "FAIL")
    return grouped.sort_values("mart_name")
