"""Data-independent unit tests for the mart validators and schemas.

These build tiny in-memory frames and never touch the gitignored data, so they
run in CI on a clean clone and give the contract layer real coverage.
"""

from __future__ import annotations

import pandas as pd

from src.data.mart_schemas import MART_CONTRACTS, MART_FILE_NAMES, MartContract
from src.data.mart_validators import (
    summarize_validation_results,
    validate_all_marts,
    validate_mart,
)

_CONTRACT = MartContract(
    mart_name="toy_mart",
    required_columns=["id", "amount", "share", "event_date"],
    key_columns=["id"],
    date_columns=["event_date"],
    non_negative_columns=["amount"],
    zero_to_one_columns=["share"],
)


def _clean_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "amount": [10.0, 0.0, 5.5],
            "share": [0.1, 0.9, 0.5],
            "event_date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
        }
    )


def _failed(result: pd.DataFrame, check: str) -> int:
    row = result[result["check_name"] == check]
    assert not row.empty, f"check missing: {check}"
    return int(row["failed_rows"].iloc[0])


def test_clean_frame_passes_every_check() -> None:
    result = validate_mart(_clean_frame(), _CONTRACT)
    assert (result["status"] == "PASS").all()


def test_duplicate_key_is_detected() -> None:
    df = _clean_frame()
    df.loc[2, "id"] = 1  # duplicate key
    assert _failed(validate_mart(df, _CONTRACT), "duplicate_key_rows") == 1


def test_negative_value_is_detected() -> None:
    df = _clean_frame()
    df.loc[0, "amount"] = -1.0
    assert _failed(validate_mart(df, _CONTRACT), "negative_values::amount") == 1


def test_out_of_range_share_is_detected() -> None:
    df = _clean_frame()
    df.loc[0, "share"] = 1.4
    assert _failed(validate_mart(df, _CONTRACT), "outside_0_1_range::share") == 1


def test_null_key_is_detected() -> None:
    df = _clean_frame()
    df.loc[0, "id"] = None
    assert _failed(validate_mart(df, _CONTRACT), "null_rows_in_key_columns") == 1


def test_missing_required_column_short_circuits() -> None:
    df = _clean_frame().drop(columns=["amount"])
    result = validate_mart(df, _CONTRACT)
    assert _failed(result, "missing_required_columns") > 0


def test_summary_flags_failing_mart() -> None:
    df = _clean_frame()
    df.loc[2, "id"] = 1
    summary = summarize_validation_results(validate_mart(df, _CONTRACT))
    assert summary["status"].iloc[0] == "FAIL"


def test_validate_all_marts_on_empty_returns_columns() -> None:
    out = validate_all_marts({})
    assert list(out.columns) == ["mart_name", "check_name", "failed_rows", "status", "details"]


def test_every_contract_has_a_file_name() -> None:
    for name in MART_CONTRACTS:
        assert name in MART_FILE_NAMES
        assert MART_FILE_NAMES[name].endswith(".csv")
