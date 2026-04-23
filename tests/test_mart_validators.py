import pandas as pd

from src.config import PROCESSED_DIR
from src.data.mart_loaders import load_mart
from src.data.mart_schemas import MART_CONTRACTS
from src.data.mart_validators import summarize_validation_results, validate_mart


def test_validate_customer_mart_passes_key_checks() -> None:
    df = load_mart("mart_customer_features", processed_dir=PROCESSED_DIR)
    contract = MART_CONTRACTS["mart_customer_features"]

    result = validate_mart(df, contract)
    duplicate_key_check = result[result["check_name"] == "duplicate_key_rows"]
    assert not duplicate_key_check.empty
    assert int(duplicate_key_check["failed_rows"].iloc[0]) == 0


def test_validator_fails_when_required_column_missing() -> None:
    contract = MART_CONTRACTS["mart_customer_features"]
    broken_df = pd.DataFrame({"customer_id": ["C1"]})

    result = validate_mart(broken_df, contract)
    missing_required = result[result["check_name"] == "missing_required_columns"]
    assert not missing_required.empty
    assert int(missing_required["failed_rows"].iloc[0]) > 0


def test_validation_summary_has_status_column() -> None:
    df = load_mart("mart_campaign_response", processed_dir=PROCESSED_DIR)
    contract = MART_CONTRACTS["mart_campaign_response"]
    details = validate_mart(df, contract)
    summary = summarize_validation_results(details)
    assert "status" in summary.columns
