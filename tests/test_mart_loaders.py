from src.config import PROCESSED_DIR
from src.data.mart_loaders import load_all_marts, load_mart, mart_path


def test_mart_path_exists_for_customer_features() -> None:
    path = mart_path("mart_customer_features", processed_dir=PROCESSED_DIR)
    assert path.exists()


def test_load_single_mart_has_expected_column() -> None:
    df = load_mart("mart_customer_features", processed_dir=PROCESSED_DIR)
    assert "customer_id" in df.columns
    assert len(df) > 0


def test_load_all_marts_returns_all_contracts() -> None:
    marts = load_all_marts(processed_dir=PROCESSED_DIR)
    assert "mart_campaign_response" in marts
    assert "mart_store_week_performance" in marts
    assert len(marts) == 6
