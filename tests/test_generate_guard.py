"""Tests for the data/raw overwrite guard (raw_data_exists + --force).

Data-independent: uses a temp directory standing in for RAW_DIR, never touches
the real (gitignored) data/ tree.
"""

from __future__ import annotations

import pandas as pd

from src.data.generate import raw_data_exists


def test_raw_data_exists_false_on_empty_dir(tmp_path) -> None:
    assert raw_data_exists(tmp_path) is False


def test_raw_data_exists_true_once_customers_written(tmp_path) -> None:
    pd.DataFrame({"customer_id": ["C000001"]}).to_csv(tmp_path / "customers.csv", index=False)
    assert raw_data_exists(tmp_path) is True


def test_raw_data_exists_ignores_unrelated_files(tmp_path) -> None:
    (tmp_path / "notes.txt").write_text("scratch")
    assert raw_data_exists(tmp_path) is False
