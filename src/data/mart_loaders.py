from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR
from src.data.mart_schemas import MART_CONTRACTS, MART_FILE_NAMES


def mart_path(mart_name: str, processed_dir: Path = PROCESSED_DIR) -> Path:
    if mart_name not in MART_FILE_NAMES:
        raise ValueError(f"Unknown mart name: {mart_name}")
    path = processed_dir / MART_FILE_NAMES[mart_name]
    if not path.exists():
        raise FileNotFoundError(f"Mart file not found: {path}")
    return path


def load_mart(mart_name: str, processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    contract = MART_CONTRACTS[mart_name]
    path = mart_path(mart_name, processed_dir)
    parse_dates = contract.date_columns if contract.date_columns else None
    return pd.read_csv(path, low_memory=False, parse_dates=parse_dates)


def load_all_marts(processed_dir: Path = PROCESSED_DIR) -> dict[str, pd.DataFrame]:
    return {name: load_mart(name, processed_dir=processed_dir) for name in MART_CONTRACTS}
