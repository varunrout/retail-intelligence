from pathlib import Path

import pandas as pd


def load_csv(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    """Load a CSV with optional date parsing and basic safety options."""
    return pd.read_csv(path, parse_dates=parse_dates, low_memory=False)


def load_raw_table(table_name: str, raw_dir: Path) -> pd.DataFrame:
    """Load a table from data/raw by name without extension."""
    file_path = raw_dir / f"{table_name}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Raw table file not found: {file_path}")
    return load_csv(file_path)
