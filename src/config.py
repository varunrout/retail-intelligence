from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SQL_DIR = PROJECT_ROOT / "sql"
DOCS_DIR = PROJECT_ROOT / "docs"


def ensure_core_dirs() -> None:
    """Create core project directories used by pipeline outputs."""
    for directory in (INTERIM_DIR, PROCESSED_DIR, OUTPUTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
