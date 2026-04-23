from src.config import INTERIM_DIR, OUTPUTS_DIR, PROCESSED_DIR, ensure_core_dirs


def test_ensure_core_dirs_creates_paths() -> None:
    ensure_core_dirs()
    assert INTERIM_DIR.exists()
    assert PROCESSED_DIR.exists()
    assert OUTPUTS_DIR.exists()
