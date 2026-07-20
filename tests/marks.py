"""Shared pytest markers.

Raw and processed data are gitignored (~1GB) and not committed, so tests that
load real marts are skipped when the processed marts are absent (e.g. on CI).
Data-independent unit tests build in-memory frames and always run.
"""

from __future__ import annotations

import pytest

from src.config import PROCESSED_DIR
from src.data.mart_schemas import MART_FILE_NAMES


def marts_available() -> bool:
    return (PROCESSED_DIR / MART_FILE_NAMES["mart_customer_features"]).exists()


requires_marts = pytest.mark.skipif(
    not marts_available(),
    reason="processed marts absent (data is gitignored ~1GB); build locally to run",
)
