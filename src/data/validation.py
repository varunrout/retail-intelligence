import pandas as pd


def null_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return null count and null rate per column."""
    summary = pd.DataFrame(
        {
            "column": df.columns,
            "null_count": df.isna().sum().values,
        }
    )
    summary["null_rate"] = summary["null_count"] / len(df) if len(df) else 0.0
    return summary.sort_values("null_rate", ascending=False)


def duplicate_count(df: pd.DataFrame, keys: list[str]) -> int:
    """Count duplicated rows over a candidate key set."""
    return int(df.duplicated(subset=keys).sum())
