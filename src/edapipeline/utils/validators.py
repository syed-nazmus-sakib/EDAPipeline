"""Input validation utilities."""

from __future__ import annotations

from typing import List, Optional

import pandas as pd


def validate_dataframe(df: pd.DataFrame) -> None:
    """Validate that the input is a non-empty DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    if df.empty:
        raise ValueError("DataFrame is empty. Provide a DataFrame with at least one row.")


def validate_columns_exist(
    df: pd.DataFrame,
    columns: List[str],
    label: str = "columns",
) -> None:
    """Ensure all specified columns exist in the DataFrame."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"The following {label} are not in the DataFrame: {missing}")


def validate_target_column(
    df: pd.DataFrame,
    target_col: Optional[str],
) -> None:
    """Validate the target column if provided."""
    if target_col is not None and target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )
