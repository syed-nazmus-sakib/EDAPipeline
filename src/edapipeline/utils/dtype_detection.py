"""Smart column type detection for DataFrames."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from ..types import ColumnProfile, ColumnType, DatasetProfile


def detect_column_types(
    df: pd.DataFrame,
    target_col: str | None = None,
) -> DatasetProfile:
    """Analyze a DataFrame and classify every column by type.

    Returns a :class:`DatasetProfile` with per-column metadata and
    grouped column lists (numerical, categorical, datetime, boolean).
    """
    numerical: List[str] = []
    categorical: List[str] = []
    datetime_cols: List[str] = []
    boolean_cols: List[str] = []
    profiles: dict[str, ColumnProfile] = {}

    for col in df.columns:
        dtype = df[col].dtype
        col_type = _classify_column(df[col])

        profile = ColumnProfile(
            name=col,
            dtype=str(dtype),
            column_type=col_type,
            null_count=int(df[col].isnull().sum()),
            null_percentage=float(df[col].isnull().mean() * 100),
            unique_count=int(df[col].nunique()),
            memory_bytes=int(df[col].memory_usage(deep=True)),
            sample_values=df[col].dropna().head(5).tolist(),
        )
        profiles[col] = profile

        # Skip target column when building feature lists
        if col == target_col:
            continue

        if col_type == ColumnType.NUMERICAL:
            numerical.append(col)
        elif col_type == ColumnType.CATEGORICAL:
            categorical.append(col)
        elif col_type == ColumnType.DATETIME:
            datetime_cols.append(col)
        elif col_type == ColumnType.BOOLEAN:
            boolean_cols.append(col)

    memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

    return DatasetProfile(
        n_rows=len(df),
        n_cols=len(df.columns),
        memory_mb=round(memory_mb, 2),
        column_profiles=profiles,
        numerical_columns=numerical,
        categorical_columns=categorical,
        datetime_columns=datetime_cols,
        boolean_columns=boolean_cols,
        target_column=target_col,
    )


def _classify_column(series: pd.Series) -> ColumnType:
    """Classify a single Series into a ColumnType."""
    dtype = series.dtype

    # Boolean
    if pd.api.types.is_bool_dtype(dtype):
        return ColumnType.BOOLEAN

    # Datetime
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return ColumnType.DATETIME

    # Numeric
    if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
        return ColumnType.NUMERICAL

    # Category dtype
    if pd.api.types.is_categorical_dtype(dtype):
        return ColumnType.CATEGORICAL

    # Object — try datetime conversion, else categorical
    if dtype == object:
        if _could_be_datetime(series):
            return ColumnType.DATETIME
        return ColumnType.CATEGORICAL

    return ColumnType.CATEGORICAL


def _could_be_datetime(series: pd.Series, sample_size: int = 50) -> bool:
    """Heuristic check: can a sample of non-null values parse as datetime?"""
    non_null = series.dropna()
    if len(non_null) == 0:
        return False

    sample = non_null.head(sample_size)
    try:
        pd.to_datetime(sample, errors="raise", infer_datetime_format=True)
        return True
    except (ValueError, TypeError, OverflowError):
        return False
