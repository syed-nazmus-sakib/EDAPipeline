"""Backward-compatible wrapper preserving the old EDAPipeline API."""

from __future__ import annotations

import warnings
from typing import List, Optional

import pandas as pd

from .config import PipelineConfig
from .pipeline import Pipeline
from .results import PipelineReport
from .types import OutlierMethod


class EDAPipeline:
    """Legacy wrapper for backward compatibility.

    .. deprecated::
        Use :class:`edapipeline.Pipeline` instead.
        ``EDAPipeline`` will be removed in v2.0.

    This wrapper maps the old API to the new modular Pipeline:

    >>> eda = EDAPipeline(df=df, target_col="target", save_outputs=True)
    >>> eda.run_complete_analysis(outlier_method="iqr")
    """

    # Expose old class-level thresholds
    HIGH_CARDINALITY_THRESHOLD = 50
    MEDIUM_CARDINALITY_THRESHOLD = 25
    TOP_N_CATEGORIES = 15
    TARGET_CARDINALITY_THRESHOLD = 10

    def __init__(
        self,
        df: pd.DataFrame,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        datetime_cols: Optional[List[str]] = None,
        target_col: Optional[str] = None,
        save_outputs: bool = False,
        output_dir: str = "./eda_outputs",
    ) -> None:
        warnings.warn(
            "EDAPipeline is deprecated. Use edapipeline.Pipeline instead. "
            "EDAPipeline will be removed in v2.0.",
            DeprecationWarning,
            stacklevel=2,
        )

        config = PipelineConfig()
        config.target_column = target_col
        config.numerical_columns = numerical_cols
        config.categorical_columns = categorical_cols
        config.datetime_columns = datetime_cols
        config.output.save_outputs = save_outputs
        config.output.output_dir = output_dir
        config.thresholds.high_cardinality = self.HIGH_CARDINALITY_THRESHOLD
        config.thresholds.medium_cardinality = self.MEDIUM_CARDINALITY_THRESHOLD
        config.thresholds.top_n_categories = self.TOP_N_CATEGORIES
        config.thresholds.target_cardinality = self.TARGET_CARDINALITY_THRESHOLD

        self._pipeline = Pipeline(df, config=config)

        # Expose column lists for old tests
        self.numerical_cols = (
            numerical_cols or self._pipeline.profile.numerical_columns
        )
        self.categorical_cols = (
            categorical_cols or self._pipeline.profile.categorical_columns
        )
        self.datetime_cols = (
            datetime_cols or self._pipeline.profile.datetime_columns
        )

    def data_overview(self) -> None:
        self._pipeline.data_overview()

    def missing_value_analysis(self, figsize=(12, 6)) -> None:
        self._pipeline.missing_values()

    def analyze_numerical_features(self, figsize=(15, 5)) -> None:
        self._pipeline.analyze_numerical()

    def analyze_categorical_features(self, figsize=(15, 5)) -> None:
        self._pipeline.analyze_categorical()

    def analyze_datetime_features(self, figsize=(15, 10)) -> None:
        self._pipeline.analyze_datetime()

    def correlation_analysis(self, figsize=(12, 8)) -> None:
        self._pipeline.correlation_analysis()

    def categorical_bivariate_analysis(self, figsize=(10, 6)) -> None:
        self._pipeline.bivariate_analysis()

    def numerical_bivariate_analysis(self, figsize=(8, 8)) -> None:
        self._pipeline.bivariate_analysis()

    def detect_outliers(self, method: str = "iqr", threshold: float = 3.0) -> None:
        self._pipeline._config.thresholds.outlier_zscore_threshold = threshold
        self._pipeline.detect_outliers(method=method)

    def run_complete_analysis(self, outlier_method: str = "iqr") -> None:
        self._pipeline._config.outlier_method = OutlierMethod(outlier_method)
        self._pipeline.run()
