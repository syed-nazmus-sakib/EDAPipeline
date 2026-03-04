"""DateTime feature analyzer."""

from __future__ import annotations

from typing import List

import pandas as pd

from ..config import PipelineConfig
from ..results import DatetimeAnalysisResult, DatetimeFeatureResult
from ..types import DatasetProfile
from .base import BaseAnalyzer


class DatetimeAnalyzer(BaseAnalyzer):
    """Analyze datetime features: temporal range, component distributions."""

    @property
    def name(self) -> str:
        return "datetime"

    def analyze(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile,
        config: PipelineConfig,
    ) -> DatetimeAnalysisResult:
        columns = config.datetime_columns or profile.datetime_columns

        if not columns:
            self.logger.info("No datetime features identified.")
            return DatetimeAnalysisResult(analyzer_name=self.name)

        self.logger.info("Analyzing %d datetime features.", len(columns))
        features: List[DatetimeFeatureResult] = []

        for col in columns:
            result = self._analyze_column(df, col)
            if result is not None:
                features.append(result)

        return DatetimeAnalysisResult(
            analyzer_name=self.name,
            features=features,
        )

    def _analyze_column(
        self,
        df: pd.DataFrame,
        col: str,
    ) -> DatetimeFeatureResult | None:
        """Analyze a single datetime column."""
        series = df[col]

        if series.isnull().all():
            self.logger.warning("Skipping '%s' — all values are NaN.", col)
            return None

        # Ensure datetime dtype
        if not pd.api.types.is_datetime64_any_dtype(series):
            try:
                series = pd.to_datetime(series, errors="coerce")
            except Exception:
                self.logger.warning("Could not parse '%s' as datetime.", col)
                return None

        min_date = series.min()
        max_date = series.max()
        date_range = max_date - min_date

        # Extract components safely
        try:
            unique_years = int(series.dt.year.nunique())
            unique_months = int(series.dt.month.nunique())
            unique_dow = int(series.dt.dayofweek.nunique())
        except AttributeError:
            self.logger.warning("Could not extract components from '%s'.", col)
            return None

        return DatetimeFeatureResult(
            column=col,
            min_date=str(min_date),
            max_date=str(max_date),
            range_days=int(date_range.days),
            range_years=round(date_range.days / 365.25, 2),
            unique_years=unique_years,
            unique_months=unique_months,
            unique_days_of_week=unique_dow,
        )
