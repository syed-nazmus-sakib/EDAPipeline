"""Numerical feature analyzer."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import normaltest

from ..config import PipelineConfig
from ..results import NumericalAnalysisResult, NumericalFeatureResult
from ..types import DatasetProfile
from .base import BaseAnalyzer


class NumericalAnalyzer(BaseAnalyzer):
    """Analyze numerical features: descriptive stats, distribution, normality."""

    @property
    def name(self) -> str:
        return "numerical"

    def analyze(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile,
        config: PipelineConfig,
    ) -> NumericalAnalysisResult:
        columns = config.numerical_columns or profile.numerical_columns

        if not columns:
            self.logger.info("No numerical features identified.")
            return NumericalAnalysisResult(analyzer_name=self.name)

        self.logger.info("Analyzing %d numerical features.", len(columns))
        features: List[NumericalFeatureResult] = []

        for col in columns:
            result = self._analyze_column(df, col, config)
            if result is not None:
                features.append(result)

        return NumericalAnalysisResult(
            analyzer_name=self.name,
            features=features,
        )

    def _analyze_column(
        self,
        df: pd.DataFrame,
        col: str,
        config: PipelineConfig,
    ) -> NumericalFeatureResult | None:
        """Analyze a single numerical column."""
        series = df[col]

        if series.isnull().all():
            self.logger.warning("Skipping '%s' — all values are NaN.", col)
            return None

        if series.empty:
            self.logger.warning("Skipping '%s' — column is empty.", col)
            return None

        clean = series.dropna()
        desc = clean.describe()

        result = NumericalFeatureResult(
            column=col,
            count=int(desc["count"]),
            mean=float(desc["mean"]),
            std=float(desc["std"]),
            min=float(desc["min"]),
            q25=float(desc["25%"]),
            median=float(desc["50%"]),
            q75=float(desc["75%"]),
            max=float(desc["max"]),
        )

        # Distribution metrics
        try:
            result.mad = float(stats.median_abs_deviation(clean))
            result.skewness = float(series.skew())
            result.kurtosis = float(series.kurt())

            # Interpret skewness
            skew = result.skewness
            if abs(skew) < 0.5:
                result.skewness_interpretation = "approximately symmetric"
            elif skew > 0.5:
                result.skewness_interpretation = "right-skewed (positive skew)"
            else:
                result.skewness_interpretation = "left-skewed (negative skew)"

            # Interpret kurtosis
            kurt = result.kurtosis
            if abs(kurt) < 0.5:
                result.kurtosis_interpretation = "mesokurtic (normal-like tails)"
            elif kurt > 0.5:
                result.kurtosis_interpretation = "leptokurtic (heavy tails, more outliers)"
            else:
                result.kurtosis_interpretation = "platykurtic (light tails, fewer outliers)"

        except Exception as exc:
            self.logger.warning("Distribution metrics failed for '%s': %s", col, exc)

        # Normality test (D'Agostino-Pearson)
        try:
            alpha = config.thresholds.normality_alpha
            stat, p_value = normaltest(clean)
            result.normality_test_stat = float(stat)
            result.normality_p_value = float(p_value)
            result.is_normal = p_value >= alpha

            if result.is_normal:
                result.normality_interpretation = (
                    f"Cannot reject normality (p={p_value:.4f} >= α={alpha})"
                )
            else:
                result.normality_interpretation = (
                    f"Reject normality (p={p_value:.4f} < α={alpha})"
                )
        except ValueError as exc:
            self.logger.warning("Normality test failed for '%s': %s", col, exc)

        return result
