"""Outlier detection analyzer."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from scipy import stats

from ..config import PipelineConfig
from ..results import OutlierAnalysisResult, OutlierFeatureResult
from ..types import DatasetProfile, OutlierMethod
from .base import BaseAnalyzer


class OutlierAnalyzer(BaseAnalyzer):
    """Detect outliers using IQR or Z-score methods."""

    @property
    def name(self) -> str:
        return "outlier"

    def analyze(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile,
        config: PipelineConfig,
    ) -> OutlierAnalysisResult:
        num_cols = config.numerical_columns or profile.numerical_columns
        method = config.outlier_method

        if not num_cols:
            self.logger.info("No numerical features for outlier detection.")
            return OutlierAnalysisResult(
                analyzer_name=self.name, method=method.value
            )

        self.logger.info(
            "Detecting outliers in %d features (%s).", len(num_cols), method.value
        )

        features: List[OutlierFeatureResult] = []
        n_total = len(df)

        for col in num_cols:
            col_data = df[col].dropna()
            if col_data.empty:
                continue

            if method == OutlierMethod.IQR:
                result = self._iqr_method(col, col_data, n_total)
            elif method == OutlierMethod.ZSCORE:
                threshold = config.thresholds.outlier_zscore_threshold
                result = self._zscore_method(col, col_data, n_total, threshold)
            else:
                self.logger.warning("Unsupported outlier method: %s", method)
                continue

            features.append(result)

        # Summary stats
        features_with_outliers = [f for f in features if f.n_outliers > 0]
        worst = max(features_with_outliers, key=lambda f: f.percentage) if features_with_outliers else None

        return OutlierAnalysisResult(
            analyzer_name=self.name,
            method=method.value,
            features=features,
            total_features_with_outliers=len(features_with_outliers),
            worst_feature=worst.column if worst else None,
            worst_percentage=worst.percentage if worst else 0.0,
        )

    def _iqr_method(
        self, col: str, data: pd.Series, n_total: int,
    ) -> OutlierFeatureResult:
        q1 = float(data.quantile(0.25))
        q3 = float(data.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_outliers = int(((data < lower) | (data > upper)).sum())

        return OutlierFeatureResult(
            column=col,
            method="iqr",
            n_outliers=n_outliers,
            percentage=round(n_outliers / n_total * 100, 2),
            lower_bound=round(lower, 4),
            upper_bound=round(upper, 4),
        )

    def _zscore_method(
        self, col: str, data: pd.Series, n_total: int, threshold: float,
    ) -> OutlierFeatureResult:
        if data.std() == 0:
            return OutlierFeatureResult(
                column=col, method="zscore", n_outliers=0, percentage=0.0,
                threshold=threshold,
            )

        z_scores = np.abs(stats.zscore(data))
        n_outliers = int((z_scores > threshold).sum())

        return OutlierFeatureResult(
            column=col,
            method="zscore",
            n_outliers=n_outliers,
            percentage=round(n_outliers / n_total * 100, 2),
            threshold=threshold,
        )
