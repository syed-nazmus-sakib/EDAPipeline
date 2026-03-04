"""Categorical feature analyzer."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ..config import PipelineConfig
from ..results import CategoricalAnalysisResult, CategoricalFeatureResult
from ..types import DatasetProfile
from .base import BaseAnalyzer


class CategoricalAnalyzer(BaseAnalyzer):
    """Analyze categorical features: cardinality, distribution, entropy."""

    @property
    def name(self) -> str:
        return "categorical"

    def analyze(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile,
        config: PipelineConfig,
    ) -> CategoricalAnalysisResult:
        columns = config.categorical_columns or profile.categorical_columns

        if not columns:
            self.logger.info("No categorical features identified.")
            return CategoricalAnalysisResult(analyzer_name=self.name)

        self.logger.info("Analyzing %d categorical features.", len(columns))
        features: List[CategoricalFeatureResult] = []

        for col in columns:
            result = self._analyze_column(df, col, config)
            features.append(result)

        return CategoricalAnalysisResult(
            analyzer_name=self.name,
            features=features,
        )

    def _analyze_column(
        self,
        df: pd.DataFrame,
        col: str,
        config: PipelineConfig,
    ) -> CategoricalFeatureResult:
        """Analyze a single categorical column."""
        series = df[col]
        value_counts = series.value_counts()
        value_pcts = series.value_counts(normalize=True) * 100
        n_unique = int(series.nunique())

        # Cardinality level
        thresholds = config.thresholds
        if n_unique > thresholds.high_cardinality:
            cardinality_level = "high"
        elif n_unique > thresholds.medium_cardinality:
            cardinality_level = "medium"
        else:
            cardinality_level = "low"

        # Mode
        mode = str(value_counts.index[0]) if len(value_counts) > 0 else None
        mode_freq = int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
        mode_pct = float(value_pcts.iloc[0]) if len(value_pcts) > 0 else 0.0

        # Top categories (up to 10)
        top_n = min(10, len(value_counts))
        top_categories = {
            str(k): int(v) for k, v in value_counts.head(top_n).items()
        }
        top_percentages = {
            str(k): round(float(v), 2) for k, v in value_pcts.head(top_n).items()
        }

        # Entropy
        entropy = None
        normalized_entropy = None
        try:
            ent = float(sp_stats.entropy(value_counts.values))
            max_ent = float(np.log(n_unique)) if n_unique > 0 else 0.0
            entropy = round(ent, 4)
            normalized_entropy = round(ent / max_ent, 4) if max_ent > 0 else 0.0
        except Exception:
            pass

        return CategoricalFeatureResult(
            column=col,
            unique_count=n_unique,
            mode=mode,
            mode_frequency=mode_freq,
            mode_percentage=round(mode_pct, 2),
            entropy=entropy,
            normalized_entropy=normalized_entropy,
            top_categories=top_categories,
            top_percentages=top_percentages,
            cardinality_level=cardinality_level,
        )
