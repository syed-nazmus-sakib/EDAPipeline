"""Correlation analyzer."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import PipelineConfig
from ..results import CorrelationPair, CorrelationResult
from ..types import DatasetProfile
from .base import BaseAnalyzer


class CorrelationAnalyzer(BaseAnalyzer):
    """Compute and interpret correlations between numerical features."""

    @property
    def name(self) -> str:
        return "correlation"

    def analyze(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile,
        config: PipelineConfig,
    ) -> CorrelationResult:
        num_cols = list(config.numerical_columns or profile.numerical_columns)

        # Include target column if it's numerical
        target = config.target_column
        if target and target in df.select_dtypes(include=np.number).columns:
            if target not in num_cols:
                num_cols.append(target)

        if len(num_cols) < 2:
            self.logger.info("Need ≥2 numerical features for correlation.")
            return CorrelationResult(analyzer_name=self.name)

        self.logger.info("Computing correlations for %d features.", len(num_cols))
        corr_matrix = df[num_cols].corr()

        # Build pair list (upper triangle only)
        pairs = self._extract_pairs(corr_matrix, config)
        pairs_sorted = sorted(pairs, key=lambda p: p.correlation, reverse=True)

        top_positive = [p for p in pairs_sorted if p.correlation > 0][:10]
        top_negative = [p for p in pairs_sorted if p.correlation < 0][-10:]
        top_negative.reverse()

        # Target correlations
        target_corrs: Optional[Dict[str, float]] = None
        strongest_pos: Optional[str] = None
        strongest_neg: Optional[str] = None

        if target and target in corr_matrix.columns:
            tc = corr_matrix[target].drop(target).sort_values(ascending=False)
            target_corrs = {k: round(float(v), 4) for k, v in tc.items()}
            if len(tc) > 0:
                strongest_pos = str(tc.idxmax())
                strongest_neg = str(tc.idxmin())

        # Serializable matrix
        matrix_dict = {
            str(r): {str(c): round(float(corr_matrix.loc[r, c]), 4) for c in corr_matrix.columns}
            for r in corr_matrix.index
        }

        return CorrelationResult(
            analyzer_name=self.name,
            matrix=matrix_dict,
            top_positive=top_positive,
            top_negative=top_negative,
            target_correlations=target_corrs,
            strongest_positive_feature=strongest_pos,
            strongest_negative_feature=strongest_neg,
        )

    def _extract_pairs(
        self,
        corr_matrix: pd.DataFrame,
        config: PipelineConfig,
    ) -> List[CorrelationPair]:
        """Extract unique correlation pairs from the upper triangle."""
        thresholds = config.thresholds
        pairs: List[CorrelationPair] = []
        cols = corr_matrix.columns

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r = float(corr_matrix.iloc[i, j])
                abs_r = abs(r)

                if abs_r >= thresholds.correlation_strong:
                    strength = "strong"
                elif abs_r >= thresholds.correlation_moderate:
                    strength = "moderate"
                elif abs_r >= thresholds.correlation_weak:
                    strength = "weak"
                else:
                    strength = "very_weak"

                direction = "positive" if r > 0 else "negative"

                pairs.append(
                    CorrelationPair(
                        feature_1=str(cols[i]),
                        feature_2=str(cols[j]),
                        correlation=round(r, 4),
                        method="pearson",
                        strength=strength,
                        direction=direction,
                    )
                )

        return pairs
