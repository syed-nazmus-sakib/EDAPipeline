"""Bivariate analysis: numerical×categorical and numerical×numerical."""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional

import pandas as pd
from scipy import stats

from ..config import PipelineConfig
from ..results import (
    BivariateAnalysisResult,
    BivariateNumCatResult,
    BivariateNumNumResult,
)
from ..types import DatasetProfile
from .base import BaseAnalyzer


class BivariateAnalyzer(BaseAnalyzer):
    """Analyze pairwise relationships between features."""

    @property
    def name(self) -> str:
        return "bivariate"

    def analyze(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile,
        config: PipelineConfig,
    ) -> BivariateAnalysisResult:
        num_cols = config.numerical_columns or profile.numerical_columns
        cat_cols = config.categorical_columns or profile.categorical_columns
        thresholds = config.thresholds

        num_cat_results = self._analyze_num_cat(df, num_cols, cat_cols, thresholds)
        num_num_results = self._analyze_num_num(df, num_cols, config)

        self.logger.info(
            "%d num×cat, %d num×num pairs analyzed.",
            len(num_cat_results), len(num_num_results),
        )

        return BivariateAnalysisResult(
            analyzer_name=self.name,
            num_cat_results=num_cat_results,
            num_num_results=num_num_results,
        )

    # ------------------------------------------------------------------
    # Numerical × Categorical
    # ------------------------------------------------------------------

    def _analyze_num_cat(
        self,
        df: pd.DataFrame,
        num_cols: List[str],
        cat_cols: List[str],
        thresholds,
    ) -> List[BivariateNumCatResult]:
        results: List[BivariateNumCatResult] = []

        if not num_cols or not cat_cols:
            return results

        for num_col in num_cols:
            for cat_col in cat_cols:
                n_unique = df[cat_col].nunique()
                if n_unique > thresholds.medium_cardinality:
                    continue

                grouped = df.groupby(cat_col)[num_col].agg(
                    ["mean", "median", "std", "count"]
                )
                group_stats: Dict[str, Dict[str, float]] = {}
                for idx, row in grouped.iterrows():
                    group_stats[str(idx)] = {
                        "mean": round(float(row["mean"]), 4) if pd.notna(row["mean"]) else 0.0,
                        "median": round(float(row["median"]), 4) if pd.notna(row["median"]) else 0.0,
                        "std": round(float(row["std"]), 4) if pd.notna(row["std"]) else 0.0,
                        "count": int(row["count"]),
                    }

                results.append(
                    BivariateNumCatResult(
                        numerical_col=num_col,
                        categorical_col=cat_col,
                        group_stats=group_stats,
                    )
                )

        return results

    # ------------------------------------------------------------------
    # Numerical × Numerical
    # ------------------------------------------------------------------

    def _analyze_num_num(
        self,
        df: pd.DataFrame,
        num_cols: List[str],
        config: PipelineConfig,
    ) -> List[BivariateNumNumResult]:
        results: List[BivariateNumNumResult] = []

        if len(num_cols) < 2:
            return results

        thresholds = config.thresholds
        seen = set()

        for col1, col2 in itertools.combinations(num_cols, 2):
            pair = tuple(sorted((col1, col2)))
            if pair in seen:
                continue
            seen.add(pair)

            try:
                clean_1 = df[col1].dropna()
                clean_2 = df[col2].dropna()
                # Align on common indices
                common = clean_1.index.intersection(clean_2.index)
                if len(common) < 3:
                    continue

                corr, p_value = stats.pearsonr(
                    df.loc[common, col1], df.loc[common, col2]
                )
                abs_r = abs(corr)

                if abs_r >= thresholds.correlation_strong:
                    strength = "strong"
                elif abs_r >= thresholds.correlation_moderate:
                    strength = "moderate"
                elif abs_r >= thresholds.correlation_weak:
                    strength = "weak"
                else:
                    strength = "very_weak"

                results.append(
                    BivariateNumNumResult(
                        col_1=col1,
                        col_2=col2,
                        correlation=round(float(corr), 4),
                        p_value=round(float(p_value), 4),
                        strength=strength,
                        direction="positive" if corr > 0 else "negative",
                        is_significant=p_value < config.thresholds.normality_alpha,
                    )
                )
            except (ValueError, TypeError):
                self.logger.warning(
                    "Could not compute correlation for '%s' vs '%s'.", col1, col2
                )

        return results
