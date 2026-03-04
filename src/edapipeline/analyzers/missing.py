"""Missing value analyzer."""

from __future__ import annotations

import pandas as pd

from ..config import PipelineConfig
from ..results import MissingValueInfo, MissingValueResult
from ..types import DatasetProfile
from .base import BaseAnalyzer


class MissingValueAnalyzer(BaseAnalyzer):
    """Analyze and report missing values across all columns."""

    @property
    def name(self) -> str:
        return "missing_values"

    def analyze(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile,
        config: PipelineConfig,
    ) -> MissingValueResult:
        missing_counts = df.isnull().sum()
        total_missing = int(missing_counts.sum())
        n_rows = len(df)

        details = []
        for col in df.columns:
            mc = int(missing_counts[col])
            if mc > 0:
                details.append(
                    MissingValueInfo(
                        column=col,
                        missing_count=mc,
                        missing_percentage=round(mc / n_rows * 100, 2),
                        total_rows=n_rows,
                    )
                )

        # Sort by percentage descending
        details.sort(key=lambda x: x.missing_percentage, reverse=True)

        cols_with_missing = len(details)

        if total_missing == 0:
            self.logger.info("No missing values found.")
        else:
            self.logger.info(
                "%d missing cells across %d columns.",
                total_missing, cols_with_missing,
            )

        return MissingValueResult(
            analyzer_name=self.name,
            total_missing_cells=total_missing,
            columns_with_missing=cols_with_missing,
            details=details,
        )
