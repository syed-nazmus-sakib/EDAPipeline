"""Dataset overview analyzer."""

from __future__ import annotations

import pandas as pd

from ..config import PipelineConfig
from ..results import OverviewResult
from ..types import DatasetProfile
from .base import BaseAnalyzer


class OverviewAnalyzer(BaseAnalyzer):
    """Generates a high-level overview of the dataset."""

    @property
    def name(self) -> str:
        return "overview"

    def analyze(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile,
        config: PipelineConfig,
    ) -> OverviewResult:
        self.logger.info(
            "Dataset: %d rows × %d cols (%.2f MB)",
            profile.n_rows, profile.n_cols, profile.memory_mb,
        )

        dtypes = {col: str(df[col].dtype) for col in df.columns}

        # Sample data (first 5 rows as dict of lists)
        sample = df.head(5)
        sample_data = {
            col: sample[col].astype(str).tolist() for col in sample.columns
        }

        return OverviewResult(
            analyzer_name=self.name,
            n_rows=profile.n_rows,
            n_cols=profile.n_cols,
            memory_mb=profile.memory_mb,
            dtypes=dtypes,
            numerical_columns=profile.numerical_columns,
            categorical_columns=profile.categorical_columns,
            datetime_columns=profile.datetime_columns,
            sample_data=sample_data,
        )
