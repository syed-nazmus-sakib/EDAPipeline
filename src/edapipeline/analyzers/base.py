"""Base class for all analyzers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from ..config import PipelineConfig
from ..results import AnalysisResult
from ..types import DatasetProfile


class BaseAnalyzer(ABC):
    """Abstract base class that all analyzers extend."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"edapipeline.{self.name}")

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this analyzer."""
        ...

    @abstractmethod
    def analyze(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile,
        config: PipelineConfig,
    ) -> AnalysisResult:
        """Run analysis and return a structured result.

        Parameters
        ----------
        df : pd.DataFrame
            The dataset to analyze.
        profile : DatasetProfile
            Pre-computed column type information.
        config : PipelineConfig
            Pipeline configuration.

        Returns
        -------
        AnalysisResult
            Structured result object.
        """
        ...
