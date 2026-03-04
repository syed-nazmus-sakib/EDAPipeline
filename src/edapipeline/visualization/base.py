"""Base visualizer protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import pandas as pd

from ..config import PipelineConfig
from ..results import AnalysisResult
from ..types import DatasetProfile


class BaseVisualizer(ABC):
    """Abstract base for visualization backends."""

    @abstractmethod
    def render_overview(self, df: pd.DataFrame, result: Any, config: PipelineConfig) -> None:
        ...

    @abstractmethod
    def render_missing(self, df: pd.DataFrame, result: Any, config: PipelineConfig) -> None:
        ...

    @abstractmethod
    def render_numerical(self, df: pd.DataFrame, result: Any, config: PipelineConfig) -> None:
        ...

    @abstractmethod
    def render_categorical(self, df: pd.DataFrame, result: Any, config: PipelineConfig) -> None:
        ...

    @abstractmethod
    def render_datetime(self, df: pd.DataFrame, result: Any, config: PipelineConfig) -> None:
        ...

    @abstractmethod
    def render_correlation(self, df: pd.DataFrame, result: Any, config: PipelineConfig) -> None:
        ...

    @abstractmethod
    def render_bivariate(self, df: pd.DataFrame, result: Any, config: PipelineConfig) -> None:
        ...

    @abstractmethod
    def render_outlier(self, df: pd.DataFrame, result: Any, config: PipelineConfig) -> None:
        ...
