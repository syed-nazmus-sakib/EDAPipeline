"""EDAPipeline — Modular, Agentic Exploratory Data Analysis Toolkit."""

from .__version__ import __version__
from .pipeline import Pipeline
from .config import PipelineConfig
from .results import PipelineReport
from .types import ColumnType, OutlierMethod, AnalysisLevel

# Backward compatibility
from .legacy import EDAPipeline

__all__ = [
    "Pipeline",
    "PipelineConfig",
    "PipelineReport",
    "ColumnType",
    "OutlierMethod",
    "AnalysisLevel",
    "EDAPipeline",  # deprecated
    "__version__",
]
