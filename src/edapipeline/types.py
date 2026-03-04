"""Type definitions, enums, and protocols for EDAPipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import pandas as pd


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ColumnType(Enum):
    """Classification of DataFrame column types."""
    NUMERICAL = auto()
    CATEGORICAL = auto()
    DATETIME = auto()
    BOOLEAN = auto()
    TEXT = auto()


class OutlierMethod(Enum):
    """Supported outlier detection methods."""
    IQR = "iqr"
    ZSCORE = "zscore"
    ISOLATION_FOREST = "isolation_forest"


class AnalysisLevel(Enum):
    """Depth of analysis to perform."""
    BASIC = "basic"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"


class CorrelationType(Enum):
    """Types of correlation measures."""
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"
    CRAMERS_V = "cramers_v"
    POINT_BISERIAL = "point_biserial"
    MUTUAL_INFO = "mutual_info"


class NormalityTest(Enum):
    """Supported normality test methods."""
    DAGOSTINO = "dagostino"
    SHAPIRO = "shapiro"
    ANDERSON = "anderson"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class ColumnProfile:
    """Metadata profile for a single DataFrame column."""

    name: str
    dtype: str
    column_type: ColumnType
    null_count: int = 0
    null_percentage: float = 0.0
    unique_count: int = 0
    memory_bytes: int = 0
    sample_values: List[Any] = field(default_factory=list)

    @property
    def has_nulls(self) -> bool:
        return self.null_count > 0


@dataclass
class DatasetProfile:
    """High-level profile of the entire dataset."""

    n_rows: int
    n_cols: int
    memory_mb: float
    column_profiles: Dict[str, ColumnProfile] = field(default_factory=dict)
    numerical_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    datetime_columns: List[str] = field(default_factory=list)
    boolean_columns: List[str] = field(default_factory=list)
    target_column: Optional[str] = None


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

@runtime_checkable
class Analyzer(Protocol):
    """Interface that all analyzers must implement."""

    @property
    def name(self) -> str:
        """Human-readable name of this analyzer."""
        ...

    def analyze(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile,
        config: Any,
    ) -> Any:
        """Run analysis and return a structured result."""
        ...


@runtime_checkable
class Visualizer(Protocol):
    """Interface for visualization backends."""

    def render(self, result: Any, config: Any) -> Any:
        """Render a visualization from an analysis result."""
        ...


@runtime_checkable
class Reporter(Protocol):
    """Interface for report generation backends."""

    def generate(self, results: Dict[str, Any], config: Any) -> Any:
        """Generate a report from aggregated analysis results."""
        ...
