"""Structured result objects returned by analyzers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    """Base class for all analysis results."""

    analyzer_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def summary(self) -> str:
        """Return a one-line summary for console output."""
        return f"[{self.analyzer_name}] Analysis complete."


# ---------------------------------------------------------------------------
# Overview & Missing
# ---------------------------------------------------------------------------

@dataclass
class OverviewResult(AnalysisResult):
    """Result from the dataset overview analyzer."""

    n_rows: int = 0
    n_cols: int = 0
    memory_mb: float = 0.0
    dtypes: Dict[str, str] = field(default_factory=dict)
    numerical_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    datetime_columns: List[str] = field(default_factory=list)
    sample_data: Optional[Dict[str, List[Any]]] = None

    def summary(self) -> str:
        return (
            f"[Overview] {self.n_rows:,} rows × {self.n_cols} cols | "
            f"{len(self.numerical_columns)} numerical, "
            f"{len(self.categorical_columns)} categorical, "
            f"{len(self.datetime_columns)} datetime | "
            f"{self.memory_mb:.2f} MB"
        )


@dataclass
class MissingValueInfo:
    """Missing value details for a single column."""

    column: str
    missing_count: int
    missing_percentage: float
    total_rows: int


@dataclass
class MissingValueResult(AnalysisResult):
    """Result from missing value analysis."""

    total_missing_cells: int = 0
    columns_with_missing: int = 0
    details: List[MissingValueInfo] = field(default_factory=list)

    def summary(self) -> str:
        if self.total_missing_cells == 0:
            return "[Missing Values] No missing values found."
        return (
            f"[Missing Values] {self.total_missing_cells:,} missing cells "
            f"across {self.columns_with_missing} columns."
        )


# ---------------------------------------------------------------------------
# Numerical
# ---------------------------------------------------------------------------

@dataclass
class NumericalFeatureResult:
    """Statistics for a single numerical column."""

    column: str
    count: int = 0
    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    q25: float = 0.0
    median: float = 0.0
    q75: float = 0.0
    max: float = 0.0
    mad: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    skewness_interpretation: str = ""
    kurtosis_interpretation: str = ""
    normality_test_stat: Optional[float] = None
    normality_p_value: Optional[float] = None
    is_normal: Optional[bool] = None
    normality_interpretation: str = ""


@dataclass
class NumericalAnalysisResult(AnalysisResult):
    """Aggregated result for all numerical features."""

    features: List[NumericalFeatureResult] = field(default_factory=list)

    def summary(self) -> str:
        return f"[Numerical] Analyzed {len(self.features)} numerical features."


# ---------------------------------------------------------------------------
# Categorical
# ---------------------------------------------------------------------------

@dataclass
class CategoricalFeatureResult:
    """Statistics for a single categorical column."""

    column: str
    unique_count: int = 0
    mode: Optional[str] = None
    mode_frequency: int = 0
    mode_percentage: float = 0.0
    entropy: Optional[float] = None
    normalized_entropy: Optional[float] = None
    top_categories: Dict[str, int] = field(default_factory=dict)
    top_percentages: Dict[str, float] = field(default_factory=dict)
    cardinality_level: str = "low"  # "low" | "medium" | "high"


@dataclass
class CategoricalAnalysisResult(AnalysisResult):
    """Aggregated result for all categorical features."""

    features: List[CategoricalFeatureResult] = field(default_factory=list)

    def summary(self) -> str:
        return f"[Categorical] Analyzed {len(self.features)} categorical features."


# ---------------------------------------------------------------------------
# DateTime
# ---------------------------------------------------------------------------

@dataclass
class DatetimeFeatureResult:
    """Statistics for a single datetime column."""

    column: str
    min_date: Optional[str] = None
    max_date: Optional[str] = None
    range_days: int = 0
    range_years: float = 0.0
    unique_years: int = 0
    unique_months: int = 0
    unique_days_of_week: int = 0


@dataclass
class DatetimeAnalysisResult(AnalysisResult):
    """Aggregated result for all datetime features."""

    features: List[DatetimeFeatureResult] = field(default_factory=list)

    def summary(self) -> str:
        return f"[DateTime] Analyzed {len(self.features)} datetime features."


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

@dataclass
class CorrelationPair:
    """A single correlation pair."""

    feature_1: str
    feature_2: str
    correlation: float
    p_value: Optional[float] = None
    method: str = "pearson"
    strength: str = ""      # "strong" | "moderate" | "weak" | "very_weak"
    direction: str = ""     # "positive" | "negative"


@dataclass
class CorrelationResult(AnalysisResult):
    """Result from correlation analysis."""

    matrix: Optional[Dict[str, Dict[str, float]]] = None
    top_positive: List[CorrelationPair] = field(default_factory=list)
    top_negative: List[CorrelationPair] = field(default_factory=list)
    target_correlations: Optional[Dict[str, float]] = None
    strongest_positive_feature: Optional[str] = None
    strongest_negative_feature: Optional[str] = None

    def summary(self) -> str:
        n_pairs = len(self.top_positive) + len(self.top_negative)
        return f"[Correlation] Computed {n_pairs} feature pair correlations."


# ---------------------------------------------------------------------------
# Bivariate
# ---------------------------------------------------------------------------

@dataclass
class BivariateNumCatResult:
    """Result for a single numerical-categorical pair."""

    numerical_col: str
    categorical_col: str
    group_stats: Optional[Dict[str, Dict[str, float]]] = None


@dataclass
class BivariateNumNumResult:
    """Result for a single numerical-numerical pair."""

    col_1: str
    col_2: str
    correlation: Optional[float] = None
    p_value: Optional[float] = None
    strength: str = ""
    direction: str = ""
    is_significant: Optional[bool] = None


@dataclass
class BivariateAnalysisResult(AnalysisResult):
    """Aggregated bivariate analysis result."""

    num_cat_results: List[BivariateNumCatResult] = field(default_factory=list)
    num_num_results: List[BivariateNumNumResult] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"[Bivariate] {len(self.num_cat_results)} num×cat, "
            f"{len(self.num_num_results)} num×num pairs."
        )


# ---------------------------------------------------------------------------
# Outlier
# ---------------------------------------------------------------------------

@dataclass
class OutlierFeatureResult:
    """Outlier statistics for a single feature."""

    column: str
    method: str
    n_outliers: int = 0
    percentage: float = 0.0
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    threshold: Optional[float] = None


@dataclass
class OutlierAnalysisResult(AnalysisResult):
    """Aggregated outlier detection result."""

    method: str = "iqr"
    features: List[OutlierFeatureResult] = field(default_factory=list)
    total_features_with_outliers: int = 0
    worst_feature: Optional[str] = None
    worst_percentage: float = 0.0

    def summary(self) -> str:
        if self.total_features_with_outliers == 0:
            return f"[Outliers] No outliers detected ({self.method.upper()})."
        return (
            f"[Outliers] {self.total_features_with_outliers} features with outliers "
            f"({self.method.upper()}). Worst: {self.worst_feature} "
            f"({self.worst_percentage:.1f}%)."
        )


# ---------------------------------------------------------------------------
# Pipeline Report (aggregates everything)
# ---------------------------------------------------------------------------

@dataclass
class PipelineReport:
    """Aggregates all analysis results into a single report."""

    results: Dict[str, AnalysisResult] = field(default_factory=dict)
    execution_time_seconds: float = 0.0
    total_plots: int = 0

    # Agentic layer outputs (Phase 3)
    insights: Optional[List[str]] = None
    investigations: Optional[List[Dict[str, Any]]] = None
    recommendations: Optional[List[str]] = None

    def add_result(self, result: AnalysisResult) -> None:
        """Add an analysis result to the report."""
        self.results[result.analyzer_name] = result

    def get_result(self, name: str) -> Optional[AnalysisResult]:
        """Retrieve a specific analysis result by analyzer name."""
        return self.results.get(name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_time_seconds": self.execution_time_seconds,
            "total_plots": self.total_plots,
            "results": {k: v.to_dict() for k, v in self.results.items()},
            "insights": self.insights,
            "investigations": self.investigations,
            "recommendations": self.recommendations,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def summary(self) -> str:
        """Print a summary of all results."""
        lines = [f"EDAPipeline Report ({self.execution_time_seconds:.1f}s)"]
        for result in self.results.values():
            lines.append(f"  {result.summary()}")
        return "\n".join(lines)
