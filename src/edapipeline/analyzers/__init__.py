"""Pluggable analyzer modules for EDAPipeline."""

from .base import BaseAnalyzer
from .overview import OverviewAnalyzer
from .missing import MissingValueAnalyzer
from .numerical import NumericalAnalyzer
from .categorical import CategoricalAnalyzer
from .datetime_analyzer import DatetimeAnalyzer
from .correlation import CorrelationAnalyzer
from .bivariate import BivariateAnalyzer
from .outlier import OutlierAnalyzer

__all__ = [
    "BaseAnalyzer",
    "OverviewAnalyzer",
    "MissingValueAnalyzer",
    "NumericalAnalyzer",
    "CategoricalAnalyzer",
    "DatetimeAnalyzer",
    "CorrelationAnalyzer",
    "BivariateAnalyzer",
    "OutlierAnalyzer",
]

# Default analyzer order
DEFAULT_ANALYZERS = [
    OverviewAnalyzer,
    MissingValueAnalyzer,
    NumericalAnalyzer,
    CategoricalAnalyzer,
    DatetimeAnalyzer,
    CorrelationAnalyzer,
    BivariateAnalyzer,
    OutlierAnalyzer,
]
