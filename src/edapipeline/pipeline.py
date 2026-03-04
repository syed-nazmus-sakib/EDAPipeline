"""Pipeline orchestrator — the main entry point for EDAPipeline."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Type

import pandas as pd

from .analyzers import DEFAULT_ANALYZERS
from .analyzers.base import BaseAnalyzer
from .config import PipelineConfig
from .results import AnalysisResult, PipelineReport
from .types import DatasetProfile
from .utils.dtype_detection import detect_column_types
from .utils.logging import get_logger
from .utils.validators import validate_dataframe, validate_target_column
from .visualization import MatplotlibVisualizer

logger = logging.getLogger("edapipeline.pipeline")


class Pipeline:
    """Modular EDA pipeline orchestrator.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to analyze.
    config : PipelineConfig, optional
        Configuration object. If not provided, defaults are used.
    target_col : str, optional
        Target/label column name. Shortcut for config.target_column.
    analyzers : list of BaseAnalyzer subclasses, optional
        Custom list of analyzers to run. Defaults to all built-in analyzers.
    save_outputs : bool
        Whether to save plots and reports to disk.
    output_dir : str
        Directory for saved outputs.
    agents : bool
        Enable the agentic LLM layer (Phase 3).
    api_key : str, optional
        API key for the LLM provider (required when agents=True).
    model : str
        Exact model name (e.g., 'qwen/qwen3-4b').
    provider : str
        LLM provider ('openrouter', 'openai', 'anthropic', 'gemini', 'ollama').
    show_plots : bool
        Whether to display plots inline.

    Examples
    --------
    >>> pipeline = Pipeline(df, target_col="price")
    >>> report = pipeline.run()
    >>> print(report.summary())

    >>> pipeline = Pipeline(df, save_outputs=True, output_dir="./results")
    >>> report = pipeline.run()
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: Optional[PipelineConfig] = None,
        target_col: Optional[str] = None,
        analyzers: Optional[List[Type[BaseAnalyzer]]] = None,
        save_outputs: bool = False,
        output_dir: str = "./eda_outputs",
        agents: bool = False,
        api_key: Optional[str] = None,
        model: str = "qwen/qwen3-4b",
        provider: str = "openrouter",
        show_plots: bool = True,
    ) -> None:
        # Validate input
        validate_dataframe(df)
        self._df = df.copy()

        # Build config
        if config is None:
            config = PipelineConfig()

        # Apply convenience params
        if target_col is not None:
            config.target_column = target_col
        config.output.save_outputs = save_outputs
        config.output.output_dir = output_dir
        config.output.show_plots = show_plots
        config.agent.enabled = agents
        if api_key is not None:
            config.agent.api_key = api_key
        config.agent.model = model
        config.agent.provider = provider

        config.validate()
        self._config = config

        # Validate target
        validate_target_column(self._df, config.target_column)

        # Setup logging
        get_logger()

        # Detect column types
        self._profile: DatasetProfile = detect_column_types(
            self._df, config.target_column
        )

        # Initialize analyzers
        analyzer_classes = analyzers or DEFAULT_ANALYZERS
        self._analyzers: List[BaseAnalyzer] = [cls() for cls in analyzer_classes]

        # Initialize visualizer
        self._visualizer = MatplotlibVisualizer()

        # Report
        self._report = PipelineReport()

        logger.info(
            "Pipeline initialized: %d rows × %d cols, %d analyzers.",
            self._profile.n_rows, self._profile.n_cols, len(self._analyzers),
        )

    @property
    def config(self) -> PipelineConfig:
        return self._config

    @property
    def profile(self) -> DatasetProfile:
        return self._profile

    @property
    def report(self) -> PipelineReport:
        return self._report

    def add_analyzer(self, analyzer: BaseAnalyzer) -> "Pipeline":
        """Register an additional analyzer.

        Parameters
        ----------
        analyzer : BaseAnalyzer
            An instantiated analyzer to add to the pipeline.

        Returns
        -------
        Pipeline
            Self, for chaining.
        """
        self._analyzers.append(analyzer)
        return self

    def run(
        self,
        analyses: Optional[List[str]] = None,
    ) -> PipelineReport:
        """Execute the pipeline.

        Parameters
        ----------
        analyses : list of str, optional
            Run only these analyzers (by name). If None, runs all.

        Returns
        -------
        PipelineReport
            Aggregated results from all analyzers.
        """
        start = time.perf_counter()
        logger.info("Starting EDA pipeline...")

        for analyzer in self._analyzers:
            if analyses and analyzer.name not in analyses:
                continue

            try:
                logger.info("Running: %s", analyzer.name)
                result = analyzer.analyze(self._df, self._profile, self._config)
                self._report.add_result(result)

                # Render visualization
                self._render(analyzer.name, result)

            except Exception as exc:
                logger.error(
                    "Analyzer '%s' failed: %s", analyzer.name, exc, exc_info=True
                )

        elapsed = time.perf_counter() - start
        self._report.execution_time_seconds = round(elapsed, 2)
        self._report.total_plots = self._visualizer.plot_count

        logger.info(
            "Pipeline complete in %.2fs. %d plots generated.",
            elapsed, self._report.total_plots,
        )

        return self._report

    def _render(self, analyzer_name: str, result: AnalysisResult) -> None:
        """Dispatch visualization rendering based on analyzer name."""
        render_map = {
            "overview": self._visualizer.render_overview,
            "missing_values": self._visualizer.render_missing,
            "numerical": self._visualizer.render_numerical,
            "categorical": self._visualizer.render_categorical,
            "datetime": self._visualizer.render_datetime,
            "correlation": self._visualizer.render_correlation,
            "bivariate": self._visualizer.render_bivariate,
            "outlier": self._visualizer.render_outlier,
        }

        renderer = render_map.get(analyzer_name)
        if renderer:
            try:
                renderer(self._df, result, self._config)
            except Exception as exc:
                logger.warning(
                    "Visualization failed for '%s': %s", analyzer_name, exc
                )

    # ------------------------------------------------------------------
    # Convenience methods (run individual analyses)
    # ------------------------------------------------------------------

    def data_overview(self) -> AnalysisResult:
        """Run only the dataset overview analysis."""
        return self.run(analyses=["overview"]).get_result("overview")

    def missing_values(self) -> AnalysisResult:
        """Run only the missing value analysis."""
        return self.run(analyses=["missing_values"]).get_result("missing_values")

    def analyze_numerical(self) -> AnalysisResult:
        """Run only the numerical feature analysis."""
        return self.run(analyses=["numerical"]).get_result("numerical")

    def analyze_categorical(self) -> AnalysisResult:
        """Run only the categorical feature analysis."""
        return self.run(analyses=["categorical"]).get_result("categorical")

    def analyze_datetime(self) -> AnalysisResult:
        """Run only the datetime feature analysis."""
        return self.run(analyses=["datetime"]).get_result("datetime")

    def correlation_analysis(self) -> AnalysisResult:
        """Run only the correlation analysis."""
        return self.run(analyses=["correlation"]).get_result("correlation")

    def bivariate_analysis(self) -> AnalysisResult:
        """Run only the bivariate analysis."""
        return self.run(analyses=["bivariate"]).get_result("bivariate")

    def detect_outliers(self, method: str = "iqr") -> AnalysisResult:
        """Run only the outlier detection analysis."""
        from .types import OutlierMethod
        self._config.outlier_method = OutlierMethod(method)
        return self.run(analyses=["outlier"]).get_result("outlier")
