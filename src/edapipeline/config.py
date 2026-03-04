"""Centralized configuration for EDAPipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .types import AnalysisLevel, CorrelationType, OutlierMethod


@dataclass
class ThresholdConfig:
    """Thresholds that control analysis behavior."""

    high_cardinality: int = 50
    medium_cardinality: int = 25
    top_n_categories: int = 15
    target_cardinality: int = 10
    pairplot_max_features: int = 6
    outlier_zscore_threshold: float = 3.0
    normality_alpha: float = 0.05
    correlation_strong: float = 0.7
    correlation_moderate: float = 0.4
    correlation_weak: float = 0.2


@dataclass
class OutputConfig:
    """Settings for output and report generation."""

    save_outputs: bool = False
    output_dir: str = "./eda_outputs"
    plot_dpi: int = 300
    plot_format: str = "png"
    report_format: str = "console"  # "console" | "html" | "json" | "markdown"
    show_plots: bool = True
    figsize_default: tuple = (15, 5)


@dataclass
class AgentConfig:
    """Settings for the agentic LLM layer."""

    enabled: bool = False
    api_key: Optional[str] = None
    model: str = "qwen/qwen3-4b"
    provider: str = "openrouter"  # "openrouter" | "openai" | "anthropic" | "gemini" | "ollama"
    base_url: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 4096
    timeout: int = 60

    def validate(self) -> None:
        """Validate agent configuration."""
        if not self.enabled:
            return

        if self.provider != "ollama" and not self.api_key:
            raise ValueError(
                "api_key is required when agents=True. "
                "Provide your API key for the chosen provider "
                "(OpenRouter, OpenAI, Anthropic, or Gemini)."
            )
        if not self.model:
            raise ValueError(
                "model is required when agents=True. "
                "Provide the exact model name, e.g. 'qwen/qwen3-4b'."
            )


@dataclass
class PipelineConfig:
    """Top-level configuration aggregating all settings."""

    analysis_level: AnalysisLevel = AnalysisLevel.DETAILED
    target_column: Optional[str] = None
    numerical_columns: Optional[List[str]] = None
    categorical_columns: Optional[List[str]] = None
    datetime_columns: Optional[List[str]] = None
    outlier_method: OutlierMethod = OutlierMethod.IQR
    correlation_methods: List[CorrelationType] = field(
        default_factory=lambda: [CorrelationType.PEARSON]
    )

    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    def validate(self) -> None:
        """Validate the entire configuration."""
        self.agent.validate()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Create configuration from a dictionary."""
        thresholds = ThresholdConfig(**data.pop("thresholds", {}))
        output = OutputConfig(**data.pop("output", {}))
        agent_data = data.pop("agent", {})
        agent = AgentConfig(**agent_data)

        # Convert string enums
        if "analysis_level" in data:
            data["analysis_level"] = AnalysisLevel(data["analysis_level"])
        if "outlier_method" in data:
            data["outlier_method"] = OutlierMethod(data["outlier_method"])

        return cls(
            thresholds=thresholds,
            output=output,
            agent=agent,
            **data,
        )
