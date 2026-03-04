"""Tests for the new modular Pipeline."""

import numpy as np
import pandas as pd
import pytest

from edapipeline import Pipeline, PipelineConfig, PipelineReport
from edapipeline.types import ColumnType, OutlierMethod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """Create a representative test DataFrame."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "age": np.random.randint(18, 80, n),
        "income": np.random.lognormal(10, 1, n),
        "score": np.random.normal(50, 10, n),
        "category": np.random.choice(["A", "B", "C", "D"], n),
        "gender": np.random.choice(["M", "F"], n),
        "signup_date": pd.date_range("2020-01-01", periods=n, freq="D"),
        "is_active": np.random.choice([True, False], n),
    })


@pytest.fixture
def small_df():
    """Minimal DataFrame for edge-case tests."""
    return pd.DataFrame({
        "num": [1.0, 2.0, 3.0, 4.0, 5.0],
        "cat": ["x", "y", "x", "y", "x"],
    })


# ---------------------------------------------------------------------------
# Pipeline Initialization
# ---------------------------------------------------------------------------

class TestPipelineInit:
    def test_basic_init(self, sample_df):
        p = Pipeline(sample_df, show_plots=False)
        assert p.profile.n_rows == 200
        assert p.profile.n_cols == 7

    def test_with_target(self, sample_df):
        p = Pipeline(sample_df, target_col="income", show_plots=False)
        assert p.config.target_column == "income"
        # Target should not be in numerical columns list
        assert "income" not in p.profile.numerical_columns

    def test_invalid_target_raises(self, sample_df):
        with pytest.raises(ValueError, match="not found"):
            Pipeline(sample_df, target_col="nonexistent", show_plots=False)

    def test_empty_df_raises(self):
        with pytest.raises(ValueError, match="empty"):
            Pipeline(pd.DataFrame(), show_plots=False)

    def test_non_df_raises(self):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            Pipeline([1, 2, 3], show_plots=False)

    def test_column_detection(self, sample_df):
        p = Pipeline(sample_df, show_plots=False)
        assert "age" in p.profile.numerical_columns
        assert "category" in p.profile.categorical_columns
        assert "signup_date" in p.profile.datetime_columns


# ---------------------------------------------------------------------------
# Full Pipeline Run
# ---------------------------------------------------------------------------

class TestPipelineRun:
    def test_full_run_returns_report(self, sample_df):
        p = Pipeline(sample_df, show_plots=False)
        report = p.run()
        assert isinstance(report, PipelineReport)
        assert report.execution_time_seconds > 0

    def test_full_run_has_all_results(self, sample_df):
        p = Pipeline(sample_df, show_plots=False)
        report = p.run()
        expected = {
            "overview", "missing_values", "numerical",
            "categorical", "datetime", "correlation",
            "bivariate", "outlier",
        }
        assert set(report.results.keys()) == expected

    def test_selective_run(self, sample_df):
        p = Pipeline(sample_df, show_plots=False)
        report = p.run(analyses=["overview", "numerical"])
        assert "overview" in report.results
        assert "numerical" in report.results
        assert "correlation" not in report.results

    def test_report_to_json(self, sample_df):
        p = Pipeline(sample_df, show_plots=False)
        report = p.run(analyses=["overview"])
        json_str = report.to_json()
        assert '"overview"' in json_str

    def test_report_summary(self, sample_df):
        p = Pipeline(sample_df, show_plots=False)
        report = p.run(analyses=["overview"])
        s = report.summary()
        assert "200" in s


# ---------------------------------------------------------------------------
# Individual Analyzers via Pipeline
# ---------------------------------------------------------------------------

class TestAnalyzers:
    def test_overview(self, sample_df):
        p = Pipeline(sample_df, show_plots=False)
        result = p.data_overview()
        assert result.n_rows == 200

    def test_missing_values_clean(self, sample_df):
        p = Pipeline(sample_df, show_plots=False)
        result = p.missing_values()
        assert result.total_missing_cells == 0

    def test_missing_values_with_nans(self, sample_df):
        df = sample_df.copy()
        df.loc[0:9, "income"] = np.nan
        p = Pipeline(df, show_plots=False)
        result = p.missing_values()
        assert result.total_missing_cells == 10
        assert result.columns_with_missing == 1

    def test_numerical_analysis(self, sample_df):
        p = Pipeline(sample_df, show_plots=False)
        result = p.analyze_numerical()
        assert len(result.features) > 0
        age_feat = next(f for f in result.features if f.column == "age")
        assert age_feat.count > 0
        assert age_feat.skewness is not None

    def test_categorical_analysis(self, sample_df):
        p = Pipeline(sample_df, show_plots=False)
        result = p.analyze_categorical()
        assert len(result.features) > 0
        cat_feat = next(f for f in result.features if f.column == "category")
        assert cat_feat.unique_count == 4
        assert cat_feat.entropy is not None

    def test_correlation_analysis(self, sample_df):
        p = Pipeline(sample_df, show_plots=False)
        result = p.correlation_analysis()
        assert result.matrix is not None
        assert len(result.top_positive) > 0 or len(result.top_negative) > 0

    def test_outlier_iqr(self, sample_df):
        p = Pipeline(sample_df, show_plots=False)
        result = p.detect_outliers(method="iqr")
        assert len(result.features) > 0

    def test_outlier_zscore(self, sample_df):
        p = Pipeline(sample_df, show_plots=False)
        result = p.detect_outliers(method="zscore")
        assert result.method == "zscore"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_default_config(self):
        cfg = PipelineConfig()
        assert cfg.thresholds.high_cardinality == 50
        assert cfg.output.plot_dpi == 300
        assert cfg.agent.enabled is False

    def test_agent_validation_no_key(self):
        cfg = PipelineConfig()
        cfg.agent.enabled = True
        cfg.agent.provider = "openrouter"
        cfg.agent.api_key = None
        with pytest.raises(ValueError, match="api_key is required"):
            cfg.validate()

    def test_agent_validation_no_model(self):
        cfg = PipelineConfig()
        cfg.agent.enabled = True
        cfg.agent.api_key = "test-key"
        cfg.agent.model = ""
        with pytest.raises(ValueError, match="model is required"):
            cfg.validate()

    def test_ollama_no_key_required(self):
        cfg = PipelineConfig()
        cfg.agent.enabled = True
        cfg.agent.provider = "ollama"
        cfg.agent.api_key = None
        cfg.agent.model = "qwen3:4b"
        cfg.validate()  # Should not raise

    def test_from_dict(self):
        data = {
            "target_column": "price",
            "thresholds": {"high_cardinality": 100},
            "output": {"plot_dpi": 150},
            "agent": {"enabled": False},
        }
        cfg = PipelineConfig.from_dict(data)
        assert cfg.target_column == "price"
        assert cfg.thresholds.high_cardinality == 100
        assert cfg.output.plot_dpi == 150
