"""Tests for backward compatibility with the legacy EDAPipeline API."""

import warnings

import numpy as np
import pandas as pd
import pytest

from edapipeline import EDAPipeline


@pytest.fixture
def legacy_df():
    np.random.seed(42)
    return pd.DataFrame({
        "numerical": np.random.rand(100),
        "categorical": np.random.choice(["A", "B", "C"], 100),
        "date": pd.date_range("2023-01-01", periods=100),
    })


class TestLegacyAPI:
    def test_initialization_warns(self, legacy_df):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            eda = EDAPipeline(legacy_df)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)
                           and "edapipeline" in str(x.filename).lower()]
            assert len(dep_warnings) >= 1
            assert "deprecated" in str(dep_warnings[0].message).lower()

    def test_initialization_properties(self, legacy_df):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            eda = EDAPipeline(legacy_df)
            assert len(eda.numerical_cols) == 1
            assert len(eda.categorical_cols) == 1
            assert len(eda.datetime_cols) == 1

    def test_data_overview_runs(self, legacy_df):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            eda = EDAPipeline(legacy_df)
            eda.data_overview()  # Should not raise

    def test_run_complete_analysis(self, legacy_df):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            eda = EDAPipeline(legacy_df)
            eda.run_complete_analysis()  # Should not raise
