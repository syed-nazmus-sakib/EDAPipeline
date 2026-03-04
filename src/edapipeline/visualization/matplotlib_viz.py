"""Matplotlib/Seaborn visualization backend."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from ..config import PipelineConfig
from ..results import (
    CategoricalAnalysisResult,
    CorrelationResult,
    DatetimeAnalysisResult,
    MissingValueResult,
    NumericalAnalysisResult,
    OutlierAnalysisResult,
    BivariateAnalysisResult,
)
from ..types import DatasetProfile
from .base import BaseVisualizer

logger = logging.getLogger("edapipeline.viz")


class MatplotlibVisualizer(BaseVisualizer):
    """Generates plots using Matplotlib and Seaborn."""

    def __init__(self) -> None:
        sns.set_theme(style="whitegrid")
        sns.set_palette("husl")
        self._plot_counter = 0

    @property
    def plot_count(self) -> int:
        return self._plot_counter

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_and_show(
        self, fig: Any, name: str, config: PipelineConfig,
    ) -> None:
        """Save plot if configured, show it, then close."""
        if config.output.save_outputs:
            self._plot_counter += 1
            plots_dir = Path(config.output.output_dir) / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{self._plot_counter:03d}_{name}.{config.output.plot_format}"
            filepath = plots_dir / filename
            fig.savefig(filepath, dpi=config.output.plot_dpi, bbox_inches="tight")
            logger.info("Saved: %s", filepath)

        if config.output.show_plots:
            plt.show()
        plt.close(fig)

    # ------------------------------------------------------------------
    # Render methods
    # ------------------------------------------------------------------

    def render_overview(self, df: pd.DataFrame, result: Any, config: PipelineConfig) -> None:
        """Overview doesn't produce plots."""
        pass

    def render_missing(self, df: pd.DataFrame, result: MissingValueResult, config: PipelineConfig) -> None:
        if result.total_missing_cells == 0:
            return

        fig = plt.figure(figsize=config.output.figsize_default)
        sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap="viridis")
        plt.title("Missing Value Heatmap")
        self._save_and_show(fig, "missing_values_heatmap", config)

    def render_numerical(self, df: pd.DataFrame, result: NumericalAnalysisResult, config: PipelineConfig) -> None:
        for feat in result.features:
            col = feat.column
            fig, axes = plt.subplots(1, 3, figsize=config.output.figsize_default)
            fig.suptitle(f"Distribution Analysis: '{col}'", fontsize=16)

            # Histogram + KDE
            sns.histplot(data=df, x=col, kde=True, ax=axes[0])
            axes[0].set_title("Histogram & KDE")

            # Box plot
            sns.boxplot(y=df[col], ax=axes[1])
            axes[1].set_title("Box Plot")

            # Q-Q plot
            clean = df[col].dropna()
            stats.probplot(clean, dist="norm", plot=axes[2])
            axes[2].set_title("Q-Q Plot (vs Normal)")
            axes[2].set_xlabel("Theoretical Quantiles")
            axes[2].set_ylabel("Sample Quantiles")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            self._save_and_show(fig, f"numerical_{col}", config)

    def render_categorical(self, df: pd.DataFrame, result: CategoricalAnalysisResult, config: PipelineConfig) -> None:
        thresholds = config.thresholds

        for feat in result.features:
            col = feat.column
            n_unique = feat.unique_count

            if n_unique == 0 or n_unique > thresholds.high_cardinality:
                continue

            value_counts = df[col].value_counts()
            value_pcts = df[col].value_counts(normalize=True) * 100

            if n_unique > thresholds.medium_cardinality:
                # Medium cardinality — top N bar
                fig = plt.figure(figsize=(max(config.output.figsize_default[0] * 0.7, 8), max(n_unique * 0.3, 5)))
                top_n = value_counts.head(thresholds.top_n_categories)
                sns.barplot(y=top_n.index, x=top_n.values, orient="h")
                plt.title(f"Top {thresholds.top_n_categories} Categories: {col}")
                plt.xlabel("Count")
                plt.ylabel(col)
                plt.tight_layout()
                self._save_and_show(fig, f"categorical_{col}_top{thresholds.top_n_categories}", config)
            else:
                # Low cardinality — full visualization
                fig = plt.figure(figsize=config.output.figsize_default)
                fig.suptitle(f"Distribution Analysis: '{col}'", fontsize=16)

                plt.subplot(1, 3, 1)
                sns.countplot(data=df, y=col, order=value_counts.index, orient="h")
                plt.title("Count Plot")
                plt.xlabel("Count")

                plt.subplot(1, 3, 2)
                value_pcts.plot(kind="barh")
                plt.title("Percentage Distribution")
                plt.xlabel("Percentage")
                plt.ylabel(col)

                if n_unique <= 10:
                    plt.subplot(1, 3, 3)
                    plt.pie(
                        value_pcts, labels=value_pcts.index,
                        autopct="%1.1f%%", startangle=90, counterclock=False,
                    )
                    plt.title("Pie Chart")
                else:
                    ax3 = fig.add_subplot(1, 3, 3)
                    ax3.text(
                        0.5, 0.5, f"Pie chart omitted\n({n_unique} categories)",
                        ha="center", va="center", fontsize=12,
                    )
                    ax3.axis("off")

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                self._save_and_show(fig, f"categorical_{col}", config)

    def render_datetime(self, df: pd.DataFrame, result: DatetimeAnalysisResult, config: PipelineConfig) -> None:
        for feat in result.features:
            col = feat.column
            series = df[col]

            if not pd.api.types.is_datetime64_any_dtype(series):
                try:
                    series = pd.to_datetime(series, errors="coerce")
                except Exception:
                    continue

            fig = plt.figure(figsize=(15, 10))
            fig.suptitle(f"DateTime Analysis: '{col}'", fontsize=16)

            plt.subplot(2, 2, 1)
            sns.countplot(data=df, x=series.dt.year)
            plt.title("Records per Year")
            plt.xticks(rotation=45)

            plt.subplot(2, 2, 2)
            sns.countplot(data=df, x=series.dt.month, palette="viridis")
            plt.title("Records per Month")
            months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            plt.xticks(ticks=np.arange(12), labels=months)

            plt.subplot(2, 2, 3)
            sns.countplot(data=df, x=series.dt.dayofweek, palette="magma")
            plt.title("Records by Day of Week")
            days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            plt.xticks(ticks=np.arange(7), labels=days)

            plt.subplot(2, 2, 4)
            sns.countplot(data=df, x=series.dt.hour, palette="plasma")
            plt.title("Records by Hour of Day")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            self._save_and_show(fig, f"datetime_{col}_distribution", config)

    def render_correlation(self, df: pd.DataFrame, result: CorrelationResult, config: PipelineConfig) -> None:
        if result.matrix is None:
            return

        # Rebuild matrix as DataFrame for plotting
        cols = list(result.matrix.keys())
        matrix_df = pd.DataFrame(result.matrix, index=cols, columns=cols).astype(float)

        fig = plt.figure(figsize=(12, 8))
        sns.heatmap(
            matrix_df, annot=True, cmap="coolwarm", fmt=".2f",
            linewidths=0.5, center=0,
        )
        plt.title("Correlation Matrix Heatmap")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        self._save_and_show(fig, "correlation_heatmap", config)

        # Pairplot (if ≤ 6 numerical features)
        num_cols = config.numerical_columns or [
            c for c in df.select_dtypes(include=np.number).columns
            if c != config.target_column
        ]
        if 2 <= len(num_cols) <= 6:
            pairplot_hue = None
            target = config.target_column
            cat_cols = config.categorical_columns or []
            if target and target in cat_cols:
                if df[target].nunique() < config.thresholds.target_cardinality:
                    pairplot_hue = target

            plot_cols = list(num_cols) + ([target] if pairplot_hue else [])
            g = sns.pairplot(
                df[plot_cols], hue=pairplot_hue,
                diag_kind="kde", plot_kws={"alpha": 0.6},
            )
            plt.suptitle("Pair Plot of Numerical Features", y=1.02)
            self._save_and_show(g.fig, "pairplot_numerical", config)

    def render_bivariate(self, df: pd.DataFrame, result: BivariateAnalysisResult, config: PipelineConfig) -> None:
        # Numerical × Categorical
        for item in result.num_cat_results:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(f"'{item.numerical_col}' by '{item.categorical_col}'", fontsize=16)

            sns.boxplot(x=df[item.categorical_col], y=df[item.numerical_col], ax=axes[0])
            axes[0].set_title("Box Plot")
            axes[0].tick_params(axis="x", rotation=45)

            sns.violinplot(x=df[item.categorical_col], y=df[item.numerical_col], ax=axes[1])
            axes[1].set_title("Violin Plot")
            axes[1].tick_params(axis="x", rotation=45)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            self._save_and_show(
                fig,
                f"bivariate_{item.numerical_col}_vs_{item.categorical_col}",
                config,
            )

        # Numerical × Numerical
        for item in result.num_num_results:
            target = config.target_column
            cat_cols = config.categorical_columns or []
            hue_col = None
            if target and target in cat_cols:
                if df[target].nunique() < config.thresholds.target_cardinality:
                    hue_col = target

            try:
                g = sns.jointplot(
                    data=df, x=item.col_1, y=item.col_2,
                    hue=hue_col, kind="scatter", height=6.4,
                )
                g.fig.suptitle(f"Joint Plot: '{item.col_1}' vs '{item.col_2}'", y=1.02)

                if item.correlation is not None:
                    g.ax_joint.text(
                        0.1, 0.9, f"r = {item.correlation:.3f}",
                        transform=g.ax_joint.transAxes,
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                    )

                plt.tight_layout()
                self._save_and_show(
                    g.fig,
                    f"numerical_bivariate_{item.col_1}_vs_{item.col_2}",
                    config,
                )
            except Exception as exc:
                logger.warning("Joint plot failed for %s vs %s: %s", item.col_1, item.col_2, exc)

    def render_outlier(self, df: pd.DataFrame, result: OutlierAnalysisResult, config: PipelineConfig) -> None:
        """Outlier analysis doesn't produce plots in the current version."""
        pass
