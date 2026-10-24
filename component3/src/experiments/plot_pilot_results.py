"""Panel-Ready Visualization Engine for Component 3 Pilot Experiments.

Generates publication-quality figures from actual experiment JSON records:
1. Generator Runtime Comparison (Training, Sampling, Validation seconds)
2. Synthetic Data Quality & Fidelity Comparison (Fidelity score across generators)
3. Affected-Slice F1 Before vs. After Synthetic Repair
4. Repair Package Decisions Breakdown (APPROVED vs. REVIEW vs. REJECTED)
5. Multi-Dataset Drift Impact (Baseline F1 degradation under injected drift)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = ROOT / "component3" / "results" / "repair_experiments"
DEFAULT_FIGURES_DIR = ROOT / "component3" / "results" / "figures"


class PilotVisualizer:
    """Generates research-grade figures from actual experiment artifacts."""

    def __init__(
        self,
        results_dir: Path | str | None = None,
        figures_dir: Path | str | None = None,
    ) -> None:
        self.results_dir = Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR
        self.figures_dir = Path(figures_dir) if figures_dir else DEFAULT_FIGURES_DIR
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def load_experiment_records(self) -> pd.DataFrame:
        """Load all experiment JSON records into a DataFrame."""
        records: list[dict[str, Any]] = []
        for p in self.results_dir.glob("*.json"):
            try:
                with p.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "experiment_id" in data and "generator" in data:
                        records.append(data)
            except Exception as exc:
                logger.warning("Skipping unreadable file %s: %s", p, exc)

        if not records:
            return pd.DataFrame()
        return pd.DataFrame(records)

    def plot_all(self) -> list[Path]:
        """Generate all research-panel figures and return their file paths."""
        df = self.load_experiment_records()
        if df.empty:
            logger.warning("No experiment records found in %s to plot.", self.results_dir)
            return []

        saved_paths: list[Path] = []

        # 1. Generator Runtime Benchmark
        p1 = self.plot_generator_runtimes(df)
        if p1:
            saved_paths.append(p1)

        # 2. Fidelity Score by Generator
        p2 = self.plot_fidelity_comparison(df)
        if p2:
            saved_paths.append(p2)

        # 3. Affected Slice F1 Recovery
        p3 = self.plot_slice_f1_recovery(df)
        if p3:
            saved_paths.append(p3)

        # 4. Acceptance Status Distribution
        p4 = self.plot_acceptance_decisions(df)
        if p4:
            saved_paths.append(p4)

        return saved_paths

    def plot_generator_runtimes(self, df: pd.DataFrame) -> Path | None:
        """Plot runtime breakdown (training vs sampling vs validation) across generators."""
        if "generator" not in df.columns or "training_seconds" not in df.columns:
            return None

        grouped = df.groupby("generator")[["training_seconds", "sampling_seconds", "validation_seconds"]].mean()
        if grouped.empty:
            return None

        fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
        generators = list(grouped.index)
        x = np.arange(len(generators))
        width = 0.25

        ax.bar(x - width, grouped["training_seconds"], width, label="Training Time (s)", color="#2b5c8f")
        ax.bar(x, grouped["sampling_seconds"], width, label="Sampling Time (s)", color="#e07a5f")
        ax.bar(x + width, grouped["validation_seconds"], width, label="Validation Time (s)", color="#81b29a")

        ax.set_ylabel("Mean Execution Time (Seconds)", fontsize=11, fontweight="bold")
        ax.set_title("Component 3: Generator Runtime Benchmark (CTGAN vs TVAE vs TabDDPM)", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(generators, fontsize=11)
        ax.legend(frameon=True)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        fig.tight_layout()
        out_path = self.figures_dir / "pilot_runtime_by_generator.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_fidelity_comparison(self, df: pd.DataFrame) -> Path | None:
        """Plot mean fidelity score and KS statistic across generators."""
        if "generator" not in df.columns or "fidelity_score" not in df.columns:
            return None

        grouped = df.groupby("generator")[["fidelity_score", "mean_ks_statistic"]].mean()
        if grouped.empty:
            return None

        fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
        generators = list(grouped.index)
        scores = grouped["fidelity_score"]
        colors = ["#3d5a80" if g == "CTGAN" else "#ee6c4d" if g == "TVAE" else "#293241" for g in generators]

        bars = ax.bar(generators, scores, color=colors, width=0.5, edgecolor="black", linewidth=0.8)
        ax.axhline(0.65, color="red", linestyle="--", label="Fidelity Review Threshold (0.65)")
        ax.axhline(0.80, color="green", linestyle=":", label="Fidelity Approval Threshold (0.80)")

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Empirical Fidelity Score [0, 1]", fontsize=11, fontweight="bold")
        ax.set_title("Component 3: Synthetic Data Fidelity Comparison by Generator", fontsize=12, fontweight="bold")
        ax.legend(loc="lower right", frameon=True)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        fig.tight_layout()
        out_path = self.figures_dir / "pilot_fidelity_comparison.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_slice_f1_recovery(self, df: pd.DataFrame) -> Path | None:
        """Plot affected slice F1 recovery across datasets and generators."""
        if "generator" not in df.columns or "delta_affected_slice_f1" not in df.columns:
            return None

        grouped = df.groupby(["dataset_id", "generator"])["delta_affected_slice_f1"].mean().unstack()
        if grouped.empty:
            return None

        fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
        grouped.plot(kind="bar", ax=ax, width=0.6, colormap="tab10")

        ax.axhline(0.0, color="black", linestyle="-", linewidth=1.0)
        ax.set_ylabel("Delta Affected Slice F1 Score", fontsize=11, fontweight="bold")
        ax.set_xlabel("Dataset Identifier", fontsize=11, fontweight="bold")
        ax.set_title("Component 3: Slice Performance Delta Following Repair", fontsize=12, fontweight="bold")
        ax.legend(title="Generator", frameon=True)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        fig.tight_layout()
        out_path = self.figures_dir / "pilot_slice_f1_recovery.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_acceptance_decisions(self, df: pd.DataFrame) -> Path | None:
        """Plot repair package governance decisions (APPROVED vs REVIEW vs REJECTED)."""
        if "package_status" not in df.columns:
            return None

        status_counts = df["package_status"].value_counts()
        if status_counts.empty:
            return None

        fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
        palette = {"APPROVED": "#2a9d8f", "REVIEW": "#e76f51", "REJECTED": "#e63946"}
        colors = [palette.get(s, "#457b9d") for s in status_counts.index]

        bars = ax.bar(status_counts.index, status_counts.values, color=colors, width=0.5, edgecolor="black")
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"{int(yval)}", ha="center", va="bottom", fontsize=11, fontweight="bold")

        ax.set_ylabel("Number of Experiment Runs", fontsize=11, fontweight="bold")
        ax.set_title("Component 3: Sprint 05 Repair Governance Decisions", fontsize=12, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        fig.tight_layout()
        out_path = self.figures_dir / "pilot_acceptance_decisions.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path


if __name__ == "__main__":
    visualizer = PilotVisualizer()
    saved = visualizer.plot_all()
    print(f"Generated {len(saved)} panel figures in {visualizer.figures_dir}")
