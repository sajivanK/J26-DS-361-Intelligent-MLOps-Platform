"""Unit tests for Component 3 Experiment Matrix Planner."""

import pytest

from component3.src.experiments.matrix_planner import (
    ExperimentMatrixPlanner,
)


def test_generate_full_matrix_size():
    planner = ExperimentMatrixPlanner()
    matrix = planner.generate_full_matrix()
    # 5 datasets * 3 models * 8 drifts * 3 generators * 4 quantities * 3 seeds = 4320
    assert len(matrix) == 4320


def test_unique_deterministic_experiment_ids():
    planner = ExperimentMatrixPlanner()
    matrix = planner.generate_full_matrix(
        dataset_ids=["uci_adult"],
        model_families=["xgboost"],
        drift_categories=["numerical_distribution_shift", "categorical_distribution_shift"],
        generators=["CTGAN", "TVAE"],
        quantities=[200, 500],
        seeds=[42, 123],
    )
    # 1 * 1 * 2 * 2 * 2 * 2 = 16
    assert len(matrix) == 16
    ids = [e.experiment_id for e in matrix]
    assert len(ids) == len(set(ids))  # Zero duplicate IDs


def test_matrix_summary_calculations():
    planner = ExperimentMatrixPlanner()
    matrix = planner.generate_full_matrix(
        dataset_ids=["uci_adult", "telco_churn"],
        model_families=["xgboost"],
        drift_categories=["numerical_distribution_shift"],
        generators=["CTGAN", "TVAE"],
        quantities=[200],
        seeds=[42],
    )
    # 2 * 1 * 1 * 2 * 1 * 1 = 4
    assert len(matrix) == 4

    summary = planner.summarize_plan(matrix)
    assert summary.total_combinations == 4
    assert summary.estimated_total_cpu_hours > 0.0
    assert summary.estimated_total_gpu_hours > 0.0
    assert summary.breakdown_by_dataset["uci_adult"] == 2
    assert summary.breakdown_by_dataset["telco_churn"] == 2
    assert summary.breakdown_by_generator["CTGAN"] == 2
    assert summary.breakdown_by_generator["TVAE"] == 2


def test_sample_balanced_subset():
    planner = ExperimentMatrixPlanner()
    matrix = planner.generate_full_matrix()

    sample_size = 50
    subset = planner.sample_balanced_subset(matrix, sample_size=sample_size, seed=42)
    assert len(subset) == sample_size

    # Check that multiple datasets, generators, and drifts are represented in the sample
    represented_datasets = {e.dataset_id for e in subset}
    represented_generators = {e.generator for e in subset}
    assert len(represented_datasets) >= 3
    assert len(represented_generators) == 3


def test_filter_by_budget():
    planner = ExperimentMatrixPlanner()
    matrix = planner.generate_full_matrix(
        dataset_ids=["uci_adult"],
        model_families=["xgboost"],
        drift_categories=["numerical_distribution_shift"],
        generators=["CTGAN", "TVAE"],
        quantities=[200, 500, 1000],
        seeds=[42, 123],
    )

    # Tight budget: 0.01 hours = 36 seconds
    budget_subset = planner.filter_by_budget(matrix, max_budget_hours=0.01, use_gpu=False)
    assert len(budget_subset) < len(matrix)
    total_sec = sum(e.estimated_seconds_cpu for e in budget_subset)
    assert total_sec <= 36.0


def test_filter_completed_experiments():
    planner = ExperimentMatrixPlanner()
    matrix = planner.generate_full_matrix(
        dataset_ids=["uci_adult"],
        model_families=["xgboost"],
        drift_categories=["numerical_distribution_shift"],
        generators=["TVAE"],
        quantities=[200],
        seeds=[42, 123],
    )
    assert len(matrix) == 2

    completed = {matrix[0].experiment_id}
    pending = planner.filter_completed_experiments(matrix, completed)
    assert len(pending) == 1
    assert pending[0].experiment_id == matrix[1].experiment_id


def test_export_plan(tmp_path):
    planner = ExperimentMatrixPlanner()
    matrix = planner.generate_full_matrix(
        dataset_ids=["uci_adult"],
        model_families=["xgboost"],
        drift_categories=["numerical_distribution_shift"],
        generators=["TVAE"],
        quantities=[200],
        seeds=[42],
    )
    plan_file = tmp_path / "test_plan.json"
    planner.export_plan(matrix, plan_file)
    assert plan_file.is_file()


def test_runtime_provenance_marking():
    """Audit 6 Regression Test: matrix planner must clearly mark GPU estimates as unverified theoretical assumptions."""
    planner = ExperimentMatrixPlanner()
    matrix = planner.generate_full_matrix(
        dataset_ids=["uci_adult"],
        model_families=["xgboost"],
        drift_categories=["numerical_distribution_shift"],
        generators=["CTGAN"],
        quantities=[200],
        seeds=[42],
    )
    exp = matrix[0]
    assert hasattr(exp, "runtime_provenance")
    assert "gpu_unverified_assumption" in exp.runtime_provenance

    summary = planner.summarize_plan(matrix)
    assert hasattr(summary, "runtime_provenance")
    assert "GPU estimates are unverified" in summary.runtime_provenance

