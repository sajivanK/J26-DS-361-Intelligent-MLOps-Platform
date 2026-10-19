"""Tests for Multi-Dataset Repair Runner, Pilot Executor, Checkpointing, and Colab Notebooks."""

import json
from pathlib import Path
import tempfile
import pytest
import nbformat

from component3.src.contracts.prescription_contract import validate_prescription_record
from component3.src.experiments.multi_dataset_runner import MultiDatasetRepairRunner
from component3.src.experiments.pilot_executor import PilotExperimentExecutor


def test_multi_dataset_runner_tvae_execution():
    """Verify end-to-end execution of TVAE generator on UCI Adult with Sprint 05 validation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        runner = MultiDatasetRepairRunner(results_dir=Path(tmp_dir))
        record = runner.run_experiment(
            dataset_id="uci_adult",
            scenario_id="S01",
            generator_name="tvae",
            n_samples=50,
            epochs=1,
            seed=42,
            model_family="xgboost",
            max_rows=150,
        )

        assert record["generator"] == "TVAE"
        assert record["dataset_id"] == "uci_adult"
        assert record["structural_status"] in ("PASS", "FAIL")
        assert 0.0 <= record["fidelity_score"] <= 1.0
        assert record["package_status"] in ("APPROVED", "REVIEW", "REJECTED")
        assert record["prescriptive_viability_label"] in (0, 1)

        # Validate against strict contract
        validate_prescription_record(record)


def test_multi_dataset_runner_tabddpm_execution():
    """Verify execution of TabDDPM diffusion generator with Sprint 05 validation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        runner = MultiDatasetRepairRunner(results_dir=Path(tmp_dir))
        record = runner.run_experiment(
            dataset_id="uci_adult",
            scenario_id="S01",
            generator_name="tabddpm",
            n_samples=30,
            epochs=1,
            seed=42,
            model_family="random_forest",
            max_rows=150,
        )

        assert record["generator"] == "TABDDPM"
        assert record["target_coverage"] == 1.0
        assert record["total_seconds"] > 0
        validate_prescription_record(record)


def test_multi_dataset_runner_telco_churn_execution():
    """Verify execution on Telco Customer Churn dataset with CTGAN."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        runner = MultiDatasetRepairRunner(results_dir=Path(tmp_dir))
        record = runner.run_experiment(
            dataset_id="telco_churn",
            scenario_id="S01",
            generator_name="ctgan",
            n_samples=40,
            epochs=1,
            seed=42,
            model_family="xgboost",
            max_rows=150,
        )

        assert record["dataset_id"] == "telco_churn"
        assert record["generator"] == "CTGAN"
        validate_prescription_record(record)


def test_pilot_executor_checkpoint_and_resume():
    """Verify that PilotExperimentExecutor skips previously completed experiment runs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        res_dir = tmp_path / "results"
        chk_dir = tmp_path / "checkpoints"

        executor = PilotExperimentExecutor(results_dir=res_dir, checkpoint_dir=chk_dir)

        # First run: 1 experiment
        results1, summary1 = executor.run_pilot(
            datasets=["uci_adult"],
            generators=["tvae"],
            models=["xgboost"],
            n_samples=20,
            epochs=1,
            seed=42,
            max_rows=100,
        )
        assert summary1.total_executed == 1
        assert summary1.total_skipped_from_checkpoint == 0

        # Second run: should be skipped from checkpoint
        results2, summary2 = executor.run_pilot(
            datasets=["uci_adult"],
            generators=["tvae"],
            models=["xgboost"],
            n_samples=20,
            epochs=1,
            seed=42,
            max_rows=100,
        )
        assert summary2.total_executed == 0
        assert summary2.total_skipped_from_checkpoint == 1


def test_colab_notebooks_structural_validity():
    """Verify all 4 Google Colab notebooks conform to nbformat v4 schema and have execution headers."""
    colab_dir = Path(__file__).resolve().parents[1] / "colab"
    notebooks = [
        "01_colab_baseline_training.ipynb",
        "02_colab_drift_experiments.ipynb",
        "03_colab_synthetic_repair.ipynb",
        "04_colab_prescription_training.ipynb",
    ]

    for nb_name in notebooks:
        nb_path = colab_dir / nb_name
        assert nb_path.is_file(), f"Notebook {nb_name} not found in {colab_dir}"

        with nb_path.open("r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        # nbformat schema validation
        nbformat.validate(nb)

        # Check execution status notice in markdown header
        header_text = nb.cells[0].source
        assert "NOT YET EXECUTED" in header_text, f"{nb_name} missing execution status declaration"
        assert len(nb.cells) >= 4, f"{nb_name} has insufficient cells ({len(nb.cells)})"
