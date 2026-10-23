"""Unit tests for automated repair experiment runner."""

from pathlib import Path
import pytest

from component3.src.experiments.automated_repair_runner import (
    AutomatedRepairExperimentRunner,
    SCENARIO_RULES,
)


def test_runner_initialization(tmp_path):
    runner = AutomatedRepairExperimentRunner(results_dir=tmp_path)
    assert runner.results_dir == tmp_path
    assert runner.results_dir.is_dir()
    assert "ctgan" in runner.plugin_manager.available_generators()


def test_runner_invalid_inputs(tmp_path):
    runner = AutomatedRepairExperimentRunner(results_dir=tmp_path)

    with pytest.raises(ValueError, match="Unsupported scenario"):
        runner.run_experiment(scenario="INVALID_S99")

    with pytest.raises(ValueError, match="Unsupported generator"):
        runner.run_experiment(generator_name="invalid_gen")

    with pytest.raises(ValueError, match="positive integers"):
        runner.run_experiment(n_samples=0)

    with pytest.raises(ValueError, match="positive integers"):
        runner.run_experiment(epochs=0)


def test_runner_end_to_end_smoke(tmp_path):
    runner = AutomatedRepairExperimentRunner(results_dir=tmp_path)
    res = runner.run_experiment(
        scenario="S04",
        generator_name="ctgan",
        n_samples=20,
        epochs=1,
        seed=42,
        label_authorized=False,
    )

    assert "experiment_id" in res
    assert res["scenario"] == "S04"
    assert res["generator"] == "CTGAN"
    assert res["summary_scores"]["target_coverage"] == 1.0
    assert res["summary_scores"]["privacy_status"] in ("PASS", "FAIL")

    # Verify JSON file was created
    expected_file = tmp_path / f"{res['experiment_id']}.json"
    assert expected_file.is_file()
