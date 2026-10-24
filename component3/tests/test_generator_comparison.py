"""Tests for the Component 3 generator comparison engine."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from component3.src.experiments.compare_generators import (
    ComparisonError,
    aggregate_results,
    load_experiments,
    require_nonnegative_number,
    run_comparison,
)


def test_reject_negative_numeric_value():
    with pytest.raises(ComparisonError):
        require_nonnegative_number(-1, "training_seconds", "test.json")


def test_reject_nonfinite_numeric_value():
    with pytest.raises(ComparisonError):
        require_nonnegative_number(
            float("nan"), "training_seconds", "test.json"
        )


def test_reject_boolean_numeric_value():
    with pytest.raises(ComparisonError):
        require_nonnegative_number(True, "training_seconds", "test.json")


def test_accept_valid_numeric_value():
    assert (
        require_nonnegative_number(12.34, "training_seconds", "test.json")
        == 12.34
    )
    assert (
        require_nonnegative_number(0, "missing_values", "test.json") == 0
    )


def test_load_experiments_success():
    df = load_experiments()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 63
    assert set(df["generator"].unique()) == {"ctgan", "tvae", "tabddpm"}
    assert set(df["seed"].unique()) == {42, 123, 456}
    assert len(df["scenario"].unique()) == 7


def test_load_missing_directory_raises(tmp_path):
    with pytest.raises(ComparisonError, match="Directory not found"):
        load_experiments(tmp_path / "non_existent_results")


def test_load_missing_combinations_raises(tmp_path):
    for gen in ("ctgan", "tvae", "tabddpm"):
        (tmp_path / f"{gen}_targeted").mkdir(parents=True)
    with pytest.raises(ComparisonError, match="Missing experiment combinations"):
        load_experiments(tmp_path)


def test_load_corrupted_json_raises(tmp_path):
    for gen in ("ctgan", "tvae", "tabddpm"):
        (tmp_path / f"{gen}_targeted").mkdir(parents=True)
    bad_file = tmp_path / "ctgan_targeted" / "bad.json"
    bad_file.write_text("{bad_json: invalid", encoding="utf-8")
    with pytest.raises(ComparisonError):
        load_experiments(tmp_path)


def test_aggregate_results_generator_level():
    df = load_experiments()
    summary = aggregate_results(df, ["generator"])
    assert len(summary) == 3
    assert set(summary["generator"]) == {"ctgan", "tvae", "tabddpm"}
    for _, row in summary.iterrows():
        assert row["experiments"] == 21
        assert row["generated_rows"] == 2100
        assert 0.0 <= row["aggregate_coverage_percent"] <= 100.0


def test_aggregate_results_scenario_level():
    df = load_experiments()
    summary = aggregate_results(df, ["generator", "scenario"])
    assert len(summary) == 21
    for _, row in summary.iterrows():
        assert row["experiments"] == 3
        assert row["generated_rows"] == 300


def test_run_comparison_creates_artifacts(tmp_path):
    out_dir = tmp_path / "comparison_output"
    results = run_comparison(output_dir=out_dir)
    assert results["detailed_path"].exists()
    assert results["summary_path"].exists()
    assert results["scenario_path"].exists()
    assert results["validation_path"].exists()

    detailed_df = pd.read_csv(results["detailed_path"])
    assert len(detailed_df) == 63
    summary_df = pd.read_csv(results["summary_path"])
    assert len(summary_df) == 3
    scenario_df = pd.read_csv(results["scenario_path"])
    assert len(scenario_df) == 21


def test_run_comparison_validation_report_contents(tmp_path):
    out_dir = tmp_path / "comparison_output"
    results = run_comparison(output_dir=out_dir)
    val = results["validation_report"]
    assert val["validation_status"] == "PASS"
    assert val["total_experiments"] == 63
    assert val["total_generated_rows"] == 6300
    assert val["total_matching_rows"] == 6265
    assert val["total_missing_values"] == 0
    assert val["errors"] == []
