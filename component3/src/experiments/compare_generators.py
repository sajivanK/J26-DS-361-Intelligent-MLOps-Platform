"""Comparison engine for Component 3 synthetic tabular data generators."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = ROOT / "component3" / "results"
OUTPUT_DIR = RESULTS_ROOT / "generator_comparison"


class ComparisonError(RuntimeError):
    """Raised when experiment results are missing or inconsistent."""


def require_nonnegative_number(
    value: Any, field_name: str, file_name: str
) -> float | int:
    """Validate that a value is a non-negative finite number and not boolean."""
    if isinstance(value, bool):
        raise ComparisonError(
            f"Field '{field_name}' in {file_name} cannot be boolean: {value}"
        )
    if not isinstance(value, (int, float)):
        raise ComparisonError(
            f"Field '{field_name}' in {file_name} must be numeric: {value}"
        )
    if not math.isfinite(value):
        raise ComparisonError(
            f"Field '{field_name}' in {file_name} must be finite: {value}"
        )
    if value < 0:
        raise ComparisonError(
            f"Field '{field_name}' in {file_name} cannot be negative: {value}"
        )
    return value


def load_experiments(results_root: Path = RESULTS_ROOT) -> pd.DataFrame:
    """Load and validate all 63 experiment results across three generators."""
    results_root = Path(results_root)
    generators = ("ctgan", "tvae", "tabddpm")
    scenarios = ("S01", "S02", "S03", "S04", "S05", "S06", "S07")
    seeds = (42, 123, 456)
    expected = {
        (gen, sc, seed)
        for gen in generators
        for sc in scenarios
        for seed in seeds
    }

    seen = set()
    errors = []
    rows = []

    for generator in generators:
        gen_dir = results_root / f"{generator}_targeted"
        if not gen_dir.exists():
            errors.append(f"Directory not found: {gen_dir}")
            continue

        for path in sorted(gen_dir.glob("*.json")):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as exc:
                errors.append(f"Failed to load {path}: {exc}")
                continue

            scenario = data.get("scenario")
            seed = data.get("seed")
            combo = (generator, scenario, seed)
            if combo in seen:
                errors.append(
                    f"Duplicate experiment combination: {combo} in {path.name}"
                )
            seen.add(combo)

            cov = data.get("target_coverage", {})
            range_diag = data.get("numerical_range_diagnostics")
            observed_violations = None
            if range_diag is not None and isinstance(range_diag, dict):
                observed_violations = range_diag.get("total_invalid_values")

            try:
                training_seconds = require_nonnegative_number(
                    data.get("training_seconds"), "training_seconds", path.name
                )
            except ComparisonError as exc:
                errors.append(str(exc))
                continue

            sampling_seconds = data.get("sampling_seconds")
            if sampling_seconds is not None:
                try:
                    sampling_seconds = require_nonnegative_number(
                        sampling_seconds, "sampling_seconds", path.name
                    )
                except ComparisonError as exc:
                    errors.append(str(exc))
                    continue

            rows.append(
                {
                    "generator": generator,
                    "scenario": scenario,
                    "slice_rule": data.get("slice_rule"),
                    "seed": seed,
                    "epochs": data.get("epochs"),
                    "training_rows": data.get("training_rows"),
                    "generated_rows": data.get("n_generated"),
                    "matching_rows": cov.get("matching_rows"),
                    "nonmatching_rows": cov.get("non_matching_rows"),
                    "coverage": cov.get("coverage"),
                    "training_seconds": training_seconds,
                    "sampling_seconds": sampling_seconds,
                    "observed_range_violations": observed_violations,
                    "missing_values": data.get("missing_values", 0),
                    "created_at_utc": data.get("created_at_utc"),
                    "readiness_source": data.get("readiness_source"),
                    "validation_status": data.get("validation_status"),
                    "artifact_name": path.name,
                }
            )

    missing = sorted(expected - seen)
    if missing:
        errors.append(f"Missing experiment combinations: {missing}")
    if errors:
        raise ComparisonError("\n".join(errors))
    if len(rows) != 63:
        raise ComparisonError(
            f"Expected 63 experiments; found {len(rows)}"
        )
    return pd.DataFrame(rows).sort_values(
        ["generator", "scenario", "seed"]
    ).reset_index(drop=True)


def aggregate_results(df: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Aggregate coverage counts and descriptive timing metrics."""
    summary = (
        df.groupby(group_columns, sort=True, dropna=False)
        .agg(
            experiments=("seed", "count"),
            training_rows_min=("training_rows", "min"),
            training_rows_max=("training_rows", "max"),
            generated_rows=("generated_rows", "sum"),
            matching_rows=("matching_rows", "sum"),
            nonmatching_rows=("nonmatching_rows", "sum"),
            missing_values=("missing_values", "sum"),
            mean_training_seconds=("training_seconds", "mean"),
            median_training_seconds=("training_seconds", "median"),
            min_training_seconds=("training_seconds", "min"),
            max_training_seconds=("training_seconds", "max"),
            mean_sampling_seconds=("sampling_seconds", "mean"),
            observed_range_violations=(
                "observed_range_violations",
                lambda values: (
                    values.sum(min_count=1)
                ),
            ),
        )
        .reset_index()
    )
    summary["aggregate_coverage"] = (
        summary["matching_rows"] / summary["generated_rows"]
    )
    summary["aggregate_coverage_percent"] = (
        summary["aggregate_coverage"] * 100
    )
    return summary


def run_comparison(
    results_root: Path = RESULTS_ROOT,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Any]:
    """Validate evidence and write reproducible comparison artifacts."""
    results_root = Path(results_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_experiments(results_root)
    generator_summary = aggregate_results(df, ["generator"])
    scenario_summary = aggregate_results(
        df, ["generator", "scenario"]
    )

    detailed_path = output_dir / "generator_comparison_detailed.csv"
    summary_path = output_dir / "generator_comparison_summary.csv"
    scenario_path = output_dir / "generator_comparison_scenario.csv"
    validation_path = output_dir / "generator_comparison_validation.json"

    df.to_csv(detailed_path, index=False)
    generator_summary.to_csv(summary_path, index=False)
    scenario_summary.to_csv(scenario_path, index=False)

    validation_report = {
        "validation_status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_experiments": int(len(df)),
        "expected_experiments": 63,
        "generators": sorted(df["generator"].unique().tolist()),
        "scenarios": sorted(df["scenario"].unique().tolist()),
        "seeds": sorted(df["seed"].unique().tolist()),
        "total_generated_rows": int(df["generated_rows"].sum()),
        "total_matching_rows": int(df["matching_rows"].sum()),
        "total_missing_values": int(df["missing_values"].sum()),
        "generator_summaries": {
            row["generator"]: {
                "experiments": int(row["experiments"]),
                "generated_rows": int(row["generated_rows"]),
                "matching_rows": int(row["matching_rows"]),
                "aggregate_coverage_percent": round(
                    float(row["aggregate_coverage_percent"]), 4
                ),
                "mean_training_seconds": round(
                    float(row["mean_training_seconds"]), 4
                ),
                "mean_sampling_seconds": (
                    None
                    if pd.isna(row["mean_sampling_seconds"])
                    else round(float(row["mean_sampling_seconds"]), 4)
                ),
                "observed_range_violations": (
                    None
                    if pd.isna(row["observed_range_violations"])
                    else int(row["observed_range_violations"])
                ),
            }
            for _, row in generator_summary.iterrows()
        },
        "errors": [],
    }

    with open(validation_path, "w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=2)

    return {
        "df": df,
        "generator_summary": generator_summary,
        "scenario_summary": scenario_summary,
        "validation_report": validation_report,
        "detailed_path": detailed_path,
        "summary_path": summary_path,
        "scenario_path": scenario_path,
        "validation_path": validation_path,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING GENERATOR COMPARISON ENGINE")
    print("=" * 60)
    results = run_comparison()
    gen_sum = results["generator_summary"]

    print("\n--- GENERATOR SUMMARY (AGGREGATE) ---")
    display_cols = [
        "generator",
        "experiments",
        "generated_rows",
        "matching_rows",
        "aggregate_coverage_percent",
        "mean_training_seconds",
        "mean_sampling_seconds",
        "observed_range_violations",
    ]
    print(gen_sum[display_cols].to_string(index=False))

    print("\n--- ARTIFACTS CREATED ---")
    print(f"Detailed CSV:    {results['detailed_path']}")
    print(f"Summary CSV:     {results['summary_path']}")
    print(f"Scenario CSV:    {results['scenario_path']}")
    print(f"Validation JSON: {results['validation_path']}")
    print("\nValidation Status: PASS (63/63 experiments verified)")
