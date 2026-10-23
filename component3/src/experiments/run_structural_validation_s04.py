"""Sprint 05 Task 02: S04 CTGAN structural validation smoke experiment."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from component3.src.contracts.safe_slice_evaluator import (
    extract_affected_slice,
)
from component3.src.core.config_loader import load_config
from component3.src.core.data_loader import load_current_dataset
from component3.src.generators.ctgan_generator import CTGANGenerator
from component3.src.generators.data_preparer import (
    prepare_generator_data,
)
from component3.src.readiness.readiness_models import (
    ReadinessDecision,
    ReadinessStatus,
)
from component3.src.validation import validate_structural

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = (
    ROOT / "component3" / "results" / "structural_validation"
)

SCENARIO = "S04"
SLICE_RULE = "race == 2"
SEED = 42
EPOCHS = 1
N_SAMPLES = 100
MIN_TARGET_COVERAGE = 0.80


def run_experiment() -> dict:
    """Execute S04 CTGAN generation and structural validation experiment."""
    # ---------------------------------------------------------
    # STEP 1: Set random seeds for reproducibility.
    # ---------------------------------------------------------
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ---------------------------------------------------------
    # STEP 2: Load configuration and current drift dataset.
    # ---------------------------------------------------------
    config = load_config()
    current_df = load_current_dataset(config, SCENARIO)

    # ---------------------------------------------------------
    # STEP 3: Extract the affected slice.
    # ---------------------------------------------------------
    affected_slice = extract_affected_slice(current_df, SLICE_RULE)

    # ---------------------------------------------------------
    # STEP 4: Exploratory-only readiness authorization.
    # ---------------------------------------------------------
    decision = ReadinessDecision(
        status=ReadinessStatus.REVIEW,
        current_rows=len(current_df),
        affected_slice_rows=len(affected_slice),
        generation_ready=True,
        supervised_retraining_ready=False,
    )

    # ---------------------------------------------------------
    # STEP 5: Prepare generator training data and metadata.
    # ---------------------------------------------------------
    prepared = prepare_generator_data(
        affected_slice,
        decision,
        config,
    )

    # ---------------------------------------------------------
    # STEP 6: Instantiate and fit CTGAN generator.
    # ---------------------------------------------------------
    generator = CTGANGenerator(
        epochs=EPOCHS,
        discrete_columns=prepared.discrete_columns,
    )
    training_start = time.perf_counter()
    generator.fit(prepared.training_data)
    training_seconds = time.perf_counter() - training_start

    # ---------------------------------------------------------
    # STEP 7: Generate 100 synthetic records.
    # ---------------------------------------------------------
    sampling_start = time.perf_counter()
    synthetic_df = generator.sample(N_SAMPLES)
    sampling_seconds = time.perf_counter() - sampling_start

    # ---------------------------------------------------------
    # STEP 8: Execute structural validation.
    # ---------------------------------------------------------
    validation_result = validate_structural(
        synthetic_data=synthetic_df,
        prepared_data=prepared,
        slice_rule=SLICE_RULE,
        min_target_coverage=MIN_TARGET_COVERAGE,
    )

    # ---------------------------------------------------------
    # STEP 9: Create the experiment evidence report.
    # ---------------------------------------------------------
    report = {
        "experiment": "sprint05_task02_s04_structural_validation",
        "scenario": SCENARIO,
        "dataset": "uci_adult",
        "generator": "ctgan",
        "slice_rule": SLICE_RULE,
        "seed": SEED,
        "epochs": EPOCHS,
        "minimum_target_coverage": MIN_TARGET_COVERAGE,
        "current_rows": int(len(current_df)),
        "affected_slice_rows": int(len(affected_slice)),
        "generator_training_rows": int(prepared.n_rows),
        "generated_rows": int(len(synthetic_df)),
        "generated_columns": synthetic_df.columns.tolist(),
        "continuous_columns": list(prepared.continuous_columns),
        "categorical_columns": list(prepared.discrete_columns),
        "generation_mode": prepared.generation_mode,
        "target_included": bool(prepared.target_included),
        "training_seconds": round(training_seconds, 4),
        "sampling_seconds": round(sampling_seconds, 4),
        "readiness_source": "synthetic_experiment_decision",
        "validation": validation_result.to_dict(),
        "fidelity_status": "NOT_EVALUATED",
        "privacy_status": "NOT_EVALUATED",
        "utility_status": "NOT_EVALUATED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    # ---------------------------------------------------------
    # STEP 10: Persist evidence JSON report.
    # ---------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = (
        RESULTS_DIR
        / f"{SCENARIO}_ctgan_structural_e{EPOCHS}_n{N_SAMPLES}_seed{SEED}.json"
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    return report


if __name__ == "__main__":
    report = run_experiment()
    val = report["validation"]
    metrics = val["metrics"]

    print("\n=== S04 CTGAN STRUCTURAL VALIDATION EXPERIMENT ===")
    print("Scenario:", report["scenario"])
    print("Slice rule:", report["slice_rule"])
    print("Seed:", report["seed"])
    print("Epochs:", report["epochs"])
    print("Current dataset rows:", report["current_rows"])
    print("Affected slice rows:", report["affected_slice_rows"])
    print("Generator training rows:", report["generator_training_rows"])
    print("Generated rows:", report["generated_rows"])
    print("Training seconds:", report["training_seconds"])
    print("Sampling seconds:", report["sampling_seconds"])
    print("\n--- Structural Validation Outcome ---")
    print("Validation status:", val["status"])
    print("Reasons:", val["reasons"])
    if "target_coverage" in metrics:
        tc = metrics["target_coverage"]
        print("Target coverage:", tc.get("coverage"))
        print("Matching rows:", tc.get("matching_rows"))
        print("Non-matching rows:", tc.get("non_matching_rows"))
    print("Missing values:", metrics.get("missing_values", 0))
    print("Unseen categories:", metrics.get("unseen_categories", {}))
    print(
        "Numerical range diagnostics count:",
        len(metrics.get("numerical_range_violations", {})),
    )
    out_of_range_summary = {
        col: diag["out_of_range_count"]
        for col, diag in metrics.get("numerical_range_violations", {}).items()
        if diag["out_of_range_count"] > 0
    }
    print("Columns with out-of-range values:", out_of_range_summary)
    print(
        "\nResults saved to:",
        RESULTS_DIR
        / f"{report['scenario']}_ctgan_structural_e{report['epochs']}_n{report['n_samples'] if 'n_samples' in report else N_SAMPLES}_seed{report['seed']}.json",
    )
    print("EXPERIMENT RUNNER: COMPLETE\n")
