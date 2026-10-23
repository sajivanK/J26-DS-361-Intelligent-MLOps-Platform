"""Automated TabDDPM targeted generation experiment runner for Component 3."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch

from component3.src.contracts.safe_slice_evaluator import extract_affected_slice
from component3.src.core.config_loader import load_config
from component3.src.core.data_loader import load_current_dataset
from component3.src.generators.data_preparer import prepare_generator_data
from component3.src.generators.tabddpm_generator import TabDDPMGenerator
from component3.src.generators.target_coverage import evaluate_target_coverage
from component3.src.readiness.readiness_models import ReadinessDecision, ReadinessStatus

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "component3" / "results" / "tabddpm_targeted"

SCENARIO_RULES = {
    "S01": "age > 0.69",
    "S02": "capital_loss > -0.20",
    "S03": "marital_status == 2",
    "S04": "race == 2",
    "S05": "age > 0.69 and education_num < -0.03",
    "S06": "marital_status == 2 and relationship == 0",
    "S07": "age > 0.69 and marital_status == 2",
}


def run_experiment(
    scenario: str = "S04",
    epochs: int = 1,
    n_samples: int = 100,
    seed: int = 42,
    num_timesteps: int = 10,
    batch_size: int = 128,
) -> dict:
    """Execute a controlled targeted TabDDPM generation experiment for a scenario."""
    if scenario not in SCENARIO_RULES:
        raise ValueError(
            f"Unsupported scenario '{scenario}'. Supported: {sorted(SCENARIO_RULES)}"
        )
    if epochs < 1 or n_samples < 1:
        raise ValueError("epochs and n_samples must be positive.")
    if num_timesteps < 1 or batch_size < 1:
        raise ValueError("num_timesteps and batch_size must be positive.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    config = load_config()
    current_df = load_current_dataset(config, scenario)
    rule = SCENARIO_RULES[scenario]
    affected_slice = extract_affected_slice(current_df, rule)

    # Exploratory-only authorization for this controlled experiment.
    # This is NOT a verified decision from the real readiness engine.
    decision = ReadinessDecision(
        status=ReadinessStatus.REVIEW,
        current_rows=len(current_df),
        affected_slice_rows=len(affected_slice),
        generation_ready=True,
        supervised_retraining_ready=False,
    )

    prepared = prepare_generator_data(
        affected_slice,
        decision,
        config,
    )

    generator = TabDDPMGenerator(
        epochs=epochs,
        batch_size=batch_size,
        num_timesteps=num_timesteps,
        discrete_columns=prepared.discrete_columns,
    )

    start_train = time.perf_counter()
    generator.fit(prepared.training_data)
    training_seconds = time.perf_counter() - start_train

    start_sample = time.perf_counter()
    synthetic_df = generator.sample(n_samples)
    sampling_seconds = time.perf_counter() - start_sample

    coverage = evaluate_target_coverage(synthetic_df, rule)
    range_diagnostics = generator.validate_numerical_ranges(synthetic_df)

    result = {
        "scenario": scenario,
        "slice_rule": rule,
        "seed": seed,
        "epochs": epochs,
        "num_timesteps": num_timesteps,
        "batch_size": batch_size,
        "training_rows": prepared.n_rows,
        "generation_mode": prepared.generation_mode,
        "n_generated": len(synthetic_df),
        "training_seconds": round(training_seconds, 4),
        "sampling_seconds": round(sampling_seconds, 4),
        "target_coverage": coverage.to_dict(),
        "numerical_range_diagnostics": range_diagnostics,
        "generated_columns": synthetic_df.columns.tolist(),
        "missing_values": int(synthetic_df.isna().sum().sum()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "readiness_source": "synthetic_experiment_decision",
        "validation_status": "diagnostic_only",
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / (
        f"{scenario}_tabddpm_e{epochs}_t{num_timesteps}_b{batch_size}_n{n_samples}_seed{seed}.json"
    )
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a controlled TabDDPM targeted generation experiment."
    )
    parser.add_argument("--scenario", default="S04")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    result = run_experiment(
        scenario=args.scenario,
        epochs=args.epochs,
        n_samples=args.samples,
        seed=args.seed,
        num_timesteps=args.timesteps,
        batch_size=args.batch_size,
    )

    print("\n=== TABDDPM TARGETED EXPERIMENT ===")
    print("Scenario:", result["scenario"])
    print("Slice rule:", result["slice_rule"])
    print("Seed:", result["seed"])
    print("Epochs:", result["epochs"])
    print("Diffusion timesteps:", result["num_timesteps"])
    print("Training rows:", result["training_rows"])
    print("Generation mode:", result["generation_mode"])
    print("Generated rows:", result["n_generated"])
    print("Training seconds:", result["training_seconds"])
    print("Sampling seconds:", result["sampling_seconds"])
    print("Target coverage:", result["target_coverage"]["coverage"])
    print("Matching rows:", result["target_coverage"]["matching_rows"])
    print("Non-matching rows:", result["target_coverage"]["non_matching_rows"])
    print(
        "Observed-range violations:",
        result["numerical_range_diagnostics"]["total_invalid_values"],
    )
    print("Missing values:", result["missing_values"])
    print("Results saved to:", RESULTS_DIR)
    print("\nEXPERIMENT RUNNER: COMPLETE")
