"""Automated CTGAN targeted generation experiment runner for Component 3."""

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
from component3.src.generators.ctgan_generator import CTGANGenerator
from component3.src.generators.data_preparer import prepare_generator_data
from component3.src.generators.target_coverage import evaluate_target_coverage
from component3.src.readiness.readiness_models import ReadinessDecision, ReadinessStatus

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "component3" / "results" / "ctgan_targeted"

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
) -> dict:
    """Execute a controlled targeted CTGAN generation experiment for a scenario."""
    if scenario not in SCENARIO_RULES:
        raise ValueError(
            f"Unsupported scenario '{scenario}'. Supported: {sorted(SCENARIO_RULES)}"
        )
    if epochs < 1 or n_samples < 1:
        raise ValueError("epochs and n_samples must be positive.")

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

    generator = CTGANGenerator(
        epochs=epochs,
        discrete_columns=prepared.discrete_columns,
    )

    start = time.perf_counter()
    generator.fit(prepared.training_data)
    training_seconds = time.perf_counter() - start

    synthetic_df = generator.sample(n_samples)
    coverage = evaluate_target_coverage(synthetic_df, rule)

    result = {
        "scenario": scenario,
        "slice_rule": rule,
        "seed": seed,
        "epochs": epochs,
        "training_rows": prepared.n_rows,
        "generation_mode": prepared.generation_mode,
        "n_generated": len(synthetic_df),
        "training_seconds": round(training_seconds, 4),
        "target_coverage": coverage.to_dict(),
        "generated_columns": synthetic_df.columns.tolist(),
        "missing_values": int(synthetic_df.isna().sum().sum()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "readiness_source": "synthetic_experiment_decision",
        "validation_status": "diagnostic_only",
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / (
        f"{scenario}_ctgan_e{epochs}_n{n_samples}_seed{seed}.json"
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run targeted CTGAN generation experiment."
    )
    parser.add_argument(
        "--scenario",
        default="S04",
        help="Drift scenario identifier (S01-S07)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of synthetic samples to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    res = run_experiment(
        scenario=args.scenario,
        epochs=args.epochs,
        n_samples=args.samples,
        seed=args.seed,
    )

    print("\n=== CTGAN TARGETED EXPERIMENT ===")
    print("Scenario:", res["scenario"])
    print("Slice rule:", res["slice_rule"])
    print("Seed:", res["seed"])
    print("Epochs:", res["epochs"])
    print("Training rows:", res["training_rows"])
    print("Generation mode:", res["generation_mode"])
    print("Generated rows:", res["n_generated"])
    print("Training seconds:", res["training_seconds"])
    print("Target coverage:", res["target_coverage"]["coverage"])
    print("Matching rows:", res["target_coverage"]["matching_rows"])
    print("Non-matching rows:", res["target_coverage"]["non_matching_rows"])
    print("Missing values:", res["missing_values"])
    print("Results saved to:", RESULTS_DIR)
    print("\nEXPERIMENT RUNNER: COMPLETE")
