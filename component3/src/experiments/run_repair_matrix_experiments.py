"""Execute representative experimental matrix across generators, scenarios, and repair quantities."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

from component3.src.experiments.automated_repair_runner import (
    AutomatedRepairExperimentRunner,
    SCENARIO_RULES,
)

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "component3" / "results" / "repair_experiments"


def execute_matrix() -> list[dict[str, Any]]:
    """Execute controlled representative repair experiment matrix."""
    runner = AutomatedRepairExperimentRunner(results_dir=OUTPUT_DIR)

    # Design representative matrix
    # Generators: CTGAN, TVAE, TabDDPM
    # Quantities: 200, 500, 1000, 2000
    # Scenarios: S04 (primary), S01 (continuous), S03 (categorical)
    experiments_plan = [
        # CTGAN across quantities on S04
        {"scenario": "S04", "generator": "ctgan", "n_samples": 200, "epochs": 5, "seed": 42},
        {"scenario": "S04", "generator": "ctgan", "n_samples": 500, "epochs": 5, "seed": 42},
        {"scenario": "S04", "generator": "ctgan", "n_samples": 1000, "epochs": 5, "seed": 42},
        {"scenario": "S04", "generator": "ctgan", "n_samples": 2000, "epochs": 5, "seed": 42},

        # TVAE across quantities on S04
        {"scenario": "S04", "generator": "tvae", "n_samples": 200, "epochs": 5, "seed": 42},
        {"scenario": "S04", "generator": "tvae", "n_samples": 500, "epochs": 5, "seed": 42},
        {"scenario": "S04", "generator": "tvae", "n_samples": 1000, "epochs": 5, "seed": 42},
        {"scenario": "S04", "generator": "tvae", "n_samples": 2000, "epochs": 5, "seed": 42},

        # TabDDPM on S04 (computationally bounded timesteps & epochs on CPU)
        {"scenario": "S04", "generator": "tabddpm", "n_samples": 200, "epochs": 3, "seed": 42},
        {"scenario": "S04", "generator": "tabddpm", "n_samples": 500, "epochs": 3, "seed": 42},

        # Cross-Scenario evaluations
        {"scenario": "S01", "generator": "ctgan", "n_samples": 200, "epochs": 5, "seed": 42},
        {"scenario": "S01", "generator": "tvae", "n_samples": 200, "epochs": 5, "seed": 42},
        {"scenario": "S03", "generator": "ctgan", "n_samples": 200, "epochs": 5, "seed": 42},
        {"scenario": "S03", "generator": "tvae", "n_samples": 200, "epochs": 5, "seed": 42},
    ]

    results = []
    print(f"Starting execution of {len(experiments_plan)} matrix experiments...")

    for i, exp in enumerate(experiments_plan, 1):
        start_t = time.perf_counter()
        print(f"[{i}/{len(experiments_plan)}] Running {exp['scenario']} | {exp['generator'].upper()} | N={exp['n_samples']} | Epochs={exp['epochs']}...")
        try:
            res = runner.run_experiment(
                scenario=exp["scenario"],
                generator_name=exp["generator"],
                n_samples=exp["n_samples"],
                epochs=exp["epochs"],
                seed=exp["seed"],
                label_authorized=True,
                mode="research",
            )
            scores = res["summary_scores"]
            elapsed = time.perf_counter() - start_t
            print(f"  -> Decision: {scores['decision']} | Coverage: {scores['target_coverage']:.2%} | Fidelity: {scores['fidelity_score']} | Utility: {scores['utility_score']} | Time: {elapsed:.2f}s")
            results.append(res)
        except Exception as exc:
            print(f"  -> FAILED: {exc}")

    print(f"\nExecution complete. Successfully executed {len(results)}/{len(experiments_plan)} experiments.")
    return results


if __name__ == "__main__":
    execute_matrix()
