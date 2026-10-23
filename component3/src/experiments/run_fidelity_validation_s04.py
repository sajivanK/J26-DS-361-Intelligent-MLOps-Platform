"""Sprint 05 Task 03: S04 CTGAN fidelity validation experiment."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from component3.src.contracts.safe_slice_evaluator import extract_affected_slice
from component3.src.core.config_loader import load_config
from component3.src.core.data_loader import load_current_dataset
from component3.src.generators.ctgan_generator import CTGANGenerator
from component3.src.generators.data_preparer import prepare_generator_data
from component3.src.readiness.readiness_models import ReadinessDecision, ReadinessStatus
from component3.src.validation import validate_fidelity

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "component3" / "results" / "fidelity_validation"

SCENARIO = "S04"
SLICE_RULE = "race == 2"
SEED = 42
EPOCHS = 1
N_SAMPLES = 100


def run_experiment() -> dict:
    """Execute S04 CTGAN generation and statistical fidelity evaluation."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    config = load_config()
    current_df = load_current_dataset(config, SCENARIO)
    affected_slice = extract_affected_slice(current_df, SLICE_RULE)

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
        epochs=EPOCHS,
        discrete_columns=prepared.discrete_columns,
    )
    training_start = time.perf_counter()
    generator.fit(prepared.training_data)
    training_seconds = time.perf_counter() - training_start

    sampling_start = time.perf_counter()
    synthetic_df = generator.sample(N_SAMPLES)
    sampling_seconds = time.perf_counter() - sampling_start

    # Run fidelity validator
    fidelity_result = validate_fidelity(
        synthetic_data=synthetic_df,
        reference_data=prepared,
    )

    evidence = {
        "experiment": "sprint05_task03_s04_fidelity_validation",
        "scenario": SCENARIO,
        "dataset": "uci_adult",
        "generator": "ctgan",
        "slice_rule": SLICE_RULE,
        "seed": SEED,
        "epochs": EPOCHS,
        "n_samples": N_SAMPLES,
        "current_rows": int(len(current_df)),
        "affected_slice_rows": int(len(affected_slice)),
        "generator_training_rows": int(prepared.n_rows),
        "generated_rows": int(len(synthetic_df)),
        "training_seconds": round(training_seconds, 4),
        "sampling_seconds": round(sampling_seconds, 4),
        "readiness_source": "synthetic_experiment_decision",
        "fidelity_validation": fidelity_result.to_dict(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"{SCENARIO}_ctgan_fidelity_e{EPOCHS}_n{N_SAMPLES}_seed{SEED}.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(evidence, file, indent=2)

    return evidence


if __name__ == "__main__":
    report = run_experiment()
    val = report["fidelity_validation"]
    metrics = val["metrics"]

    print("\n=== S04 CTGAN FIDELITY VALIDATION EXPERIMENT ===")
    print("Scenario:", report["scenario"])
    print("Generator:", report["generator"])
    print("Validation status:", val["status"])
    print("Reasons:", val["reasons"])
    print("Composite fidelity score:", metrics.get("fidelity_score"))
    print("Continuous summary:", metrics.get("continuous_metrics_summary"))
    print("Categorical summary:", metrics.get("categorical_metrics_summary"))
    print("Correlation diff:", metrics.get("correlation_difference"))
    print("Diversity:", metrics.get("diversity_metrics"))
    print("Results saved to:", RESULTS_DIR / f"{report['scenario']}_ctgan_fidelity_e{report['epochs']}_n{report['n_samples']}_seed{report['seed']}.json")
    print("EXPERIMENT RUNNER: COMPLETE\n")
