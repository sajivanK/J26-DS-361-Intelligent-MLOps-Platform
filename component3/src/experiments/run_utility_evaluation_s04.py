"""Sprint 05 Task 05: S04 offline predictive utility evaluation experiment."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time

import joblib
import numpy as np
import pandas as pd
import torch

from component3.src.contracts.safe_slice_evaluator import extract_affected_slice
from component3.src.core.config_loader import load_config
from component3.src.core.data_loader import load_current_dataset
from component3.src.generators.ctgan_generator import CTGANGenerator
from component3.src.generators.data_preparer import prepare_generator_data
from component3.src.readiness.readiness_models import ReadinessDecision, ReadinessStatus
from component3.src.validation import evaluate_offline_utility

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "component3" / "results" / "utility_evaluation"

SCENARIO = "S04"
SLICE_RULE = "race == 2"
SEED = 42
EPOCHS = 1
N_SAMPLES = 100


def run_experiment() -> dict:
    """Execute S04 offline utility evaluation comparing unauthorized and authorized oracle conditions."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    config = load_config()
    current_df = load_current_dataset(config, SCENARIO)
    affected_slice = extract_affected_slice(current_df, SLICE_RULE)

    # 1. Unsupervised Generation
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
    generator.fit(prepared.training_data)
    synthetic_df = generator.sample(N_SAMPLES)

    # 2. Condition A: Production / Unsupervised Default (Labels NOT authorized)
    ref_train = pd.read_csv(ROOT / "shared" / "uci_adult" / "data" / "reference.csv")
    eval_test = pd.read_csv(ROOT / "shared" / "uci_adult" / "data" / "test.csv")
    baseline_model = joblib.load(ROOT / "shared" / "uci_adult" / "models" / "base_model_adult.pkl")

    unauthorized_res = evaluate_offline_utility(
        synthetic_data=synthetic_df,
        reference_train_data=ref_train,
        evaluation_test_data=eval_test,
        slice_rule=SLICE_RULE,
        target_column="target",
        baseline_model=baseline_model,
        label_authorized=False,
    )

    # 3. Condition B: Controlled Research Oracle (Exploratory label assignment via baseline model)
    # Research oracle assigns pseudo-labels to synthetic features for offline utility simulation
    feature_order = [c for c in ref_train.columns if c != "target"]
    synthetic_df = synthetic_df[feature_order].copy()
    synthetic_labeled = synthetic_df.copy()
    synthetic_labeled["target"] = baseline_model.predict(synthetic_df)

    authorized_res = evaluate_offline_utility(
        synthetic_data=synthetic_labeled,
        reference_train_data=ref_train,
        evaluation_test_data=eval_test,
        slice_rule=SLICE_RULE,
        target_column="target",
        baseline_model=baseline_model,
        label_authorized=True,
    )

    evidence = {
        "experiment": "sprint05_task05_s04_utility_evaluation",
        "scenario": SCENARIO,
        "slice_rule": SLICE_RULE,
        "seed": SEED,
        "n_samples": N_SAMPLES,
        "unauthorized_evaluation": unauthorized_res.to_dict(),
        "authorized_oracle_evaluation": authorized_res.to_dict(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"{SCENARIO}_ctgan_utility_e{EPOCHS}_n{N_SAMPLES}_seed{SEED}.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(evidence, file, indent=2)

    return evidence


if __name__ == "__main__":
    report = run_experiment()
    unauth = report["unauthorized_evaluation"]
    auth = report["authorized_oracle_evaluation"]

    print("\n=== S04 OFFLINE UTILITY EVALUATION EXPERIMENT ===")
    print("Scenario:", report["scenario"])
    print("\n--- Condition A: Unauthorized (Governance Gate Active) ---")
    print("Status:", unauth["status"])
    print("Reasons:", unauth["reasons"])

    print("\n--- Condition B: Authorized Controlled Oracle Retraining ---")
    print("Status:", auth["status"])
    print("Reasons:", auth["reasons"])
    print("Utility Score:", auth["metrics"]["utility_score"])
    print("Deltas:", auth["metrics"]["deltas"])
    print("Overall Baseline vs Augmented:",
          "Base Acc:", auth["metrics"]["baseline_metrics"]["overall"]["accuracy"],
          "Aug Acc:", auth["metrics"]["augmented_retrained_metrics"]["overall"]["accuracy"])
    print("Affected Slice Baseline vs Augmented F1:",
          "Base F1:", auth["metrics"]["baseline_metrics"]["affected_slice"]["f1"],
          "Aug F1:", auth["metrics"]["augmented_retrained_metrics"]["affected_slice"]["f1"])
    print("Non-Affected Baseline vs Augmented F1:",
          "Base F1:", auth["metrics"]["baseline_metrics"]["non_affected"]["f1"],
          "Aug F1:", auth["metrics"]["augmented_retrained_metrics"]["non_affected"]["f1"])
    print("\nResults saved to:", RESULTS_DIR)
    print("EXPERIMENT RUNNER: COMPLETE\n")
