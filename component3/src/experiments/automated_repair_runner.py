"""Automated synthetic tabular data repair experiment runner for Component 3."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from component3.src.contracts.safe_slice_evaluator import extract_affected_slice
from component3.src.core.config_loader import load_config
from component3.src.core.data_loader import load_current_dataset
from component3.src.generators.data_preparer import prepare_generator_data
from component3.src.generators.plugin_manager import GeneratorPluginManager
from component3.src.readiness.readiness_models import ReadinessDecision, ReadinessStatus
from component3.src.validation import (
    evaluate_offline_utility,
    evaluate_repair_acceptance,
    validate_fidelity,
    validate_privacy,
    validate_structural,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = ROOT / "component3" / "results" / "repair_experiments"

SCENARIO_RULES: dict[str, str] = {
    "S01": "age > 0.69",
    "S02": "capital_loss > -0.20",
    "S03": "marital_status == 2",
    "S04": "race == 2",
    "S05": "age > 0.69 and education_num < -0.03",
    "S06": "marital_status == 2 and relationship == 0",
    "S07": "age > 0.69 and marital_status == 2",
}


class AutomatedRepairExperimentRunner:
    """Orchestrates end-to-end synthetic data generation and multi-stage validation experiments."""

    def __init__(self, results_dir: Path | str | None = None) -> None:
        self.results_dir = Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.plugin_manager = GeneratorPluginManager()

    def run_experiment(
        self,
        scenario: str = "S04",
        generator_name: str = "ctgan",
        n_samples: int = 100,
        epochs: int = 1,
        seed: int = 42,
        label_authorized: bool = False,
        label_source: str | None = None,
        mode: str = "research",
    ) -> dict[str, Any]:
        """Execute a controlled repair experiment across all generation and validation stages."""
        if scenario not in SCENARIO_RULES:
            raise ValueError(f"Unsupported scenario '{scenario}'. Supported: {sorted(SCENARIO_RULES)}")
        if generator_name.lower() not in self.plugin_manager.available_generators():
            raise ValueError(f"Unsupported generator '{generator_name}'. Available: {self.plugin_manager.available_generators()}")
        if n_samples < 1 or epochs < 1:
            raise ValueError("n_samples and epochs must be positive integers.")

        total_start = time.perf_counter()

        # 1. Reproducibility Seeding
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # 2. Load Configuration and Datasets
        config = load_config()
        current_df = load_current_dataset(config, scenario)
        ref_train_path = ROOT / "shared" / "uci_adult" / "data" / "reference.csv"
        eval_test_path = ROOT / "shared" / "uci_adult" / "data" / "test.csv"
        model_path = ROOT / "shared" / "uci_adult" / "models" / "base_model_adult.pkl"

        ref_train = pd.read_csv(ref_train_path)
        eval_test = pd.read_csv(eval_test_path)
        baseline_model = joblib.load(model_path) if model_path.is_file() else None

        # 3. Extract Affected Slice
        slice_rule = SCENARIO_RULES[scenario]
        affected_slice = extract_affected_slice(current_df, slice_rule)

        # 4. Readiness Governance
        # Generator is always trained on features only (untrusted target excluded)
        decision = ReadinessDecision(
            status=ReadinessStatus.REVIEW,
            current_rows=len(current_df),
            affected_slice_rows=len(affected_slice),
            generation_ready=True,
            supervised_retraining_ready=False,
        )

        # 5. Prepare Generator Data
        prepared = prepare_generator_data(
            affected_slice=affected_slice,
            readiness_decision=decision,
            config=config,
        )

        # 6. Instantiate Generator via Plugin Manager
        gen_kwargs: dict[str, Any] = {
            "epochs": epochs,
            "discrete_columns": prepared.discrete_columns,
        }
        generator = self.plugin_manager.create(generator_name.lower(), **gen_kwargs)

        # 7. Fit Generator
        train_start = time.perf_counter()
        generator.fit(prepared.training_data)
        training_seconds = time.perf_counter() - train_start

        # 8. Sample Synthetic Records
        sample_start = time.perf_counter()
        synthetic_df = generator.sample(n_samples)
        sampling_seconds = time.perf_counter() - sample_start

        # Align feature column ordering
        feature_order = [c for c in ref_train.columns if c != "target"]
        synthetic_df = synthetic_df[feature_order].copy()

        # 9. Multi-Stage Validation
        val_start = time.perf_counter()

        # Stage 1: Structural & Target Coverage
        structural_result = validate_structural(
            synthetic_data=synthetic_df,
            prepared_data=prepared,
            slice_rule=slice_rule,
        )

        # Stage 2: Statistical Fidelity & Diversity
        fidelity_result = validate_fidelity(
            synthetic_data=synthetic_df,
            reference_data=prepared,
        )

        # Stage 3: Privacy Risk & DCR
        privacy_result = validate_privacy(
            synthetic_data=synthetic_df,
            reference_data=prepared,
            holdout_data=eval_test,
        )

        # Stage 4: Offline Utility Evaluation
        if label_authorized and baseline_model is not None:
            synthetic_for_eval = synthetic_df.copy()
            synthetic_for_eval["target"] = baseline_model.predict(synthetic_df)
        else:
            synthetic_for_eval = synthetic_df.copy()

        utility_result = evaluate_offline_utility(
            synthetic_data=synthetic_for_eval,
            reference_train_data=ref_train,
            evaluation_test_data=eval_test,
            slice_rule=slice_rule,
            target_column="target",
            baseline_model=baseline_model,
            label_authorized=label_authorized,
        )

        validation_seconds = time.perf_counter() - val_start

        # 10. Repair Acceptance Policy Decision
        acceptance_outcome = evaluate_repair_acceptance(
            structural=structural_result,
            fidelity=fidelity_result,
            privacy=privacy_result,
            utility=utility_result,
            mode=mode,
        )

        total_seconds = time.perf_counter() - total_start

        # 11. Compile Structured Experiment Record
        experiment_id = f"{scenario}_{generator_name.lower()}_e{epochs}_n{n_samples}_seed{seed}"
        fidelity_metrics = fidelity_result.metrics
        privacy_metrics = privacy_result.metrics
        utility_metrics = utility_result.metrics

        if label_authorized:
            effective_label_source = label_source if label_source is not None else "teacher_pseudo_label"
        else:
            effective_label_source = "unlabeled"

        evidence_report = {
            "experiment_id": experiment_id,
            "scenario": scenario,
            "dataset": "uci_adult",
            "generator": generator_name.upper(),
            "slice_rule": slice_rule,
            "seed": seed,
            "epochs": epochs,
            "requested_samples": n_samples,
            "generated_samples": len(synthetic_df),
            "label_authorized": label_authorized,
            "label_source": effective_label_source,
            "mode": mode,
            "training_seconds": round(training_seconds, 4),
            "sampling_seconds": round(sampling_seconds, 4),
            "validation_seconds": round(validation_seconds, 4),
            "total_seconds": round(total_seconds, 4),
            "summary_scores": {
                "package_status": acceptance_outcome.package_status,
                "decision": acceptance_outcome.decision.value,
                "fidelity_score": fidelity_metrics.get("fidelity_score"),
                "utility_score": utility_metrics.get("utility_score"),
                "target_coverage": (structural_result.metrics.get("target_coverage") or {}).get("coverage", 0.0),
                "privacy_status": privacy_result.status.value,
                "dcr_p05": privacy_metrics.get("dcr_metrics", {}).get("dcr_p05"),
                "exact_match_count": privacy_metrics.get("exact_matches", {}).get("exact_match_count", 0),
                "delta_overall_f1": utility_metrics.get("deltas", {}).get("delta_overall_f1"),
                "delta_affected_slice_f1": utility_metrics.get("deltas", {}).get("delta_affected_slice_f1"),
                "delta_non_affected_f1": utility_metrics.get("deltas", {}).get("delta_non_affected_f1"),
                "delta_baseline_slice_f1": (utility_metrics.get("deltas_vs_baseline") or {}).get("delta_affected_slice_f1"),
            },
            "acceptance_decision": acceptance_outcome.to_dict(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        # Save JSON artifact
        output_file = self.results_dir / f"{experiment_id}.json"
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(evidence_report, f, indent=2)

        return evidence_report


if __name__ == "__main__":
    runner = AutomatedRepairExperimentRunner()
    res = runner.run_experiment(
        scenario="S04",
        generator_name="ctgan",
        n_samples=100,
        epochs=1,
        seed=42,
        label_authorized=True,
    )
    scores = res["summary_scores"]
    print("\n=== AUTOMATED REPAIR EXPERIMENT COMPLETE ===")
    print("Experiment ID:", res["experiment_id"])
    print("Decision:", scores["decision"])
    print("Package Status:", scores["package_status"])
    print("Fidelity Score:", scores["fidelity_score"])
    print("Utility Score:", scores["utility_score"])
    print("Target Coverage:", scores["target_coverage"])
    print("Privacy Status:", scores["privacy_status"])
    print("DCR p05:", scores["dcr_p05"])
    print("Delta Slice F1:", scores["delta_affected_slice_f1"])
    print("Total Seconds:", res["total_seconds"])
    print("Output Path:", runner.results_dir / f"{res['experiment_id']}.json\n")
