"""Runnable local demonstration workflows for Component 3 (Scenarios A through H).

Demonstrates real working execution across:
A. No drift / no justified repair -> NO_REPAIR
B. Valid drift with authentic labels -> Authentic adaptation candidate
C. Valid drift with synthetic generation -> Generator, validation, and retraining
D. Insufficient labeled data -> Safe abstention (NO_REPAIR)
E. Invalid / Oracle diagnostic payload -> TARGETED_REPAIR_BLOCKED
F. Invalid synthetic records -> Rejection
G. Resource limits constrained -> Graceful fallback
H. Accepted candidate -> Versioned RepairPackage without deployment
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any

# Ensure workspace root is in sys.path when script is executed directly
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import pandas as pd

from component3.src.contracts.repair_package import RepairPackage, RepairPackageStatus
from component3.src.contracts.repair_request import (
    DriftSeverity,
    DriftType,
    ExecutionMode,
    InvalidRepairRequestError,
    RepairObjective,
    RepairRequest,
)
from component3.src.contracts.targeted_repair import (
    DiagnosticProvenance,
    DiagnosticSliceDefinition,
    TargetedRepairStatus,
)
from component3.src.orchestration.orchestrator_models import ExecutionState
from component3.src.orchestration.repair_orchestrator import RepairOrchestrator
from component3.src.prescription.prescription_models import RepairStrategy
from component3.src.validation.validation_models import AcceptanceDecision
from component3.src.validation.validation_pipeline import UnifiedValidationPipeline

REPO_ROOT = Path(__file__).resolve().parents[3]
DEMO_OUTPUT_DIR = REPO_ROOT / "component3" / "results" / "demonstrations"


def run_all_demonstrations(
    output_dir: Path | str | None = None,
) -> dict[str, dict[str, Any]]:
    """Execute Scenarios A through H and return demonstration outcomes."""
    out_dir = Path(output_dir) if output_dir else DEMO_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    orchestrator = RepairOrchestrator()
    results: dict[str, dict[str, Any]] = {}

    print("=" * 78)
    print("COMPONENT 3: RUNNING LOCAL DEMONSTRATION WORKFLOWS (SCENARIOS A - H)")
    print("=" * 78)

    # -------------------------------------------------------------------------
    # Scenario A: No Drift Detected -> NO_REPAIR
    # -------------------------------------------------------------------------
    print("\n--- Scenario A: No Drift Detected ---")
    req_a = RepairRequest(
        request_id="DEMO-SCENARIO-A",
        drift_detected=False,
        drift_type=DriftType.NONE,
        drift_severity=DriftSeverity.NONE,
        drift_severity_score=0.0,
    )
    pkg_a, log_a = orchestrator.execute_repair(req_a, output_directory=out_dir / "scenario_a")
    print(f"Scenario A Outcome: Status={pkg_a.package_status.value}, Strategy={pkg_a.chosen_strategy}")
    assert pkg_a.package_status == RepairPackageStatus.ABSTAINED_NO_REPAIR
    assert pkg_a.chosen_strategy == "NO_REPAIR"
    results["scenario_a"] = {
        "status": pkg_a.package_status.value,
        "strategy": pkg_a.chosen_strategy,
        "passed": True,
    }

    # -------------------------------------------------------------------------
    # Scenario B: Valid Drift, Low Labeled Budget (B=100) -> Authentic Weighted Adaptation
    # -------------------------------------------------------------------------
    print("\n--- Scenario B: Valid Drift with Low Labeled Budget (B=100) ---")
    req_b = RepairRequest(
        request_id="DEMO-SCENARIO-B",
        drift_detected=True,
        drift_type=DriftType.CONCEPT_DRIFT,
        drift_severity=DriftSeverity.HIGH,
        drift_severity_score=0.60,
        adaptation_budget_size=100,
    )
    pkg_b, log_b = orchestrator.execute_repair(req_b, output_directory=out_dir / "scenario_b")
    print(f"Scenario B Outcome: Status={pkg_b.package_status.value}, Strategy={pkg_b.chosen_strategy}")
    assert pkg_b.chosen_strategy == "AUTHENTIC_WEIGHTED_ADAPTATION"
    results["scenario_b"] = {
        "status": pkg_b.package_status.value,
        "strategy": pkg_b.chosen_strategy,
        "passed": True,
    }

    # -------------------------------------------------------------------------
    # Scenario C: Valid Drift (B=200), Synthetic Generation Enabled -> Hybrid Repair
    # -------------------------------------------------------------------------
    print("\n--- Scenario C: Valid Drift with Synthetic Augmentation Enabled ---")
    cand_path = REPO_ROOT / "component3" / "results" / "experimental_models" / "batch05k_candidates_tvae_k50.csv"
    pre_gen_pool = pd.read_csv(cand_path) if cand_path.is_file() else None

    req_c = RepairRequest(
        request_id="DEMO-SCENARIO-C",
        drift_detected=True,
        drift_type=DriftType.CONCEPT_DRIFT,
        drift_severity=DriftSeverity.HIGH,
        drift_severity_score=0.65,
        adaptation_budget_size=200,
        source_retention_tolerance=0.05,
    )
    pkg_c, log_c = orchestrator.execute_repair(
        req_c,
        output_directory=out_dir / "scenario_c",
        pre_generated_candidates_df=pre_gen_pool,
    )
    print(f"Scenario C Outcome: Status={pkg_c.package_status.value}, Strategy={pkg_c.chosen_strategy}, Qty={pkg_c.quantity_selected}")
    assert pkg_c.chosen_strategy == "SYNTHETIC_WEIGHTED_HYBRID"
    results["scenario_c"] = {
        "status": pkg_c.package_status.value,
        "strategy": pkg_c.chosen_strategy,
        "quantity_selected": pkg_c.quantity_selected,
        "passed": True,
    }

    # -------------------------------------------------------------------------
    # Scenario D: Insufficient Labeled Data -> Safe Abstention (NO_REPAIR)
    # -------------------------------------------------------------------------
    print("\n--- Scenario D: Unlabeled Drift / Zero Adaptation Data ---")
    req_d = RepairRequest(
        request_id="DEMO-SCENARIO-D",
        drift_detected=True,
        has_authentic_adaptation_labels=False,
        adaptation_budget_size=0,
    )
    pkg_d, log_d = orchestrator.execute_repair(req_d, output_directory=out_dir / "scenario_d")
    print(f"Scenario D Outcome: Status={pkg_d.package_status.value}, Strategy={pkg_d.chosen_strategy}")
    assert pkg_d.package_status == RepairPackageStatus.ABSTAINED_NO_REPAIR
    results["scenario_d"] = {
        "status": pkg_d.package_status.value,
        "strategy": pkg_d.chosen_strategy,
        "passed": True,
    }

    # -------------------------------------------------------------------------
    # Scenario E: Invalid / Oracle Diagnostic Slice -> TARGETED_REPAIR_BLOCKED
    # -------------------------------------------------------------------------
    print("\n--- Scenario E: Oracle Diagnostic Slice Rejected ---")
    oracle_slice = DiagnosticSliceDefinition(
        slice_id="ORACLE-RULE-01",
        slice_predicate="tenure > 29.0",
        affected_features=("tenure",),
        reported_slice_support=0.50,
        provenance=DiagnosticProvenance.SIMULATION_ORACLE,
        is_non_oracle=False,
    )
    req_e = RepairRequest(
        request_id="DEMO-SCENARIO-E",
        drift_detected=True,
        adaptation_budget_size=200,
        diagnostic_slice=oracle_slice,
    )
    pkg_e, log_e = orchestrator.execute_repair(
        req_e,
        output_directory=out_dir / "scenario_e",
        pre_generated_candidates_df=pre_gen_pool,
    )
    print(f"Scenario E Outcome: Targeted Repair Blocked, Fallback Strategy={pkg_e.chosen_strategy}")
    # Must fallback to non-targeted strategy (e.g. SYNTHETIC_WEIGHTED_HYBRID)
    assert pkg_e.chosen_strategy != "TARGETED_SYNTHETIC_REPAIR"
    results["scenario_e"] = {
        "status": pkg_e.package_status.value,
        "fallback_strategy": pkg_e.chosen_strategy,
        "passed": True,
    }

    # -------------------------------------------------------------------------
    # Scenario F: Invalid Synthetic Records -> Pipeline Rejection
    # -------------------------------------------------------------------------
    print("\n--- Scenario F: Invalid Synthetic Records Rejection ---")
    val_pipe = UnifiedValidationPipeline()
    ref_df = pd.DataFrame({
        "tenure": [10, 20, 30],
        "MonthlyCharges": [50.0, 70.0, 90.0],
        "Churn": ["No", "Yes", "No"],
    })
    # Corrupt synthetic dataframe: missing target column
    bad_synth_df = pd.DataFrame({
        "tenure": [15, 25],
        "MonthlyCharges": [60.0, 80.0],
    })
    val_report_f = val_pipe.validate(bad_synth_df, ref_df, target_column="Churn")
    print(f"Scenario F Outcome: Decision={val_report_f.overall_decision.value}, PackageStatus={val_report_f.package_status}")
    assert val_report_f.overall_decision == AcceptanceDecision.REJECT
    results["scenario_f"] = {
        "overall_decision": val_report_f.overall_decision.value,
        "package_status": val_report_f.package_status,
        "passed": True,
    }

    # -------------------------------------------------------------------------
    # Scenario G: Resource Limits Constrained -> Safe Authentic Fallback
    # -------------------------------------------------------------------------
    print("\n--- Scenario G: Resource Constrained Adaptation ---")
    req_g = RepairRequest(
        request_id="DEMO-SCENARIO-G",
        drift_detected=True,
        adaptation_budget_size=200,
        max_memory_mb=512.0,  # Below 1024 MB threshold for deep generators
        max_runtime_seconds=20.0,
    )
    pkg_g, log_g = orchestrator.execute_repair(req_g, output_directory=out_dir / "scenario_g")
    print(f"Scenario G Outcome: Strategy={pkg_g.chosen_strategy} (TVAE restricted due to RAM/time)")
    assert pkg_g.chosen_strategy == "AUTHENTIC_WEIGHTED_ADAPTATION"
    results["scenario_g"] = {
        "status": pkg_g.package_status.value,
        "strategy": pkg_g.chosen_strategy,
        "passed": True,
    }

    # -------------------------------------------------------------------------
    # Scenario H: Accepted Candidate -> Versioned RepairPackage Emitted
    # -------------------------------------------------------------------------
    print("\n--- Scenario H: Accepted Candidate Repair Package ---")
    req_h = RepairRequest(
        request_id="DEMO-SCENARIO-H",
        drift_detected=True,
        adaptation_budget_size=200,
        source_retention_tolerance=0.05,
    )
    pkg_h, log_h = orchestrator.execute_repair(
        req_h,
        output_directory=out_dir / "scenario_h",
        pre_generated_candidates_df=pre_gen_pool,
    )
    manifest_file = Path(log_h.package_path) if log_h.package_path else None
    print(f"Scenario H Outcome: Package Manifest={manifest_file}")
    assert manifest_file is not None and manifest_file.is_file()
    assert pkg_h.package_status == RepairPackageStatus.APPROVED_FOR_REVIEW
    assert pkg_h.requires_human_approval is True
    results["scenario_h"] = {
        "package_id": pkg_h.package_id,
        "manifest_path": str(manifest_file),
        "status": pkg_h.package_status.value,
        "requires_human_approval": pkg_h.requires_human_approval,
        "passed": True,
    }

    print("\n" + "=" * 78)
    print("ALL 8 DEMONSTRATION WORKFLOWS COMPLETED SUCCESSFULLY.")
    print("=" * 78)

    # Save demonstration summary
    summary_path = out_dir / "demonstration_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_all_demonstrations()
