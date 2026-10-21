# Component 3 — Teammate & Team Leader Integration Runbook

**Component:** Component 3 — Adaptive Intelligent Synthetic Data Repair Engine  
**Project:** J26-DS-361 Intelligent MLOps Platform  
**Owner:** Mathusan K (IT23174580)  
**Readiness Level:** Level 2 (Contract-Ready for Integration Testing)  
**Target Audience:** Team Leader (Sajivan K), Component 1/2 Engineers, Component 4 Engineer (Kesigan M), Platform Orchestrator Engineers  

---

## 1. Quick Start

Component 3 is fully integrated with the multi-component contract interfaces and ready for integration testing.

### Prerequisites
- Python 3.11 environment with repository dependencies installed (`.venv`).
- Working directory set to the repository root.

```bash
# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Optional: Run quick automated test verification (758 tests)
pytest component3/tests/test_hardening_failures.py
```

---

## 2. CLI Execution for Integration Testing

Component 3 provides a unified CLI command `contract-repair` that takes an agreed 3-section contract JSON file, validates inputs, executes the repair engine, and exports the authoritative 12-field Component 4 output.

```bash
# Execute contract repair via CLI
python -m component3.src.cli contract-repair \
  --input component3/results/final_integration_hardening/example_contract_payload.json \
  --output component3/results/final_integration_hardening/cli_c4_output.json \
  --out-dir component3/results/final_integration_hardening/cli_run/
```

### Expected CLI Output:
```text
Loaded 3-section contract payload from component3/results/final_integration_hardening/example_contract_payload.json
Validation passed: model='telco_rf_experimental_source', dataset='telco_churn'
Execution completed successfully!
Run ID: RUN-XXXXXXXXXX
Package ID: PKG-XXXXXXXXXXXX
Package Status: APPROVED_FOR_REVIEW
Chosen Strategy: TARGETED_SYNTHETIC_REPAIR
Selected Quantity: 100
Saved flat C4 contract JSON to: component3/results/final_integration_hardening/cli_c4_output.json
```

---

## 3. Python API Integration for Platform Orchestrator

For programmatic invocation by the platform orchestrator or upstream pipelines:

```python
from pathlib import Path
from component3.src.api import execute_contract_repair

# 1. Prepare or load 3-section payload
payload_path = Path("component3/results/final_integration_hardening/example_contract_payload.json")

# 2. Execute end-to-end repair
c4_contract, repair_pkg, exec_log = execute_contract_repair(
    request=payload_path,
    output_c4_path="component3/results/runs/c4_handoff.json",
    output_directory="component3/results/runs/run_001/",
    decision_mode="RULE_BASED",
)

# 3. Inspect C4 contract fields
print("Status:", c4_contract["repair_package_status"])  # "APPROVED" or "REJECTED"
print("Repair Dataset URI:", c4_contract["repair_dataset_uri"])
print("Fidelity Score:", c4_contract["fidelity_score"])
print("Utility Score:", c4_contract["utility_score"])
print("Privacy Status:", c4_contract["privacy_status"])
```

---

## 4. Input Contract Specification

Component 3 expects a dictionary or JSON file with three top-level keys:

```json
{
  "c1_evidence": {
    "predicted_harm": 0.18,
    "confidence_interval": [0.12, 0.24],
    "drift_type": "covariate_shift",
    "drift_severity": "HIGH",
    "affected_features": ["Contract", "MonthlyCharges", "tenure"]
  },
  "c2_diagnosis": {
    "root_cause_feature": "Contract",
    "root_cause_category": "covariate_shift",
    "affected_slice_rule": "Contract == 'Month-to-month' and tenure <= 12",
    "affected_features_ranked": ["Contract", "MonthlyCharges", "tenure"],
    "affected_value_ranges": {
      "Contract": ["Month-to-month"],
      "tenure": [0, 12]
    },
    "slice_support": 0.28,
    "diagnosis_stability": 0.85,
    "drift_impact_score": 0.15,
    "temporal_window": 30
  },
  "data_context": {
    "dataset_id": "telco_churn",
    "model_id": "telco_rf_experimental_source",
    "reference_dataset_uri": "component3/data/raw/telco_churn.csv",
    "current_dataset_uri": "component3/data/raw/telco_churn.csv",
    "target_column": "Churn",
    "schema_version": "1.0",
    "label_status": "AVAILABLE",
    "preprocessing_uri": "component3/results/trained_models/telco_preprocessor_experimental_source.joblib"
  }
}
```

### Allowed Values & Constraints:
- `c1_evidence.drift_type`: One of `['combined', 'concept_drift', 'covariate_shift', 'interaction', 'marginal', 'marginal_multi', 'marginal_single', 'no_drift', 'none', 'subpopulation_shift']`.
- `c1_evidence.drift_severity`: One of `['LOW', 'MODERATE', 'HIGH', 'CRITICAL']`.
- `c2_diagnosis.root_cause_category`: One of `['combined', 'concept_shift', 'covariate_shift', 'data_quality', 'external_factor', 'interaction', 'marginal', 'sampling_bias', 'unknown']`.
- `c2_diagnosis.affected_slice_rule`: Python-compatible boolean expression evaluated safely via restricted AST (e.g. `"feature > 10 and category == 'X'"`).

---

## 5. Output Contract Specification (for Component 4)

Component 3 exports the authoritative 12-field flat JSON adhering to `shared/contracts/c3_output_schema.json`:

```json
{
  "repair_dataset_uri": "component3/results/runs/scenario_1_real_tvae/pkg-859b8f17c113_candidates_k100.csv",
  "repair_package_status": "APPROVED",
  "generator": "TVAE",
  "n_synthetic_rows": 100,
  "augmentation_ratio": 0.0267,
  "fidelity_score": 0.7236,
  "utility_score": 0.6400,
  "privacy_status": "PASS",
  "target_coverage": 1.0,
  "prescription_confidence": 0.8,
  "repair_version": "c3-70e7cdd4",
  "repair_timestamp": "2026-10-21T04:32:10.644749+00:00"
}
```

---

## 6. Configurable Artifact Storage

Component 3 automatically writes artifacts using portable, forward-slash relative URIs.

To configure a custom shared storage directory for team staging:
```bash
# Set environment variable (PowerShell)
$env:C3_ARTIFACT_STORAGE_ROOT = "shared/data/repairs"

# Or Linux/macOS
export C3_ARTIFACT_STORAGE_ROOT="shared/data/repairs"
```
When set, all candidate CSVs, model joblib files, and JSON manifests are written into subdirectories under this root. If unset, storage defaults to `component3/results/runs/`.

---

## 7. Troubleshooting & Common Errors

| Error / Exception | Cause | Resolution |
| :--- | :--- | :--- |
| `DependencyResolutionError: Reference dataset not found...` | The file referenced by `reference_dataset_uri` does not exist on disk. | Verify URI is relative to repository root and file exists (e.g. `component3/data/raw/telco_churn.csv`). |
| `DependencyResolutionError: Model 'champion-v12' is neither registered...` | Referenced model is missing. Component 3 does not silently substitute models. | Supply the actual trained model artifact or use registered model `telco_rf_experimental_source`. |
| `UnsafeSliceRuleError` | `affected_slice_rule` contains syntax errors or unallowed expressions (e.g. function calls, `eval`, `__import__`). | Provide valid boolean comparisons using allowlisted binary operators (`>`, `<`, `==`, `!=`, `in`, `and`, `or`). |
| `ContractValidationError` | Missing required fields, out-of-range floats, or unrecognized enums. | Validate payload using `validate_external_contract(payload)` before running repair. |
