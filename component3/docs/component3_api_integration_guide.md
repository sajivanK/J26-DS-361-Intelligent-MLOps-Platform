# Component 3 API & CLI Integration Guide

**Research Component:** Component 3 — Adaptive Intelligent Synthetic Data Repair Engine  
**Author:** Senior MLOps Architect & API Integration Specialist  
**Target Audience:** Component 1 (Drift Detection), Component 2 (Root Cause Diagnosis), Component 4 (Model Retraining), and Platform Orchestration Engineers  
**Specification Version:** Contract-Ready v1.0 (October 2026)  

---

## 1. Overview and Architecture

Component 3 receives multi-component diagnostic evidence from Components 1 and 2, correlates it with platform data and model context, prescribes an optimal synthetic data repair strategy, samples synthetic records using trained generative models, strictly validates quality and privacy, evaluates downstream utility and source retention, and emits an agreed 12-field flat contract package for Component 4 retraining.

```
+------------------+    +------------------+    +-----------------------+
|   Component 1    |    |   Component 2    |    |  Platform Orchestrator|
|  (Drift Evidence)|    | (Root Cause Diag)|    |    (Data Context)     |
+--------+---------+    +--------+---------+    +-----------+-----------+
         |                       |                          |
         +-----------------------+--------------------------+
                                 |
                                 v
          +----------------------------------------------+
          | Component 3 Canonical 3-Section JSON Payload |
          |      { c1_evidence, c2_diagnosis,            |
          |        data_context }                        |
          +----------------------+-----------------------+
                                 |
                                 v
          +----------------------------------------------+
          | Strict Schema & Type Validation              |
          | validate_three_section_contract()            |
          +----------------------+-----------------------+
                                 |
                                 v
          +----------------------------------------------+
          | Dependency & Artifact Resolution             |
          | resolve_data_context_artifacts()             |
          +----------------------+-----------------------+
                                 |
                                 v
          +----------------------------------------------+
          | Boundary Adapters -> Internal RepairRequest  |
          | C1DriftAdapter & C2DiagnosisAdapter          |
          +----------------------+-----------------------+
                                 |
                                 v
          +----------------------------------------------+
          | Intelligent Repair Orchestrator              |
          | 1. Sequential Governance Checks (10 Gates)   |
          | 2. Prescription Engine (Strategy/Gen/Qty)    |
          | 3. Generative Modeling (CTGAN/TVAE/TabDDPM)  |
          | 4. Multi-Stage Validation (Fidelity/Privacy) |
          | 5. Offline Utility & Source Retention Check  |
          +----------------------+-----------------------+
                                 |
                                 v
          +----------------------------------------------+
          | C4 Retraining Adapter                        |
          | export_c4_flat_contract()                    |
          +----------------------+-----------------------+
                                 |
                                 v
          +----------------------------------------------+
          | Exact 12-Field C4 JSON Contract              |
          | shared/contracts/c3_output_schema.json       |
          +----------------------------------------------+
```

---

## 2. Canonical Input Specification

Component 3's public interface accepts a JSON document containing three mandatory top-level sections:
1. `c1_evidence`
2. `c2_diagnosis`
3. `data_context`

### Reference Payload Example

```json
{
  "c1_evidence": {
    "predicted_accuracy_drop": 14.2,
    "confidence_interval": [10.9, 17.5],
    "drift_type": "interaction",
    "drift_severity": "high",
    "affected_features": [
      "education_num",
      "occupation"
    ]
  },
  "c2_diagnosis": {
    "root_cause_feature": "education_num",
    "root_cause_category": "interaction",
    "affected_slice_rule": "education_num > 12 AND occupation == 'Tech-support'",
    "affected_features_ranked": [
      "education_num",
      "occupation",
      "age"
    ],
    "affected_value_ranges": {
      "education_num": [12, 16],
      "occupation": [
        "Tech-support",
        "Prof-specialty"
      ]
    },
    "slice_support": 0.18,
    "diagnosis_stability": 0.85,
    "drift_impact_score": 0.72,
    "temporal_window": 3
  },
  "data_context": {
    "model_id": "champion-v12",
    "dataset_id": "adult-v1",
    "reference_dataset_uri": "shared/uci_adult/data/reference.csv",
    "current_dataset_uri": "shared/uci_adult/data/current.csv",
    "target_column": "income",
    "schema_version": "adult-schema-v1",
    "label_status": "AVAILABLE",
    "preprocessing_uri": "shared/uci_adult/preprocessing.py",
    "adaptation_budget_size": 200,
    "source_retention_tolerance": 0.05
  }
}
```

### Field Definitions and Constraints

#### `c1_evidence` (from Component 1)
| Field | Type | Valid Values / Constraints | Description |
|---|---|---|---|
| `predicted_accuracy_drop` | Float | $\ge 0.0$. Values $> 1.0$ treated as percentage points (e.g. $14.2\% \to 0.142$). | Estimated downstream model accuracy degradation. |
| `confidence_interval` | [Float, Float] | Monotonic pair $[low, high]$ where $low \le high$. | 95% confidence interval for estimated accuracy drop. |
| `drift_type` | String | `"interaction"`, `"marginal"`, `"combined"`, `"covariate_shift"`, `"concept_drift"`, `"subpopulation_shift"`, `"none"`. | Drift classification detected by C1. |
| `drift_severity` | String | `"none"`, `"low"`, `"moderate"`, `"medium"`, `"high"`, `"critical"`. | Drift severity level. |
| `affected_features` | List[String] | Non-empty feature names. | Features exhibiting statistical divergence. |

#### `c2_diagnosis` (from Component 2)
| Field | Type | Valid Values / Constraints | Description |
|---|---|---|---|
| `root_cause_feature` | String | Valid column name. | Primary feature identified as drift driver. |
| `root_cause_category` | String | `"interaction"`, `"marginal"`, `"combined"`, `"covariate_shift"`, `"concept_shift"`. | Diagnostic category of root cause. |
| `affected_slice_rule` | String | Boolean expression, e.g. `col > 10 AND col2 == 'X'`. | Subpopulation condition exhibiting severe degradation. Evaluated via restricted AST. |
| `affected_features_ranked` | List[String] | Ordered list of feature names. | Features ranked by drift attribution score. |
| `affected_value_ranges` | Dict | Keys: column names, Values: [min, max] or list of categories. | Empirical boundaries of problematic slice. |
| `slice_support` | Float | $[0.0, 1.0]$ | Proportion of data matching the slice rule. |
| `diagnosis_stability` | Float | $[0.0, 1.0]$ | Temporal bootstrap stability of diagnosis. |
| `drift_impact_score` | Float | $[0.0, 1.0]$ | Estimated degradation attributable to slice. |
| `temporal_window` | Integer | $\ge 1$ | Diagnosis observation window index. |

#### `data_context` (from Platform Orchestrator)
| Field | Type | Valid Values / Constraints | Description |
|---|---|---|---|
| `model_id` | String | Valid identifier registered in `ModelRegistry` or path to `.joblib`. | Current champion model to repair. |
| `dataset_id` | String | Registered dataset ID (e.g. `telco_churn`, `adult-v1`). | Target dataset identifier. |
| `reference_dataset_uri` | String | Path to reference baseline CSV. | Baseline historical distribution. |
| `current_dataset_uri` | String | Path to current drifted batch CSV. | Current operational batch to adapt. |
| `target_column` | String | Column name in dataset. | Classification target column. |
| `schema_version` | String | Version identifier. | Data schema version. |
| `label_status` | String | `"AVAILABLE"`, `"UNAVAILABLE"`. | Ground truth label availability in current batch. |
| `adaptation_budget_size` | Integer | $\ge 1$ (default: 200). | Maximum authentic samples available for adaptation. |
| `source_retention_tolerance`| Float | $[0.0, 1.0]$ (default: 0.05). | Maximum tolerable source reference performance degradation. |

---

## 3. Public Python Programming Interface

Component 3 exposes clean, modular entry points in `component3.src.api`:

### 1. Validate Payload Only
```python
from component3.src.api import validate_external_contract

payload = { ... } # 3-section payload
is_valid, error_list = validate_external_contract(payload)
if not is_valid:
    print(f"Validation failed: {error_list}")
```

### 2. Execute Complete Repair Workflow
```python
from pathlib import Path
from component3.src.api import execute_contract_repair

c4_dict, repair_package, logs = execute_contract_repair(
    request=payload,
    output_c4_path=Path("results/c4_output.json"),
    output_directory=Path("results/run_artifacts"),
    decision_mode="RULE_BASED",  # or "LEARNED", "SHADOW_MODE"
)

print(f"Repair Status: {c4_dict['repair_package_status']}")
print(f"Synthetic Rows: {c4_dict['n_synthetic_rows']}")
print(f"Fidelity Score: {c4_dict['fidelity_score']}")
print(f"Utility Score: {c4_dict['utility_score']}")
```

### 3. Export Existing RepairPackage to C4 Flat Contract
```python
from component3.src.api import export_c4_flat_contract

c4_output_dict = export_c4_flat_contract(
    repair_package,
    output_path=Path("handoff/c4_repair_package.json"),
    repair_version="c3-001"
)
```

---

## 4. Command-Line Interface (CLI)

Component 3 includes a dedicated CLI command `contract-repair` for batch pipelines and orchestrator invocations:

### Command Syntax
```powershell
python -m component3.src.cli contract-repair --input <path_to_request.json> [--output-c4 <path_to_c4.json>] [--decision-mode RULE_BASED]
```

### Example Invocations

#### Full Repair Execution with Handoff Export
```powershell
.venv\Scripts\python.exe -m component3.src.cli contract-repair `
    --input component3/results/contract_integration_verification/scenario_02_real_data_execution/input_payload.json `
    --output-c4 component3/results/contract_integration_verification/scenario_02_real_data_execution/c4_output_package.json
```

#### Exit Codes
- `0`: Successful execution (Package evaluated, validated, and exported).
- `1`: Validation error, missing required artifact (`DependencyResolutionError`), or execution exception.

---

## 5. Output Specification (Delivered to Component 4)

Component 3 delivers a flat 12-field JSON object conforming strictly to `shared/contracts/c3_output_schema.json`:

```json
{
  "repair_dataset_uri": "component3/results/runs/RUN-DD7E609F84/synthetic_candidates.csv",
  "repair_package_status": "APPROVED",
  "generator": "CTGAN",
  "n_synthetic_rows": 200,
  "augmentation_ratio": 0.25,
  "fidelity_score": 0.9124,
  "utility_score": 0.8650,
  "privacy_status": "PASS",
  "target_coverage": 0.9400,
  "prescription_confidence": 0.8800,
  "repair_version": "c3-001",
  "repair_timestamp": "2026-10-21T02:24:50Z"
}
```

### Field Descriptions and Semantics
1. `repair_dataset_uri` (string): Relative or absolute URI to the generated synthetic/augmented repair CSV.
2. `repair_package_status` (string, enum `["APPROVED", "REJECTED"]`): Overall validation decision. `APPROVED` signifies that all structural, fidelity, privacy, utility, and retention gates passed.
3. `generator` (string, enum `["CTGAN", "TVAE", "TabDDPM", "NONE"]`): Primary generative algorithm selected by the prescription engine.
4. `n_synthetic_rows` (integer, $\ge 0$): Actual count of synthetic records produced.
5. `augmentation_ratio` (float, $[0.0, 1.0]$): Ratio of synthetic records to baseline training size.
6. `fidelity_score` (float, $[0.0, 1.0]$): Normalized statistical fidelity score ($1 - \text{mean KS distance}$).
7. `utility_score` (float, $[0.0, 1.0]$): Empirical downstream classification F1 score achieved by candidate model on target test slice.
8. `privacy_status` (string, enum `["PASS", "FAIL"]`): Verification that zero exact row memorization occurred and distance-to-closest-record (DCR) threshold was satisfied.
9. `target_coverage` (float, $[0.0, 1.0]$): Proportion of diagnostic slice feature distribution covered by synthetic samples.
10. `prescription_confidence` (float, $[0.0, 1.0]$): Normalized confidence metric computed from diagnosis stability, slice support, and evidence strength.
11. `repair_version` (string): Component 3 pipeline release tag.
12. `repair_timestamp` (string, ISO 8601 UTC): Generation timestamp.

---

## 6. Error Handling and Diagnostics

When incoming payloads or dependencies are invalid, Component 3 fails fast with structured error messages:

| Exception Type | Trigger Condition | Example Error Response |
|---|---|---|
| `ContractValidationError` | Missing top-level section, missing required field, out-of-range value, or unsupported enum. | `ContractValidationError: c1_evidence missing required field: 'predicted_accuracy_drop'` |
| `DependencyResolutionError` | `reference_dataset_uri`, `current_dataset_uri`, or `model_id` does not exist on disk. | `DependencyResolutionError: Current drifted dataset not found at URI 'shared/uci_adult/data/current.csv'` |
| `UnsafeSliceRuleError` | `affected_slice_rule` contains function calls, attribute traversal, syntax errors, or unknown column names. | `UnsafeSliceRuleError: Function calls are prohibited in slice rules: '__import__'` |
| `Abstention / Rejection` | Drift severity $< 0.10$, authentic labels missing, fidelity gate failure, or negative utility delta. | Returns JSON package with `repair_package_status: "REJECTED"`, `n_synthetic_rows: 0`, and detailed reasons in execution log. |
