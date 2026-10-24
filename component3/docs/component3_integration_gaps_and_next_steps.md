# Component 3 — Integration Gaps and Next Steps Report

## Collaborative Research Project: J26-DS-361 Intelligent MLOps Platform
**Component 3:** Adaptive Intelligent Synthetic Data Repair Engine  
**Branch:** `c3-mathusan` | **Committed Revision:** `fb74cdcf0ca4ea47a3cc5e1dedf7d3c1eadd2c45`  
**Audit Date:** October 2026 | **Auditor Role:** Senior MLOps Architect & Quality Assurance Auditor  

---

## 1. Executive Summary

This report catalogues all empirical integration gaps identified during the comprehensive integration readiness audit of Component 3.

**Key Finding:** Component 3's internal data repair engine is scientifically robust and functionally complete (732/732 regression tests pass, all 10 scenario tests pass). However, **Component 3 is currently at LEVEL 1 (INTERNAL ENGINE WORKING)** because its peripheral adapters (`c1_drift_adapter.py`, `c2_diagnosis_adapter.py`, `c4_retraining_adapter.py`) do not align with the frozen contracts in `shared/contracts/`.

Resolving these gaps requires **zero changes to the core repair algorithms**, only lightweight updates to the adapter layer to translate contract keys accurately.

---

## 2. Comprehensive Discovered Gap Inventory

### Gap 1 (CRITICAL): C2 Diagnosis Slice Rule Key Name Mismatch
- **Severity:** **CRITICAL** (Blocks live integration with Component 2).
- **Exact File & Location:** `component3/src/adapters/c2_diagnosis_adapter.py`, lines 147–158.
- **Actual Problem:** The frozen contract `shared/contracts/c2_output_to_c3_schema.json` defines the slice filter condition under the key `"affected_slice_rule"`. Component 3's `C2DiagnosisAdapter` checks for `"diagnosed_problematic_slice"`, `"slice_predicate"`, or `"subpopulation_filter"`. When passed a real C2 payload, `c2_adapter.adapt(raw_c2)` throws:
  ```python
  ValueError: Diagnostic slice_predicate must be a non-empty string.
  ```
- **Reproduction Steps:**
  ```python
  from component3.src.adapters.c2_diagnosis_adapter import C2DiagnosisAdapter
  adapter = C2DiagnosisAdapter()
  c2_payload = {
      "diagnosis_id": "DIAG-01",
      "affected_slice_rule": "tenure < 12",
      "affected_features_ranked": ["tenure"],
      "diagnosis_confidence": 0.9,
  }
  adapter.adapt(c2_payload)  # Raises ValueError
  ```
- **Evidence:** Confirmed in `component3/results/integration_readiness_audit/contract_compatibility_audit.json`.
- **Impact:** Any real C2 diagnosis sent directly to C3 fails immediately with a runtime exception.
- **Proposed Solution:** In `C2DiagnosisAdapter.adapt()`, add `raw.get("affected_slice_rule")` as the primary key when resolving the slice predicate string.
- **Code Change Needed:** Yes (`c2_diagnosis_adapter.py`).
- **Teammate Input Needed:** No (the teammate's format is already frozen in `shared/contracts/c2_output_to_c3_schema.json`).
- **Blocks Integration:** **YES**.

---

### Gap 2 (HIGH): C1 Drift Severity and Detection Status Mismatch
- **Severity:** **HIGH** (Causes false abstention or degraded severity mapping).
- **Exact File & Location:** `component3/src/adapters/c1_drift_adapter.py`, lines 122–126.
- **Actual Problem:**
  1. The frozen contract `shared/contracts/c1_output_schema.json` does not provide an explicit `drift_detected` boolean; it outputs `drift_type` (`"combined"`, `"marginal_single"`, etc.) and `drift_severity`. C3's adapter defaults `drift_detected = False` when the field is omitted, causing C3 to abstain from repairing valid drift.
  2. C1 outputs its continuous impact score as `predicted_harm` (float in `[0, 1]`), while C3 expects `severity_score`.
- **Reproduction Steps:**
  ```python
  from component3.src.adapters.c1_drift_adapter import C1DriftAdapter
  adapter = C1DriftAdapter()
  c1_payload = {
      "detection_id": "DET-01",
      "drift_type": "combined",
      "drift_severity": "high",
      "predicted_harm": 0.85,
      "features_drifted": ["tenure"],
  }
  event = adapter.adapt(c1_payload)
  print(event.drift_detected)      # Outputs False!
  print(event.drift_severity_score) # Outputs 0.0!
  ```
- **Evidence:** Confirmed in `component3/results/integration_readiness_audit/contract_compatibility_audit.json`.
- **Impact:** Component 3 treats valid C1 drift alerts as "no drift", producing an `ABSTAINED_NO_REPAIR` package.
- **Proposed Solution:** In `C1DriftAdapter.adapt()`:
  - Derive `drift_detected`: `raw.get("drift_detected", raw.get("drift_type", "none").lower() != "none" and raw.get("drift_severity", "none").lower() != "none")`.
  - Extract severity score: `float(raw.get("predicted_harm", raw.get("severity_score", 0.0)))`.
  - Map `drift_type` enums: add mappings for `combined` $\to$ `COVARIATE`, `interaction` $\to$ `CONCEPT`, `none` $\to$ `NONE`.
- **Code Change Needed:** Yes (`c1_drift_adapter.py`).
- **Teammate Input Needed:** No (frozen schema available).
- **Blocks Integration:** **YES**.

---

### Gap 3 (CRITICAL): C3 $\to$ C4 Package Structure Incompatibility
- **Severity:** **CRITICAL** (Component 4 cannot consume current C3 output).
- **Exact File & Location:** `component3/src/adapters/c4_retraining_adapter.py`, lines 95–130.
- **Actual Problem:** The team has standardized `shared/contracts/c3_output_schema.json` on a **flat 12-field schema**:
  - `repair_dataset_uri`
  - `repair_package_status`
  - `generator`
  - `n_synthetic_rows`
  - `augmentation_ratio`
  - `fidelity_score`
  - `utility_score`
  - `privacy_status`
  - `target_coverage`
  - `prescription_confidence`
  - `repair_version`
  - `repair_timestamp`
  However, `C4RetrainingAdapter.to_c4_review_payload()` produces a nested structure with headers `metadata`, `prescription`, `validation`, `evaluation`, and `artifacts`. It contains **0 of the 12 required flat top-level fields**.
- **Reproduction Steps:**
  ```python
  import jsonschema, json
  from pathlib import Path
  from component3.src.adapters.c4_retraining_adapter import C4RetrainingAdapter
  schema = json.loads(Path("shared/contracts/c3_output_schema.json").read_text())
  # Any package adapted with to_c4_review_payload fails validation
  ```
- **Evidence:** Confirmed in `integration_scenarios_execution_report.json` (Scenario 10).
- **Impact:** Component 4's schema validator will reject the output package from Component 3.
- **Proposed Solution:** In `C4RetrainingAdapter`, implement `to_c4_flat_contract(package: RepairPackage) -> dict[str, Any]` that flattens the required 12 fields exactly as specified in `shared/contracts/c3_output_schema.json`.
- **Code Change Needed:** Yes (`c4_retraining_adapter.py`).
- **Teammate Input Needed:** No (C4's contract is fully defined).
- **Blocks Integration:** **YES**.

---

### Gap 4 (MEDIUM): Platform Layer Context Injection Specification
- **Severity:** **MEDIUM** (Operational gap, not a code defect).
- **Exact File & Location:** `component3/src/api.py`, `component3/src/orchestration/repair_orchestrator.py`.
- **Actual Problem:** Neither C1 nor C2 supplies dataset CSVs, baseline model binaries, or target column names. If teammates attempt to pass C1/C2 JSON directly into `execute_repair()`, execution fails because data files are missing.
- **Impact:** Teammates might assume Component 3 is self-sufficient from C1/C2 outputs alone.
- **Proposed Solution:** Formally publish the `data_context` specification (as documented in `component3_integration_input_output_guide.md`) so the team's shared orchestrator knows to inject dataset URIs and model references.
- **Code Change Needed:** No code change in C3; platform orchestration alignment needed.
- **Teammate Input Needed:** Yes (Platform / Orchestration team member).
- **Blocks Integration:** **YES (for live end-to-end execution)**.

---

### Gap 5 (LOW): Absence of HTTP REST API Server
- **Severity:** **LOW** (Architectural design choice).
- **Exact File & Location:** `component3/src/api.py` and `component3/src/cli.py`.
- **Actual Problem:** Component 3 provides Python programmatic APIs (`execute_repair()`, etc.) and a CLI interface (`python -m component3.src.cli`), but no HTTP server (FastAPI/Flask).
- **Impact:** Teammates running on separate machines cannot make REST HTTP calls unless an API server or microservice wrapper is added.
- **Proposed Solution:** If the team decides on microservices rather than a shared Python package, write a lightweight FastAPI wrapper exposing `POST /repair/request` and `GET /repair/status/{id}`.
- **Code Change Needed:** Optional / Future phase.
- **Teammate Input Needed:** Yes (confirming whether integration is via shared Python environment or REST microservices).
- **Blocks Integration:** **NO (if integrated via Python module or CLI)**.

---

## 3. Recommended Action Plan & Next Steps

| Step | Action Item | Target File | Effort | Blocker Status |
|---|---|---|---|---|
| **Step 1** | Update `C2DiagnosisAdapter` to parse `"affected_slice_rule"` and list-of-strings features | `component3/src/adapters/c2_diagnosis_adapter.py` | 1 hour | Blocks C2 integration |
| **Step 2** | Update `C1DriftAdapter` to derive `drift_detected` and map `predicted_harm` | `component3/src/adapters/c1_drift_adapter.py` | 1 hour | Blocks C1 integration |
| **Step 3** | Implement `to_c4_flat_contract()` in `C4RetrainingAdapter` matching `c3_output_schema.json` | `component3/src/adapters/c4_retraining_adapter.py` | 2 hours | Blocks C4 integration |
| **Step 4** | Add unit tests validating adapters directly against schemas in `shared/contracts/` | `component3/tests/test_shared_contract_adapters.py` | 1.5 hours | Safety assurance |
| **Step 5** | Align with platform team on `data_context` injection in orchestration runner | Shared pipeline scripts | Meeting / Alignment | Blocks live E2E |

Once Steps 1–4 are executed and verified, Component 3 will immediately advance from **LEVEL 1 (INTERNAL ENGINE WORKING)** to **LEVEL 2 (CONTRACT-READY FOR INTEGRATION TESTING)**.
