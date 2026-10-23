# Component 3 End-to-End Execution Evidence and Verification Report

**Project:** J26-DS-361 Intelligent MLOps Platform  
**Component:** Component 3 — Adaptive Intelligent Synthetic Data Repair Engine  
**Author:** Mathusan K (IT23174580)  
**Date:** October 2026  
**Environment:** Python 3.11.0, pytest 9.1.1, Windows, CPU-Only Execution

---

## 1. Overview

This document presents physical execution evidence for the 8 core operational and research scenarios (A through H) tested in `component3/tests/test_system_audit_scenarios.py`.

All executions were performed on CPU using bounded budgets and verified against physical artifacts, SHA-256 digests, and JSON schemas.

---

## 2. Test Execution Summary

```powershell
.venv\Scripts\python.exe -m pytest component3/tests/test_system_audit_scenarios.py
```

### Result:
- **Collected:** 8 items
- **Passed:** **8 passed**
- **Failed:** **0 failed**
- **Duration:** 128.27s (CPU-only execution, bounded generator epochs)
- **Status:** **100% Pass Rate**

---

## 3. Scenario Execution Breakdown

### Scenario A — Successful End-to-End Repair
- **Dataset:** Telco Customer Churn (`telco_churn.csv`)
- **Model:** Verified Random Forest Classifier (`telco_rf_experimental_source.joblib`)
- **Input:** High covariate shift ($18\%$ predicted accuracy drop, $CI = [12\%, 24\%]$, slice: `Contract == 'Month-to-month' and tenure <= 24`).
- **Engine Execution:**
  - Prescription: Prescribed `TARGETED_SYNTHETIC_REPAIR` via `TVAE`.
  - Quantity Search: Evaluated $K \in [25, 50, 100, 200]$.
  - Selection: $K=100$ accepted as smallest passing candidate.
  - Multi-Gate Validation:
    - Structural: 0 type mismatches, 0 range violations.
    - Statistical Fidelity: Mean KS distance $0.2764 \le 0.350$, discrete TVD $0.2817 \le 0.350$.
    - Empirical Privacy: 5th percentile DCR $0.2237 > 0.0100$, 0 exact duplicates.
    - Utility Gain: Target slice F1 restored from $0.9231 \rightarrow 0.9615$ ($+3.84\text{ pp}$ gain, exceeding $+1.0\text{ pp}$ threshold).
    - Source Retention: Source eval F1 delta $-0.0080 \le 0.050$.
- **Generated Artifacts:**
  - `pkg-..._candidates_k100.csv` (100 rows, valid schema).
  - `pkg-..._candidate_model.joblib` (retrained model).
  - SHA-256 digest verified and recorded.
- **C4 Flat Contract:** Emitted status `APPROVED`, $n=100$, generator `TVAE`. Schema validation passed against `shared/contracts/c3_output_schema.json`.

---

### Scenario B — Safe Abstention (NO_REPAIR)
- **Input:** Drift severity score $0.04 < 0.10$ threshold.
- **Engine Execution:**
  - Gate 2 of `RepairPrescriptionEngine` evaluates drift severity.
  - Prescribes `NO_REPAIR` with reason: *"No significant drift detected (severity score < 0.10). Model repair is unnecessary."*
  - Quantity planned: $K=0$.
  - Generative models safely bypassed (0 training seconds consumed).
- **Output:** Package status `ABSTAINED_NO_REPAIR`. C4 flat contract exports status `REJECTED`, $n=0$, fidelity $0.0$, utility $0.0$.

---

### Scenario C — Candidate Rejection
- **Input:** Candidate pool with corrupted labels simulating negative utility gain.
- **Engine Execution:**
  - Retraining evaluator detects target slice F1 delta $\le 0.010$.
  - Rejection reason recorded: *"Retraining utility gain insufficient."*
  - Candidate generation halts.
- **Output:** Package status `REJECTED`. No candidate dataset released to C4.

---

### Scenario D — Missing Dependency (DEP-01)
- **Input:** Contract referencing uncommitted file `shared/uci_adult/data/nonexistent_batch.csv`.
- **Engine Execution:**
  - `resolve_data_context_artifacts()` checks physical file existence.
  - Raises `DependencyResolutionError` with message: *"Component 3 will not silently substitute another dataset."*
- **Output:** Safe failure. HTTP 400 emitted by API. Zero silent fallback.

---

### Scenario E — Malformed Contract
- **Input:** Payload with negative accuracy drop ($-5.0$) and missing `data_context`.
- **Engine Execution:**
  - `validate_three_section_contract()` collects all validation errors.
  - Reports:
    - `"predicted_accuracy_drop must be a non-negative number, got -5.0."`
    - `"Missing required top-level section: 'data_context' (or 'context')."`
- **Output:** Validation fails fast. HTTP 422 emitted by API.

---

### Scenario F — Unsupported / Malicious Slice Rule
- **Input:** Adversarial injection attempts:
  - `"__import__('os').system('echo hacked')"`
  - `"eval('1 + 1')"`
  - `"df.drop(columns=['age'])"`
- **Engine Execution:**
  - `safe_slice_evaluator.py` inspects AST node types.
  - Detects disallowed call and attribute access.
  - Raises `UnsafeSliceRuleError` without executing any arbitrary Python code.
- **Output:** 100% execution safety verified.

---

### Scenario G — API and Engine Consistency
- **Execution:** Invoked `/api/c3/health` and `/api/c3/scenarios` on FastAPI server.
- **Verification:** Responses validated against JSON schemas. Server accurately returns supported generators `["TVAE", "CTGAN", "TabDDPM"]` and verified operational status for `telco_churn`.

---

### Scenario H — Artifact Cryptographic Integrity
- **Execution:** Inspected candidate CSV emitted during Scenario A.
- **Verification:**
  - Physical file existence confirmed on disk.
  - Row count exactly matches `quantity_selected` ($100$).
  - SHA-256 computed on disk matches `candidate_dataset_sha256` recorded in the manifest.
