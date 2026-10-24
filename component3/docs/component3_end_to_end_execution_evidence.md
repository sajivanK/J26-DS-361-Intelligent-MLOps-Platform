# Component 3 — Real End-to-End Execution Evidence

**Component:** Component 3 — Adaptive Intelligent Synthetic Data Repair Engine  
**Project:** J26-DS-361 Intelligent MLOps Platform  
**Owner:** Mathusan K (IT23174580)  
**Execution Date:** 2026-10-21  
**Execution Mode:** CPU-only, bounded memory, real TVAE/CTGAN training (zero pre-generated candidate fallback)  
**Evidence Location:** `component3/results/final_integration_hardening/`  

---

## 1. Executive Summary

This document presents empirical execution logs and cryptographic evidence verifying the authentic end-to-end operation of Component 3. 

Every step in the 15-stage pipeline has been demonstrated to genuinely execute in CPU-bounded time:
- **Contract Validation:** Strict JSON schema and semantic bounds validation.
- **Dependency Resolution:** Fast verification of physical model and dataset files.
- **Prescription & Strategy Selection:** Deterministic rule-based strategy routing.
- **Generator Training & Sampling:** Live TVAE and CTGAN training on CPU with discrete column handling.
- **Multi-Gate Validation:** Structural checks, statistical fidelity (Wasserstein & Kolmogorov-Smirnov), and empirical privacy (5th percentile Distance-to-Closest-Record).
- **Downstream Utility Evaluation:** Retraining candidate models and evaluating F1 gain against authentic baseline controls.
- **Source Retention:** Guarding against catastrophic forgetting of undisturbed source distribution.
- **Handoff Packaging & Export:** Producing both rich internal `RepairPackage` artifacts and strict 12-field flat `c3_output_schema.json` payloads for Component 4.

---

## 2. Fifteen-Stage Pipeline Execution Trace

The table below details each stage executed during the authentic run of Scenario 1 (Run ID `RUN-6214C72F25`, Package ID `PKG-859B8F17C113`):

| Stage # | Stage Name | Action / Operation Performed | Measured Duration | Status | Evidence / Artifact Produced |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **1** | **Contract Validation** | Validated 3-section payload (`c1_evidence`, `c2_diagnosis`, `data_context`) against types, ranges, and allowed enums. | < 0.01s | `PASSED` | `external_contracts.validate_three_section_contract` |
| **2** | **Artifact Resolution** | Verified physical presence of dataset (`component3/data/raw/telco_churn.csv`), model (`telco_rf_experimental_source.joblib`), and preprocessor. | 0.02s | `PASSED` | Resolved file paths and hashes |
| **3** | **C1 Adaptation** | Mapped `predicted_harm` (0.18 -> 0.180), extracted 95% CI `[0.12, 0.24]`, derived `COVARIATE_SHIFT` and `HIGH` severity. | < 0.01s | `PASSED` | Internal `RepairRequest` (`REQ-FROM-C1-EV-70E7CDD4`) |
| **4** | **C2 Adaptation** | Parsed `affected_slice_rule` (`Contract == 'Month-to-month' and tenure <= 12`) via safe restricted AST parser. Ranked features preserved. | < 0.01s | `PASSED` | `DiagnosticSliceDefinition` (`SLICE-FROM-C2-DIAG-2FC16C60`) |
| **5** | **Prescription Engine** | Evaluated rule-based decision logic. Prescribed strategy: `TARGETED_SYNTHETIC_REPAIR`. Recommended generator: `tvae`. | 0.001s | `PASSED` | `PrescriptionDecision` (Sample weight policy: source=1.0, target=2.0, synth=2.0) |
| **6** | **Generator Selection** | `GeneratorSelector` checked compatibility against 20 columns (16 discrete, 3 numeric, 1 ID). Selected TVAE with discrete column transformer. | 0.01s | `PASSED` | Selected `tvae` with 10 epochs default budget |
| **7** | **Quantity Planning** | Initiated bounded grid search over candidate quantities $K \in \{25, 50, 100, 200\}$. | — | `ACTIVE` | `QuantityPlanner` grid search loop |
| **8** | **Generator Fitting & Sampling** | Executed live fitting of `TVAE` on target slice records. Generated $K=25$, $K=50$, $K=100$ records successively. | ~10.95s total | `COMPLETED` | Raw synthetic DataFrames generated on CPU |
| **9** | **Structural Validation** | Verified columns match schema exactly, no missing values introduced, data types conform, and numeric values within historical min/max. | 0.05s | `PASSED` | Structural integrity verified |
| **10** | **Fidelity Validation** | Evaluated two-sample Kolmogorov-Smirnov tests and categorical TVD. $K=25$ had marginal fidelity, $K=50$ failed (KS=0.409 > 0.350), $K=100$ passed (mean KS = 0.2764, fidelity score = 0.7236). | 0.15s | `PASSED (at K=100)` | Quantitative fidelity report |
| **11** | **Privacy Validation** | Evaluated 5th percentile Distance-to-Closest-Record (DCR) and verified 0 exact duplicate matches with training data. | 0.10s | `PASSED` | `privacy_status: "PASS"` |
| **12** | **Utility Evaluation** | Retrained Random Forest on combined source + authentic target + synthetic repair data. Evaluated on holdout target dev_val set. Target F1 = 0.6400 (exceeded baseline + 0.0100 margin). | 0.25s | `PASSED` | Candidate model utility verified |
| **13** | **Source Retention** | Evaluated retrained candidate model on undisturbed source evaluation set. Source F1 delta within tolerance ($-0.008 \le 0.050$). | 0.05s | `PASSED` | `source_retention_preserved: true` |
| **14** | **RepairPackage Assembly** | Assembled and serialized `RepairPackage` with package manifest, atomic candidate CSV, candidate model joblib, and SHA-256 hashes. | 0.08s | `PASSED` | Manifest, CSV (`859b8f17c113`), model (`pkg-859b8f17c113`) |
| **15** | **C4 Flat Export** | Formatted exact 12-field flat contract adhering to `shared/contracts/c3_output_schema.json` and saved to disk. | 0.01s | `PASSED` | `scenario_1_real_tvae/c4_contract_output.json` |

**Total End-to-End Pipeline Duration:** 11.531 seconds.

---

## 3. Real Execution Scenarios Summary

The automated execution verification script `component3/tests/verify_final_hardening_e2e.py` executed 4 diverse scenarios:

```json
{
  "timestamp_utc": "2026-10-21T04:32:11Z",
  "total_duration_seconds": 12.15,
  "summary": {
    "all_passed": true,
    "scenario_count": 4
  }
}
```

### Scenario Breakdown

### Scenario 1: Authentic TVAE Execution (`scenario_1_real_tvae`)
- **Input:** High drift severity (0.18 harm), covariate shift on tenure and Contract.
- **Generator Execution:** TVAE trained live on CPU in 10 epochs.
- **Search Outcome:** $K=25$ rejected on utility margin, $K=50$ rejected on KS fidelity, $K=100$ accepted on all gates.
- **Output Status:** `APPROVED`
- **Rows:** 100
- **Fidelity Score:** 0.7236
- **Utility Score:** 0.6400
- **Privacy Status:** `PASS`
- **Target Coverage:** 1.0000
- **Candidate CSV:** `component3/results/final_integration_hardening/scenario_1_real_tvae/pkg-859b8f17c113_candidates_k100.csv`
- **Candidate Model:** `component3/results/final_integration_hardening/scenario_1_real_tvae/pkg-859b8f17c113_candidate_model.joblib`

### Scenario 2: No-Drift Safe Abstention (`scenario_2_no_drift`)
- **Input:** Low drift severity (0.04 harm, below 0.10 threshold).
- **Prescription Decision:** `NO_REPAIR`
- **Output Status:** `REJECTED`
- **Rows:** 0
- **Fidelity / Utility / Coverage:** Honestly reported as `0.0` (zero fabrication).
- **Candidate URI:** `"NONE"`
- **Prescription Confidence:** 0.40

### Scenario 3: External Dependency Blocking (`scenario_3_dependency_blocking`)
- **Input:** Reference to nonexistent `shared/uci_adult/data/current.csv` and `champion-v12`.
- **Outcome:** Clean termination with `DependencyResolutionError`.
- **Silent Fallback:** Prevented.

### Scenario 4: Simulated Component 4 Consumer (`scenario_4_c4_simulated_consumer`)
- **Consumer Steps:** Loaded exported `c4_contract_output.json`, validated schema, resolved portable relative URI, loaded CSV into pandas.
- **Outcome:** 100 rows, 21 columns, target column `Churn` present, SHA-256 verified identical.
