# Component 3 Comprehensive Issue Register

**Project:** J26-DS-361 Intelligent MLOps Platform  
**Component:** Component 3 — Adaptive Intelligent Synthetic Data Repair Engine  
**Author:** Mathusan K (IT23174580)  
**Date:** October 2026  
**Status:** All P0, P1, P2 issues resolved inside Component 3. Remaining items are external team dependencies.

---

## 1. Issue Classification Taxonomy

- **P0 — Critical:** Security risks, arbitrary code execution, silent fallback, data leakage, artifact corruption.
- **P1 — High:** Contract mismatch, broken operational workflow, incorrect statistical validation, misleading research claims.
- **P2 — Medium:** State persistence bugs, error message clarity, and resource budget limits.
- **P3 — Low:** Cosmetic styling, documentation typos, code comments, and non-blocking warnings.

---

## 2. Issue Log

| Issue ID | Severity | Category | Description | Root Cause | Resolution / Fix Applied | Verification Evidence | Status |
|---|---|---|---|---|---|---|---|
| **ISSUE-01** | P0 | Security | Potential code injection via unvalidated `affected_slice_rule` query strings. | Using `df.query()` or string evaluation allows arbitrary code execution. | Implemented `safe_slice_evaluator.py` using restricted AST-only evaluation. Disallowed all calls, attributes, and imports. | Tested in `test_scenario_f_unsupported_or_malicious_slice`. Verified that `__import__`, `eval()`, and `lambda` raise `UnsafeSliceRuleError`. | **CLOSED** |
| **ISSUE-02** | P0 | Data Integrity | Potential silent fallback to Telco Churn when UCI Adult datasets or models are missing. | Loose error handling falling back to local defaults. | Implemented `resolve_data_context_artifacts()` and `DependencyResolutionError` in `external_contracts.py`. Fails fast with clear error message. | Tested in `test_scenario_d_missing_dependency`. | **CLOSED** |
| **ISSUE-03** | P1 | Contract Compatibility | Upstream orchestration contract format mismatch (`upstream["context"]`, `upstream["c1"]`, `upstream["c2_to_c3"]`). | `validate_three_section_contract` previously only accepted exact keys `c1_evidence`, `c2_diagnosis`, `data_context`. | Enhanced `validate_three_section_contract` in `external_contracts.py` to normalize `upstream` wrappers and aliases, and inherit pass-through fields (`model_id`, `reference_dataset_uri`, `current_dataset_uri`). | Tested in `test_upstream_orchestration_contract_format`. | **CLOSED** |
| **ISSUE-04** | P1 | Contract Schema | C4 contract schema violation for rejected and abstained packages. | When packages were rejected, fields emitted `None` or missing enum values, violating `c3_output_schema.json`. | Hardened `to_c4_flat_contract()` to always emit valid enum `REJECTED`, valid generator enum (`TVAE`), and schema-compliant default floats (`0.0`). | Validated with `validate_c4_flat_contract()` against `c3_output_schema.json`. | **CLOSED** |
| **ISSUE-05** | P1 | Validation Methodology | Statistical fidelity KS test applied to discrete/categorical features. | KS test assumes continuous distributions; discrete features cause artificial step discontinuities. | In `fidelity_validator.py`, separated continuous metrics (2-sample KS test, Wasserstein) from discrete metrics (Total Variation Distance, Jensen-Shannon divergence). | Tested in `test_fidelity_validator.py`. Verified in Section 8 of audit. | **CLOSED** |
| **ISSUE-08** | P2 | Research Traceability | Apparent discrepancy between Target F1 `0.6275 -> 0.6400` and `0.9231 -> 0.9615`. | Transcribing early offline global test benchmarks alongside later multi-seed target-slice evaluations without context. | Investigated source artifacts: `0.6275` traced to early offline CTGAN global test (`sprint05`), while `0.9231 -> 0.9615` is the authoritative multi-seed target-slice F1 from `RUN-6214C72F25`. Documented explicitly in reproducibility report. | Documented in `component3_research_reproducibility_audit.md`. | **CLOSED** |
| **ISSUE-09** | P3 | Infrastructure | Artifact paths previously used Windows-specific absolute paths (`D:\...`). | Hardcoded `str(path)` on Windows machines. | Added `format_relative_uri()` in `artifact_storage.py` and forward-slash normalization for all exported contract URIs. | Tested in `test_external_contracts.py`. | **CLOSED** |
| **ISSUE-10** | P1 | External Dependency | `shared/uci_adult/data/current.csv` does not exist on disk. | External teammates (C1/C2) have not yet generated or committed the drifted production batch. | Component 3 safely blocks execution with `DependencyResolutionError` (DEP-01). | Verified in Scenario 4. Action required from team leader. | **OPEN (EXTERNAL)** |
| **ISSUE-11** | P1 | External Dependency | Teammate contracts reference `champion-v12`, but disk contains `base_model_adult.pkl`. | Model alias naming mismatch across components. | Documented alias mapping in `component3_remaining_dependencies.md` (DEP-02). | Action required in platform deployment configuration. | **OPEN (EXTERNAL)** |
