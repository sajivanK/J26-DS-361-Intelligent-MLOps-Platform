# Sprint 05 — Task 06: Repair Acceptance and Rejection Policy

## 1. Project Information

Project: Intelligent MLOps Platform  
Group: J26-DS-361  
Component: Component 3 — Adaptive Intelligent Synthetic Data Repair Engine  
Sprint: 05 — Validation Engine and Repair Experiments  
Task: 06 — Implement Repair Acceptance and Rejection Decisions  

Date: October 18, 2026  

## 2. Objective

Design, implement, and verify a deterministic, multi-stage repair acceptance policy
that integrates structural validity, statistical fidelity, privacy risk, and offline
predictive utility into structured release decisions: `ACCEPT`, `REVIEW`, and `REJECT`.

The policy strictly protects downstream deployment by preventing unverified or harmful
synthetic augmentations from advancing to Component 4.

## 3. Implemented Files

- `component3/src/validation/acceptance_policy.py`
- `component3/src/validation/validation_models.py` (`AcceptanceDecision`, `RepairPackageDecision`)
- `component3/src/validation/__init__.py`
- `component3/tests/test_acceptance_policy.py`

## 4. Reused Components and Standards

- `component3/src/validation/validation_models.py` (`ValidationStatus`, `ValidationStageResult`)
- `shared/contracts/c3_output_schema.json` (`repair_package_status`: `"APPROVED"` | `"REJECTED"`, `privacy_status`: `"PASS"` | `"FAIL"`)

## 5. Policy Rules and Architecture

### 5.1 Stage Decision Hierarchy

| Stage | Failure Consequence | Policy Rationale |
|---|---|---|
| **Structural & Coverage** | Immediate `REJECT` | Column mismatch, missing values, or insufficient slice coverage cannot be repaired. |
| **Privacy (DCR / Exact)** | Immediate `REJECT` | Component 4 hard gate forbids any privacy failure regardless of other scores. |
| **Fidelity (KS / JS)** | `REJECT` in production; `REVIEW` in exploratory | Distinguishes statistical underfitting from fatal structural flaws. |
| **Utility (Retraining)** | `REJECT` if negative transfer; `REVIEW` if unevaluated | Prevents model performance degradation on non-affected populations. |

### 5.2 Mandatory Evidence Preservation
- **Never Collapse `NOT_EVALUATED` into `PASS`**: If ground truth labels are unavailable or offline utility is skipped, the overall status is marked `REVIEW` with `supervised_retraining_approved = False`.
- **Fail Closed**: Missing structural evidence, malformed payloads, or unhandled exceptions fail closed to `REJECT`.
- **Contract Alignment**: Only `AcceptanceDecision.ACCEPT` maps to `package_status = "APPROVED"`. All other decisions (`REVIEW`, `REJECT`) output `"REJECTED"`.

## 6. Unit Testing Evidence

Test File: `component3/tests/test_acceptance_policy.py`  
Command: `pytest component3/tests/test_acceptance_policy.py -v`  
Result: 10 passed in 5.36 seconds (100% pass rate).

Test Scenarios Evaluated:
1. `test_all_stages_pass_accepts`: All four stages pass $\rightarrow$ `ACCEPT`, `package_status="APPROVED"`, `supervised_retraining_approved=True`.
2. `test_structural_fail_rejects`: Structural failure immediately forces `REJECT`.
3. `test_privacy_fail_rejects`: Privacy failure immediately triggers C4 hard gate `REJECT`.
4. `test_fidelity_fail_rejects_in_production`: Statistical divergence triggers `REJECT` in production mode.
5. `test_fidelity_fail_reviews_in_exploratory`: Exploratory mode allows `REVIEW` for divergence.
6. `test_utility_fail_rejects_in_production`: Negative transfer triggers `REJECT`.
7. `test_utility_not_evaluated_results_in_review`: Missing label authorization produces `REVIEW` without retraining approval.
8. `test_missing_stages_prevent_accept`: Omitted validation stages block `ACCEPT`.
9. `test_malformed_structural_input_rejects`: Null or invalid objects safely fail closed.
10. `test_oop_interface_and_invalid_modes`: Class methods and mode validations function cleanly.

## 7. Research Significance

This policy solves a critical research problem identified in Task 01:  
A generator can achieve 100% target coverage (Task 02) while failing statistical fidelity (Task 03) and failing offline utility (Task 05).  
Under this policy, such an artifact is definitively rejected, preventing erroneous model repair.

## 8. Conclusion

Sprint 05 — Task 06 is fully implemented and verified with 10 unit tests.  
Proceeding to **Task 07: Build Automated Repair Experiment Runner**.
