# Sprint 03 — Task 08: Test Core C3 Modules

**Project:** Intelligent MLOps Platform — J26-DS-361\
**Component:** C3 — Adaptive Intelligent Synthetic Data Repair Engine\
**Sprint:** 03 — Core C3 Architecture & Repair Readiness\
**Task:** 08 — Test Core C3 Modules\
**Original Planned Period:** September 26–30, 2026\
**Actual Implementation Date:** October 13, 2026\
**Status:** Implementation and testing verified; documentation awaiting file verification

---

## 1. Objective

Verify that the existing Component 3 core modules work correctly together as an integrated readiness workflow.

The task focuses on testing interactions between configuration loading, dataset loading, C1/C2 evidence loading, input contract validation, cross-component consistency validation, affected-slice extraction, and repair readiness decisions.

The tests must also confirm that invalid inputs fail safely and that processing does not modify the supplied datasets or evidence.

## 2. Existing Architecture Reviewed

The following modules were inspected before developing integration tests.

### Core Modules

- `component3/src/core/config_loader.py`
- `component3/src/core/data_loader.py`
- `component3/src/core/model_loader.py`

### Contract Modules

- `component3/src/contracts/input_validator.py`
- `component3/src/contracts/schema_validator.py`
- `component3/src/contracts/safe_slice_evaluator.py`

### Readiness Modules

- `component3/src/readiness/readiness_models.py`
- `component3/src/readiness/readiness_checks.py`
- `component3/src/readiness/readiness_engine.py`

### Existing Tests

Reviewed the existing unit, contract, security, readiness, and slice-extraction tests.

Before Task 08, the full Component 3 pytest suite contained 220 passing tests.

The primary identified gap was the absence of a dedicated test file covering the complete readiness workflow from saved inputs through the final readiness decision.

## 3. Integration Test Implementation

Created:

`component3/tests/test_core_integration.py`

The new test file integrates the existing modules rather than introducing a duplicate production workflow.

### Tested Processing Sequence

1. Load the central Component 3 configuration.
2. Load the reference dataset.
3. Load the official C1 and C2 integration contracts.
4. Load the current dataset for a controlled drift scenario.
5. Load the corresponding saved C1 evidence and C2 diagnosis.
6. Validate both payloads against their contracts.
7. Verify cross-component consistency.
8. Assess repair readiness.
9. Serialize the resulting readiness decision.
10. Extract the affected slice independently.
11. Compare the extracted row count against the readiness decision.
12. Verify that input data and evidence remain unchanged.

The integration tests use existing project artifacts and do not retrain models, generate new synthetic datasets, or modify shared team resources.

## 4. Positive Integration Tests

Implemented eight parameterized integration tests covering controlled drift scenarios S00–S07.

### Scenario Coverage

| Scenario                            | Expected Readiness | Expected Generation Eligibility |
| ----------------------------------- | ------------------ | ------------------------------- |
| S00 — No drift                      | BLOCKED            | False                           |
| S01 — Age drift                     | REVIEW             | True                            |
| S02 — Capital-loss drift            | REVIEW             | True                            |
| S03 — Marital-status drift          | REVIEW             | True                            |
| S04 — Race drift                    | REVIEW             | True                            |
| S05 — Numerical multi-feature drift | REVIEW             | True                            |
| S06 — Categorical interaction drift | REVIEW             | True                            |
| S07 — Combined drift                | REVIEW             | True                            |

These expectations apply to the tested simulated evidence and missing verified label provenance.

### Verified Conditions

For every scenario:

- Configuration and data loading succeeded.
- Saved C1 and C2 payloads passed their respective input contract checks.
- Cross-component consistency validation passed.
- Readiness decisions were generated successfully.
- Decision dictionaries were JSON serializable.
- Independently extracted affected-slice row counts matched the readiness decision.
- Supervised retraining eligibility remained False without verified label provenance.
- Input DataFrames and C1/C2 payload dictionaries remained unchanged.

For S00, the engine returned BLOCKED with the NO_DRIFT_DETECTED reason.

For S01–S07, the engine returned REVIEW with LABEL_PROVENANCE_UNKNOWN and generation_ready=True.

Generation eligibility in this context does not independently authorize production deployment or supervised retraining.

### Positive Test Result

**8/8 PASS in 0.59 seconds.**

## 5. Negative Integration Tests

Added six additional tests to verify safe handling of invalid inputs.

| Failure Scenario                 | Expected Status | Expected Reason               |
| -------------------------------- | --------------- | ----------------------------- |
| Missing required C1 field        | BLOCKED         | INVALID_INPUT_CONTRACT        |
| Missing required C2 slice rule   | BLOCKED         | INVALID_INPUT_CONTRACT        |
| Incorrect affected-slice support | BLOCKED         | CROSS_COMPONENT_INCONSISTENCY |
| Unsafe slice expression          | BLOCKED         | INVALID_INPUT_CONTRACT        |
| Insufficient current data        | BLOCKED         | INSUFFICIENT_CURRENT_ROWS     |
| Invalid readiness configuration  | BLOCKED         | INVALID_INPUT_CONTRACT        |

### Verified Safety Properties

Each invalid scenario returned:

- `status=BLOCKED`
- The expected machine-readable reason code.
- `generation_ready=False`
- `supervised_retraining_ready=False`

The negative tests intentionally modified copies of the original configuration or evidence. They did not modify saved project artifacts.

## 6. Test Snapshot Issue and Correction

During the first execution of the negative integration tests, four assertions failed.

**Initial result: 10 passed, 4 failed.**

The failures were caused by a test-design mistake rather than a confirmed readiness-engine failure.

The test captured C1/C2 input snapshots before intentionally modifying those inputs to simulate invalid evidence.

Consequently, the final immutability assertions compared the modified test inputs against their original, unmodified state.

### Correction Applied

Moved the input snapshots to immediately after the deliberate test mutations and before the readiness engine invocation.

This ensures that the immutability checks verify the intended property:

The readiness engine must not modify the exact inputs it receives.

### Corrected Test Result

**14/14 PASS in 0.62 seconds.**

No production code changes were required to resolve the snapshot issue.

## 7. Input Data Integrity Verification

The integration tests used independent copies of the input datasets and evidence.

The tests verified that the readiness engine did not modify:

- The reference DataFrame.
- The current DataFrame.
- The C1 evidence dictionary.
- The C2 diagnosis dictionary.

DataFrame equality was checked using pandas testing utilities.

Dictionary equality was checked against deep-copied snapshots.

All tested data-integrity assertions passed.

## 8. Complete Integration Test Results

**Test command:**

```
.\.venv\Scripts\python.exe -m pytest component3/tests/test_core_integration.py -v
```

| Test Category                        | Passed |
| ------------------------------------ | ------ |
| Positive S00–S07 workflow tests      | 8      |
| Negative input and failure scenarios | 6      |
| **Total**                            | **14** |

**Final integration result: 14/14 PASS in 0.62 seconds.**

## 9. Full Component 3 Regression Testing

After adding the new integration tests, the complete Component 3 pytest suite was executed.

**Command:**

```
.\.venv\Scripts\python.exe -m pytest component3/tests/ -v
```

| Test File                         | Passing Tests |
| --------------------------------- | ------------- |
| test_core_loaders.py              | 36            |
| test_input_contracts.py           | 18            |
| test_schema_security.py           | 44            |
| test_readiness_engine.py          | 53            |
| test_readiness_checks.py          | 55            |
| test_affected_slice_extraction.py | 12            |
| test_core_integration.py          | 14            |
| CTGAN and TVAE smoke tests        | 2             |
| **Total**                         | **234**       |

**Final result: 234/234 PASS in 5.34 seconds.**

- Failures: 0
- Warnings: 3

The three warnings are existing XGBoost and scikit-learn serialized-model compatibility warnings from trusted model-loading tests.

No regressions were detected in the tested modules.

## 10. Research Significance

Component 3 receives drift evidence and root-cause diagnosis from upstream components.

Therefore, its readiness workflow must be reliable across multiple interacting modules.

This task demonstrates that the current readiness workflow can:

- Load and validate saved upstream evidence.
- Enforce cross-component consistency.
- Extract the affected data slice.
- Produce structured readiness decisions.
- Prevent supervised retraining when label provenance is not verified.
- Reject invalid evidence and unsafe expressions.
- Preserve the original data and evidence during processing.

This establishes a tested foundation for the synthetic-data generation, validation, and repair-prescription modules planned for later sprints.

## 11. Limitations and Future Work

1. Integration tests currently use controlled drift scenarios and simulated C1/C2 evidence, not live outputs from the final teammate components.
2. The tests cover the readiness workflow, not the complete synthetic repair and retraining pipeline.
3. Baseline model loading has separate unit tests but is not part of this new integrated readiness workflow.
4. Readiness thresholds remain provisional.
5. Verified label provenance is not independently authenticated by these tests.
6. Generator quality, privacy, utility, repair sizing, and model recovery require later experiments.
7. Production deployment and real C1/C2/C4 integration remain pending.

## 12. Conclusion

Sprint 03 Task 08 successfully added dedicated integration testing for the existing Component 3 core readiness workflow.

Eight positive tests verified all controlled drift scenarios, and six negative tests verified safe rejection of invalid inputs.

An initial test snapshot-placement issue was identified and corrected without changing production code.

The final integration suite passed all 14 tests, and the complete Component 3 regression suite passed all 234 tests.

**Integration Testing:** Verified\
**Full Regression Testing:** Verified\
**Technical Documentation:** Awaiting saved-file verification\
**Task Completion:** Pending documentation verification
