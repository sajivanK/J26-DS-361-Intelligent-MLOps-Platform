# Sprint 03 — Input Schema Validation Technical Report

## 1. Task Information

- **Project:** Intelligent MLOps Platform
- **Component:** C3 — Adaptive Intelligent Synthetic Data Repair Engine
- **Sprint:** 03 — Core C3 Architecture & Repair Readiness
- **Task:** 03 — Implement Input Schema Validation
- **Original planned period:** September 26–30, 2026
- **Actual implementation and testing date:** October 13, 2026
- **Status:** Implementation and automated testing completed; final documentation verification pending

## 2. Objective

Implement a reliable input validation layer for Component 3.

Before Component 3 processes drift evidence or root-cause diagnosis, it must verify that incoming information follows the official integration contracts.

The validation layer must:

1. Validate Component 1 drift evidence.
2. Validate Component 2 diagnosis information.
3. Reject missing or invalid fields.
4. Check supported data types and numerical constraints.
5. Safely evaluate affected-slice expressions.
6. Prevent unsupported Python expressions from being executed.
7. Preserve compatibility with existing Component 3 modules.

## 3. Official Integration Contracts

The validation implementation uses the team-owned contracts:

- `shared/contracts/c1_output_schema.json`
- `shared/contracts/c2_output_to_c3_schema.json`

These files were reviewed without modifying the shared team contracts.

### Component 1 — Drift Evidence

The C1 contract defines 13 required fields:

1. `evidence_version`
2. `model_id`
3. `reference_dataset_uri`
4. `current_dataset_uri`
5. `drift_type`
6. `drift_severity`
7. `predicted_harm`
8. `harm_interval`
9. `harm_metric`
10. `affected_features`
11. `detector_scores`
12. `detectors_fired`
13. `confidence_stats`

Supported field types include strings, floating-point values, integers, typed arrays, and objects.

The validator additionally checks relevant domain constraints, including valid enumerated values, numerical values, ordered harm intervals, and required detector and confidence statistics.

### Component 2 — Root-Cause Diagnosis

The C2 contract defines 9 required fields:

1. `root_cause_feature`
2. `root_cause_category`
3. `affected_slice_rule`
4. `affected_features_ranked`
5. `affected_value_ranges`
6. `slice_support`
7. `diagnosis_stability`
8. `drift_impact_score`
9. `temporal_window`

The official contract specifies the range 0.0–1.0 for:

- `slice_support`
- `diagnosis_stability`
- `drift_impact_score`

The validator also checks that the slice expression refers to supported DataFrame columns and follows the restricted expression syntax.

## 4. Implemented Modules

### 4.1 Schema Validator

**File:** `component3/src/contracts/schema_validator.py`

Main responsibilities:

- Validate required fields and field types.
- Check numerical values and supported constraints.
- Reject malformed C1 evidence.
- Reject malformed C2 diagnosis.
- Return validation errors through `SchemaValidationError`.
- Validate affected-slice expressions without relying on the experiment simulators.

Main interfaces:

- `validate_c1_schema()`
- `validate_c2_schema()`

### 4.2 Safe Slice Evaluator

**File:** `component3/src/contracts/safe_slice_evaluator.py`

Main responsibilities:

- Parse affected-slice expressions using Python's abstract syntax tree (AST).
- Support a restricted set of comparison and Boolean operations.
- Evaluate supported expressions against DataFrame columns.
- Reject unsupported constructs such as arbitrary function calls, attribute access, and subscripting.
- Reject unknown or ambiguous column references in the tested cases.
- Reject empty datasets.
- Handle missing numerical values in the tested comparison case.

Main interfaces:

- `evaluate_slice_rule()`
- `extract_affected_slice()`

**Resource limits implemented:**

- Maximum expression length: 2,000 characters.
- Maximum membership-list size: 100 elements.

The evaluator does not use unrestricted Python `eval()` or `exec()` to execute slice expressions.

These controls reduce risk but do not establish complete security against every possible malformed or resource-intensive expression.

### 4.3 Input Validator Integration

**File:** `component3/src/contracts/input_validator.py`

The input validation layer was updated to use the standalone schema validator and safe slice evaluator.

The existing public validation interfaces were preserved.

The main input validator no longer directly depends on the C1 and C2 simulation modules for schema validation.

The previous use of `DataFrame.query()` in the main cross-component consistency check was replaced with restricted slice evaluation.

## 5. Validation Workflow

1. Receive Component 1 drift evidence and Component 2 diagnosis.
2. Load the official integration contracts.
3. Validate C1 required fields, types, and constraints.
4. Validate C2 required fields, types, and constraints.
5. Parse and validate the affected-slice expression.
6. Evaluate the affected slice against the relevant dataset.
7. Check cross-component consistency, including slice support.
8. Accept valid input for further C3 processing or report a validation failure.

Successful schema validation does not automatically mean that the input is suitable for synthetic generation or supervised retraining. Repair readiness will be handled separately.

## 6. Test Coverage and Results

### 6.1 Valid Scenario Tests

Eight simulated scenario pairs, S00–S07, were successfully validated against the C1 and C2 contracts.

These inputs are simulated integration evidence, not outputs from the final live C1/C2 engines.

### 6.2 Negative Schema Tests

**Result: 17/17 passed.**

Tested invalid inputs included:

- Missing required fields.
- Invalid strings and enumeration values.
- NaN and infinity in numeric fields.
- Boolean values incorrectly supplied as numbers or integers.
- Reversed harm intervals.
- Missing detector scores.
- Invalid confidence statistics.
- Negative stability values.
- Unsafe function calls.
- Unknown DataFrame columns.

All 17 invalid inputs were rejected using the expected schema validation exception.

### 6.3 Edge-Case and Resource Tests

**Result: 8/8 passed.**

The tests covered:

- Duplicate column names.
- Empty datasets.
- Excessively long slice expressions.
- Oversized membership lists.
- Unsupported function calls.
- Unsupported dictionary expressions.
- Excessive parenthesis nesting.
- Missing values in a numeric column.

The evaluator rejected the tested invalid expressions and correctly evaluated the missing-value example.

### 6.4 Permanent Automated Tests

**File:** `component3/tests/test_schema_security.py`

Automated test result:

**44 passed in 2.43 seconds.**

The test suite covers valid schema pairs, invalid payloads, unsafe expressions, supported operators, and selected edge cases.

### 6.5 Combined Regression Testing

Executed:

`.\.venv\Scripts\python.exe -m pytest component3/tests/test_schema_security.py component3/tests/test_input_contracts.py component3/tests/test_core_loaders.py -q`

Results:

| Test file                 | Passed |
| ------------------------- | ------ |
| `test_schema_security.py` | 44     |
| `test_input_contracts.py` | 18     |
| `test_core_loaders.py`    | 36     |
| **Total**                 | **98** |

**Final reported result:** 98 passed, 3 warnings, in 23.09 seconds.

There were no test failures. Three warnings were reported during the combined run and should be reviewed separately.

## 7. Integration and Compatibility

The implementation was tested against the existing Component 3 configuration, data-loading, and integration-contract modules.

The shared C1 and C2 contract files were not modified.

The tests demonstrate compatibility with the eight existing simulated scenarios and the tested regression cases.

Integration with live Component 1 and Component 2 services remains future work.

## 8. Limitations and Future Improvements

1. The current evidence uses simulated C1 and C2 payloads.
2. The input validator is not yet integrated into a complete production C3 API.
3. Passing the current tests does not prove that all possible malformed inputs are handled.
4. Further testing may be needed for deeply complex expressions, unusual DataFrame data types, and nested payload constraints.
5. The three regression-test warnings should be inspected and classified.
6. Schema validity alone does not prove label trust, data sufficiency, diagnosis reliability, or repair readiness.

These remaining concerns should be addressed during subsequent implementation and integration work.

## 9. Research and Engineering Significance

This task establishes a controlled interface between drift detection, root-cause diagnosis, and the synthetic data repair engine.

The validation layer helps prevent malformed evidence from entering the repair pipeline and provides a safer mechanism for selecting affected records.

It also supports modular development because Component 3 can validate inputs independently of the simulation scripts.

## 10. Task Outcome

The following work has been implemented and tested:

- C1 and C2 contract-driven schema validation.
- Standalone validation modules.
- Restricted AST-based slice evaluation.
- Basic resource limits for slice expressions.
- Integration with the existing C3 input validator.
- Permanent automated security and schema regression tests.
- Successful combined regression testing: 98/98 tests passed.

**Final task closure:** Pending confirmation that this documentation file has been saved and verified in the repository.
