# Sprint 02 — Component 1 and Component 2 Input Contract Validation

## 1. Task Information

- **Project:** Intelligent MLOps Platform
- **Research Component:** Component 3 — Adaptive Intelligent Synthetic Data Repair Engine
- **Sprint:** 02 — Controlled Drift and Diagnosis Simulation
- **Task:** 07 — Validate C1/C2 Input Contracts
- **Actual implementation and testing date:** 12 October 2026
- **Status:** Implementation and automated tests completed; documentation verification pending

## 2. Objective

The objective of this task was to validate incoming drift evidence from Component 1 and diagnostic information from Component 2 before Component 3 uses these inputs for synthetic data repair.

The implementation checks individual input contracts and consistency between the two components.

## 3. Input Contracts

### Component 1 — Drift Detection Evidence

Contract file: `shared/contracts/c1_output_schema.json`

The C1 payload contains 13 required fields, including model and dataset references, drift type, drift severity, predicted harm, affected features, detector scores, detector counts, and confidence statistics.

### Component 2 — Drift Diagnosis

Contract file: `shared/contracts/c2_output_to_c3_schema.json`

The C2 payload contains nine required fields, including root-cause feature, root-cause category, affected slice rule, ranked affected features, affected value ranges, slice support, diagnosis stability, drift impact score, and temporal window.

These contracts define the expected structure of the information received by Component 3.

## 4. Implementation

The following files were created:

- `component3/src/contracts/__init__.py`
- `component3/src/contracts/input_validator.py`
- `component3/tests/test_input_contracts.py`

The input validator implements the following functions:

**`load_contracts()`**

Loads the official C1 and C2 contract definitions.

**`validate_c1_input()`**

Validates Component 1 evidence against the expected field and value constraints. Invalid payloads raise `InputContractError`.

**`validate_c2_input()`**

Validates Component 2 diagnosis against its expected constraints and checks that its affected slice expression can be evaluated against reference data.

**`validate_cross_component_consistency()`**

Checks consistency between the two payloads and the current dataset, including drift category correspondence, affected feature alignment, affected slice execution, and reported slice support.

The implementation separates hard errors from warnings. Hard errors indicate invalid input combinations, while warnings identify possible disagreements that require review.

## 5. Validation Experiments

### 5.1 Valid Input Scenarios

Eight simulated drift scenarios, S00–S07, were evaluated.

All eight passed C1 contract validation, C2 contract validation, and cross-component consistency validation.

**Result: 8/8 PASS, zero errors, zero warnings.**

### 5.2 Cross-Component Negative Tests

Four deliberately modified C2 payloads were tested.

| Fault                              | Expected behavior | Result |
| ---------------------------------- | ----------------- | ------ |
| Incorrect slice support            | Hard error        | PASS   |
| Drift category disagreement        | Warning           | PASS   |
| Unexpected root-cause feature      | Warning           | PASS   |
| Invalid slice rule on current data | Hard error        | PASS   |

**Result: 4/4 PASS.**

### 5.3 Malformed Input Tests

Six malformed C1/C2 inputs were tested.

| Fault                           | Result   |
| ------------------------------- | -------- |
| Missing C1 `model_id`           | Rejected |
| String C1 `predicted_harm`      | Rejected |
| C1 `detectors_fired = 7`        | Rejected |
| Missing C2 `root_cause_feature` | Rejected |
| Invalid C2 root-cause category  | Rejected |
| C2 `temporal_window = 0`        | Rejected |

**Result: 6/6 PASS.**

## 6. Automated Regression Testing

A permanent automated test suite was created at:

`component3/tests/test_input_contracts.py`

Execution command:

`.\.venv\Scripts\python.exe -m pytest component3/tests/test_input_contracts.py -v`

Test environment:

- Python 3.11.0
- pytest 9.1.1
- Windows virtual environment `.venv`

Test coverage:

| Category                    | Tests  | Passed |
| --------------------------- | ------ | ------ |
| Valid scenario inputs       | 8      | 8      |
| Malformed C1/C2 inputs      | 6      | 6      |
| Cross-component hard errors | 2      | 2      |
| Cross-component warnings    | 2      | 2      |
| **Total**                   | **18** | **18** |

**Final automated test result: 18 passed in 3.09 seconds.**

## 7. Research and Engineering Limitations

1. C1 and C2 payloads used in these experiments are simulated. Passing the tests does not prove compatibility with the final implementations of the other components.
2. Simulated C1 severity and harm values are based on controlled experimental proxies, not the final calibrated C1 detection system.
3. Simulated C2 diagnosis uses predefined drift information and heuristic values, not actual Temporal SHAP or Sliceline outputs.
4. Current contract validation reuses validation functions from simulator modules. These should be separated into production-focused validation modules before final integration.
5. The C2 affected slice rule is evaluated using pandas query expressions. Before accepting untrusted external inputs, expression safety and allowed syntax must be reviewed and restricted.
6. Passing input contract validation does not establish that synthetic data generation or repair will succeed.
7. The tests cover selected valid and invalid cases rather than every possible malformed payload.

## 8. Conclusion

Sprint 02 Task 07 established an initial input validation layer for Component 3.

The implementation successfully validated eight simulated C1/C2 input pairs and passed 18 automated regression tests.

This provides a tested foundation for later repair-readiness checks, affected slice extraction, and full integration with Components 1 and 2.

The next development stages should address production independence, safe slice rule evaluation, and testing with actual cross-component outputs.
