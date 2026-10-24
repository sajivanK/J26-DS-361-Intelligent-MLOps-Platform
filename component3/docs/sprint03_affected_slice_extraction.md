# Sprint 03 — Task 06: Affected Slice Extraction

**Project:** Intelligent MLOps Platform — J26-DS-361\
**Component:** C3 — Adaptive Intelligent Synthetic Data Repair Engine\
**Sprint:** 03 — Core C3 Architecture & Repair Readiness\
**Task:** 06 — Implement Affected Slice Extraction\
**Original Planned Period:** September 26–30, 2026\
**Actual Implementation and Verification Date:** October 13, 2026\
**Status:** Implementation and tests verified; documentation awaiting file verification

---

## 1. Objective

Implement and verify a safe, reusable mechanism for extracting the records identified as affected by drift in the Component 2 diagnosis.

The extracted records will support targeted synthetic data repair rather than indiscriminate generation across the entire dataset.

The implementation must:

- Interpret Component 2's `affected_slice_rule`.
- Select records satisfying the diagnosis rule.
- Preserve the original dataset's structure.
- Avoid modifying source data.
- Reject unsupported or unsafe expressions.
- Handle empty and full-batch affected slices.
- Integrate with existing Component 3 validation and readiness modules.

## 2. Existing Implementation Reused

The affected-slice extraction functionality was originally developed during Sprint 03 Task 03 as part of input schema and security validation.

Instead of creating a duplicate implementation, Task 06 reused and independently verified the existing functionality.

**Implementation file:**

`component3/src/contracts/safe_slice_evaluator.py`

**Primary functions:**

`evaluate_slice_rule(df, rule)`

Evaluates a restricted diagnosis expression and returns a Boolean mask identifying matching records.

`extract_affected_slice(df, rule)`

Applies the Boolean mask and returns a detached copy of the selected records.

The extraction uses:

```
mask = evaluate_slice_rule(df, rule)
return df.loc[mask].copy()
```

The `.copy()` operation prevents changes to the extracted records from modifying the original dataset.

## 3. Component 2 Integration

Component 3 obtains the affected-slice rule from the official Component 2 diagnosis contract.

**Required contract field:**

`affected_slice_rule`

The field name was confirmed by inspecting the existing diagnosis payload. An initial temporary test incorrectly used `slice_rule`, causing a `KeyError`. The test was corrected to use the official contract field without modifying the implementation.

The extraction functions are already integrated with:

- `component3/src/contracts/input_validator.py`
- `component3/src/contracts/schema_validator.py`
- `component3/src/readiness/readiness_engine.py`

The readiness engine uses affected-slice extraction as part of its readiness assessment.

## 4. Restricted Expression Evaluation

The evaluator uses Python's Abstract Syntax Tree (AST) to interpret an allowlisted subset of expressions.

Supported operations include:

- Numerical and categorical comparisons.
- Boolean `and`, `or`, and `not`.
- Membership comparisons using `in` and `not in`.
- Column references and permitted literal values.

Unsupported constructs include function calls, attribute access, subscripting, arithmetic expressions, and chained comparisons.

The evaluator does not use Python `eval()`, `exec()`, or `pandas.DataFrame.query()`.

**Configured restrictions:**

- Maximum rule length: 2,000 characters.
- Maximum membership-list size: 100 items.

Unsupported or invalid expressions raise `UnsafeSliceRuleError`.

These controls reduce the risk of unsafe execution when interpreting diagnosis expressions.

## 5. Extraction Accuracy and Data Integrity

An independent test was performed using a five-record DataFrame and the following rule:

```
age > 0.69 and education_num < -0.03
```

**Verified results:**

- Exactly records with indices `[1, 4]` were extracted.
- Non-matching records were excluded.
- All extracted records satisfied both conditions.
- The original DataFrame remained unchanged.
- Modifying the extracted DataFrame did not modify the source DataFrame.

**Result: 5/5 checks passed.**

## 6. Controlled Drift Scenario Verification

Extraction was verified against all eight existing controlled drift scenarios and simulated Component 2 diagnoses.

Each scenario used its stored drifted dataset and `affected_slice_rule`.

| Scenario | Extracted Records | Total Records | Slice Support |
| -------- | ----------------- | ------------- | ------------- |
| S00      | 0                 | 6,000         | 0.00%         |
| S01      | 2,159             | 6,000         | 35.98%        |
| S02      | 6,000             | 6,000         | 100.00%       |
| S03      | 1,723             | 6,000         | 28.72%        |
| S04      | 360               | 6,000         | 6.00%         |
| S05      | 342               | 6,000         | 5.70%         |
| S06      | 1,674             | 6,000         | 27.90%        |
| S07      | 1,550             | 6,000         | 25.83%        |

For every scenario, the following properties were verified:

1. Extracted record count matched the Boolean-mask count.
2. Extracted indices matched the source DataFrame's selected indices.
3. Calculated slice support matched the simulated Component 2 diagnosis within a tolerance of `1e-5`.
4. Original input data remained unchanged.

**Result: 8/8 scenarios verified.**

### Important Edge Cases

**S00 — Empty affected slice**

The control scenario produced zero matching records. This verifies correct extraction behavior when no records satisfy the rule.

**S02 — Full-batch affected slice**

The rule selected all 6,000 records. This verifies correct extraction behavior when the affected slice covers the entire current batch.

These are extraction edge cases, not evidence that synthetic generation should automatically proceed.

## 7. Additional Safeguard Diagnostics

Five additional diagnostic cases were executed.

| Diagnostic                           | Observed Result                      |
| ------------------------------------ | ------------------------------------ |
| Duplicate column names               | Safely rejected                      |
| Incompatible numeric/text comparison | Safely rejected                      |
| Missing numerical values             | Accepted with valid Boolean mask     |
| Invalid membership operand           | Safely rejected                      |
| Comparison against `None`            | Accepted with all-False Boolean mask |

No unhandled exceptions occurred in these five diagnostics.

The observed behavior was consistent with the tested evaluator implementation. These tests do not establish safety against every possible malformed input or adversarial expression.

## 8. Permanent Automated Tests

**Test file:**

`component3/tests/test_affected_slice_extraction.py`

The permanent test suite covers:

- Eight controlled drift scenarios.
- Empty affected-slice extraction.
- Full-batch extraction.
- Independent copying and source-data protection.
- Preservation of original column order.

**Execution command:**

```
.\.venv\Scripts\python.exe -m pytest component3/tests/test_affected_slice_extraction.py -v
```

**Result: 12/12 tests passed in 0.59 seconds.**

Security-specific tests remain in the existing `test_schema_security.py` suite rather than being duplicated.

## 9. Full Component 3 Regression Testing

The complete Component 3 test suite was executed after adding the permanent extraction tests.

**Command:**

```
.\.venv\Scripts\python.exe -m pytest component3/tests/ -v
```

| Test Group                 | Passed  |
| -------------------------- | ------- |
| Core loaders               | 36      |
| Input contracts            | 18      |
| Schema security            | 44      |
| Readiness engine           | 39      |
| Readiness checks           | 55      |
| CTGAN and TVAE smoke tests | 2       |
| Affected-slice extraction  | 12      |
| **Total**                  | **206** |

**Final regression result: 206/206 PASS in 5.39 seconds.**

- Failures: 0.
- Warnings: 3.

The warnings originate from existing serialized model compatibility differences involving XGBoost and scikit-learn. They are not caused by the affected-slice extraction tests and remain an outstanding compatibility risk.

## 10. Research Significance

Affected-slice extraction is an important foundation for the Component 3 targeted synthetic repair workflow.

Instead of generating replacement records for every observation, the system can identify a subset associated with the diagnosis.

For example, scenario S05 identifies 342 affected records out of a 6,000-record current batch.

These records can provide a targeted starting point for later repair planning, generator training-data preparation, synthetic generation, and validation.

However, extracted records are not automatically trustworthy training data. Readiness checks, label provenance, generator suitability, fidelity, privacy, and downstream utility must still be considered.

## 11. Limitations and Future Work

1. The scenario diagnoses are simulated and use rules derived from known drift injections. The tests do not verify the accuracy of a real Component 2 diagnosis.
2. A syntactically valid rule does not necessarily identify the true root cause of model degradation.
3. The evaluator supports a restricted expression language rather than arbitrary Python or pandas expressions.
4. Missing values may be excluded by ordinary comparison semantics; this behavior should be considered when designing future diagnosis rules.
5. Additional integration tests will be required when the real Component 2 output becomes available.
6. Passing extraction tests does not establish that synthetic generation or supervised retraining is ready or authorized.

## 12. Conclusion

The existing affected-slice extraction mechanism was successfully reused, inspected, and independently verified.

The implementation correctly extracted records for all eight controlled drift scenarios, preserved source-data integrity, handled empty and full-batch cases, and safely handled the five additional diagnostic inputs.

Permanent automated tests were added, and the full Component 3 regression suite passed with 206 tests.

**Implementation Status:** Verified\
**Testing Status:** Verified\
**Documentation Status:** Awaiting saved-file verification\
**Task Completion:** Pending final documentation verification
