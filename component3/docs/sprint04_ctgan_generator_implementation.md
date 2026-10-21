# Sprint 04 — Task 03: CTGAN Generator Implementation

**Research Project:** J26-DS-361 — Intelligent MLOps Platform\
**Component:** C3 — Adaptive Intelligent Synthetic Data Repair Engine\
**Sprint:** 04 — Synthetic Data Generator Implementation\
**Task:** 03 — Implement CTGAN Generator\
**Implementation Date:** October 13, 2026\
**Status:** Implementation and tests verified; documentation verification pending

## 1. Objective

Implement and verify CTGAN as a synthetic tabular-data generator for Component 3.

The generator must support the shared generator interface, correctly identify categorical features, train on prepared affected-slice data, and produce synthetic records.

The implementation must reuse existing project code without breaking other Component 3 modules.

## 2. Existing Implementation and Improvements

An existing CTGAN adapter was already available in:

`component3/src/generators/ctgan_generator.py`

The adapter uses the CTGAN library and implements the shared `BaseGenerator` interface.

The existing implementation automatically identified categorical columns using pandas data types.

However, the UCI Adult dataset contains numerically encoded categorical features. Detecting categorical features using pandas data types alone may classify these features incorrectly.

The adapter was therefore enhanced to accept explicit categorical-column metadata.

## 3. Updated CTGAN Generator Interface

The updated generator supports:

- `CTGANGenerator(epochs=300, discrete_columns=None)`
- `fit(data)` to train the generator.
- `sample(n_rows)` to generate synthetic records.

The `discrete_columns` parameter accepts a list or tuple of categorical column names.

If explicit metadata is provided, the generator passes those columns to the underlying CTGAN model.

If metadata is omitted, the generator uses the original pandas data-type detection as a fallback.

The implementation remains compatible with the common generator interface.

## 4. Input Validation and Safety Checks

The generator validates:

- The training input is a pandas DataFrame.
- The training DataFrame is not empty.
- Column names are not duplicated.
- Training data contains no missing values.
- Explicit categorical metadata has valid column names.
- Categorical metadata does not contain duplicate names.
- Every declared categorical column exists in the training DataFrame.
- Training epochs are valid positive integers.
- Sampling requests use positive integer row counts.
- Sampling is not permitted before successful fitting.

The generator also passes a deep copy of the training DataFrame to the underlying CTGAN model to protect caller-owned training data from in-place modification.

A failed underlying retraining attempt clears the fitted state before training begins.

These safeguards validate the generator's interface and inputs. They do not independently establish the statistical quality or privacy safety of generated records.

## 5. Integration with Component 3 Data Preparation

The CTGAN adapter was connected to the existing generator data-preparation workflow.

The preparation module provides:

- Training DataFrame
- Continuous-feature metadata
- Categorical-feature metadata
- Training-row count
- Generation mode

For the tested scenarios, the readiness decision permitted exploratory generation but did not authorize supervised retraining.

Therefore, the target label was excluded from generator training.

The verified generation mode was:

`unsupervised_exploratory`

This distinction prevents exploratory synthetic feature generation from being incorrectly presented as validated, labeled repair data.

## 6. Unit Testing

The new test file is:

`component3/tests/test_ctgan_metadata.py`

Eight tests were executed successfully.

Verified behaviors included:

1. Passing explicit categorical metadata to CTGAN.
2. Accepting tuple-based categorical metadata.
3. Falling back to data-type detection when metadata is absent.
4. Rejecting invalid categorical metadata.
5. Rejecting categorical columns missing from training data.
6. Protecting original training data from mutation.
7. Clearing the fitted state after failed retraining.
8. Forwarding sampling requests to the underlying CTGAN model.

**Result:** 8/8 tests passed in 10.38 seconds.

These tests used a mocked underlying CTGAN model to verify adapter behavior.

## 7. Real CTGAN Training Smoke Test — S04

**Scenario:** S04\
**Affected-slice condition:** `race == 2`

The generator was trained on the actual prepared affected-slice dataset using the CTGAN library.

| Metric                    | Observed result |
| ------------------------- | --------------- |
| Training records          | 360             |
| Continuous features       | 5               |
| Categorical features      | 7               |
| Training epochs           | 1               |
| Training duration         | 4.49 seconds    |
| Generated records         | 10              |
| Generated feature columns | 12              |
| Missing generated values  | 0               |
| Output column order       | Correct         |
| Fitted state              | True            |

**Result:** Real CTGAN smoke test passed.

This confirms functional training and sampling with the prepared UCI Adult dataset.

One training epoch is only a functional test. It is not sufficient for final fidelity, privacy, or downstream utility conclusions.

## 8. S04 Target-Slice Coverage Diagnostic

A second S04 experiment generated 100 synthetic records after training CTGAN on the `race == 2` affected slice.

**Results:**

- Generated records: 100
- Records satisfying `race == 2`: 100
- Observed target coverage: 100%

The training slice contained only the categorical value `race == 2`, and CTGAN generated that same category in this run.

This is a scenario-specific observation and does not guarantee perfect coverage for other affected slices or future runs.

## 9. S05 Continuous-Slice Coverage Diagnostic

**Scenario:** S05\
**Affected-slice condition:** `age > 0.69 and education_num < -0.03`

The S05 affected slice contained 342 training records.

CTGAN was trained for one epoch and generated 100 synthetic records.

| Condition                         | Matching generated records |
| --------------------------------- | -------------------------- |
| `age > 0.69`                      | 96                         |
| `education_num < -0.03`           | 97                         |
| Both conditions simultaneously    | 93                         |
| Outside the complete target slice | 7                          |

**Observed target coverage:** 93%

This experiment demonstrated that CTGAN sampling can produce values outside the continuous conditions defining its training slice.

Therefore, targeted generation cannot rely only on training-data filtering.

Post-generation target coverage validation is required before accepting synthetic records as candidates for affected-slice repair.

## 10. S05 Continuous Feature-Range Diagnostic

A separate one-epoch CTGAN experiment generated 100 records using the S05 affected slice.

Each generated continuous value was compared against the minimum and maximum observed for that feature in the training slice.

| Feature        | Training minimum | Training maximum | Generated minimum | Generated maximum | Out-of-range values |
| -------------- | ---------------- | ---------------- | ----------------- | ----------------- | ------------------- |
| age            | 0.6941           | 4.3509           | 0.9804            | 4.8514            | 4                   |
| education_num  | -2.9338          | -0.1195          | -3.2261           | 0.0263            | 6                   |
| capital_gain   | -0.1453          | 0.7340           | -0.3842           | 0.5814            | 61                  |
| capital_loss   | -0.2174          | 9.4128           | -1.6877           | 10.7443           | 23                  |
| hours_per_week | -3.1795          | 4.7127           | -3.5494           | 4.5221            | 2                   |

**Total:** 96 feature-level range violations across 100 generated records.

The total counts feature-level violations, not unique records. A single record can contribute more than one violation.

The strongest observed range deviation occurred for `capital_gain`, followed by `capital_loss`.

These results identify possible range-validity problems, but observed training limits are not necessarily absolute real-world feature limits.

Further structural validation must distinguish empirical range deviations from genuine domain violations.

## 11. Full Regression Verification

The complete Component 3 test suite was executed using:

`.\.venv\Scripts\python.exe -m pytest component3/tests/ -q`

**Results:**

- Total tests: 279
- Passed: 279
- Failed: 0
- Errors: 0
- Non-failing warnings: 4
- Execution time: 13.77 seconds

The previous verified baseline contained 271 passing tests.

Eight CTGAN metadata tests were added, increasing the suite to 279 tests.

The warnings concerned PyTorch deprecation and serialized-model compatibility notices. They did not cause test failures.

No regression was detected in the tested Component 3 modules.

## 12. Research Significance

The CTGAN implementation contributes to Component 3 in three ways.

**First: Correct categorical handling.** Explicit feature metadata supports numerically encoded categorical columns that data-type detection alone may miss.

**Second: Targeted generator integration.** CTGAN can train using prepared affected-slice data under the existing readiness and data-preparation controls.

**Third: Experimental evidence of validation requirements.** The S04 and S05 diagnostics demonstrate why successful sampling is not equivalent to validated synthetic repair data.

The S05 experiments provide preliminary evidence supporting target-coverage checks and feature-range diagnostics in the later validation engine.

These experiments are implementation diagnostics, not final comparative research evaluations.

## 13. Limitations

The following limitations remain:

- Only one-epoch CTGAN runs were used for the reported real-data diagnostics.
- Results were obtained from individual stochastic runs.
- No repeated-seed or multi-run statistical analysis has been completed.
- No formal fidelity, privacy, or downstream utility evaluation has been completed.
- Target coverage has not been tested across all eight drift scenarios.
- No automatic target-coverage filtering or acceptance policy has been implemented.
- Observed training-range deviations have not been classified against verified domain constraints.
- The generated records are not approved for supervised model retraining.
- No evidence of actual prediction-performance recovery has been established.

## 14. Related Project Files

**Implementation**

`component3/src/generators/ctgan_generator.py`

`component3/src/generators/base.py`

`component3/src/generators/data_preparer.py`

**Tests**

`component3/tests/test_ctgan_metadata.py`

`component3/tests/test_ctgan.py`

`component3/tests/test_generator_interface.py`

**Configuration**

`component3/configs/c3_config.json`

## 15. Next Planned Work

The next original Sprint 04 task is:

**Task 04 — Test CTGAN Targeted Data Generation**

This task will build on the verified generator implementation and investigate targeted synthetic generation more systematically.

Later Sprint 05 tasks will implement structural validation, target-coverage validation, fidelity assessment, privacy assessment, and offline utility evaluation.

The final research objective is not simply to generate realistic records. It is to identify whether synthetic records can safely and measurably support recovery in affected model-performance slices.

## 16. Task Completion Assessment

CTGAN adapter implementation: **Completed**

Categorical metadata integration: **Completed**

Focused adapter tests: **8/8 passed**

Real CTGAN smoke test: **Passed**

Preliminary S04/S05 diagnostics: **Completed**

Full Component 3 regression suite: **279/279 passed**

Technical documentation: **Prepared; file verification pending**

**Final task status:** Pending documentation verification before marking Sprint 04 Task 03 complete.
