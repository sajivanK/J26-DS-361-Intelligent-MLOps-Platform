# Sprint 04 — Task 02: Generator Training Data and Metadata

## 1. Task Information

- Component: C3 — Adaptive Intelligent Synthetic Data Repair Engine
- Sprint: 04 — Synthetic Data Generator Implementation
- Task: 02 — Prepare Generator Training Data and Metadata
- Original planned period: October 1–5, 2026
- Actual implementation date: October 13, 2026
- Status: Implemented; final documentation verification pending

## 2. Objective

Prepare affected-slice data for synthetic tabular generators using
consistent feature metadata, readiness controls, and label-trust rules.

The preparation module is designed to support CTGAN, TVAE, and
TabDDPM without performing generator training itself.

## 3. Implemented Files

### Production Module

component3/src/generators/data_preparer.py

Main elements:

- DataPreparationError
- PreparedGeneratorData
- prepare_generator_data()

### Automated Tests

component3/tests/test_generator_data_preparer.py

17 automated tests covering valid and invalid preparation scenarios.

## 4. Preparation Workflow

1. Receive an affected-slice DataFrame.
2. Receive a readiness decision.
3. Receive the Component 3 configuration.
4. Reject blocked or generation-unready requests.
5. Validate the input DataFrame and required features.
6. Preserve feature ordering from the configuration.
7. Apply target inclusion or exclusion rules.
8. Create a detached training-data copy.
9. Validate missing values and continuous feature types.
10. Identify continuous and categorical feature metadata.
11. Validate row-count boundaries and affected-slice consistency.
12. Return the prepared dataset and metadata.

## 5. Readiness and Label Safety

Generation is rejected when the readiness decision blocks generation.

When supervised_retraining_ready is False:

- Target is excluded.
- Generation mode is unsupervised_exploratory.

When supervised_retraining_ready is True and the target exists:

- Target may be included.
- Generation mode is supervised_repair.

The module relies on the supplied readiness decision.
It does not independently authenticate label provenance.

Generated records must not automatically be treated as trusted
supervised-retraining examples.

## 6. Dataset and Feature Metadata

Dataset: UCI Adult

Configured input features: 12

Continuous features: 5

- age
- education_num
- capital_gain
- capital_loss
- hours_per_week

Categorical features: 7

- workclass
- marital_status
- occupation
- relationship
- race
- sex
- native_country

Target column: target

Feature types are obtained from the central C3 configuration
rather than inferred only from pandas data types.

## 7. Real Dataset Verification — Scenario S04

Scenario: S04

Affected-slice rule: race == 2

Observed results:

- Current dataset rows: 6000
- Affected-slice rows: 360
- Prepared training rows: 360
- Slice support: 0.06 (6%)
- Continuous features: 5
- Categorical features: 7
- Target included: False
- Generation mode: unsupervised_exploratory
- Missing values: 0
- Feature ordering: Verified
- Real-data preparation: PASS

The readiness decision in this verification was explicitly
constructed for the test. This was not a fresh end-to-end
validation of actual C1/C2 evidence.

## 8. Input Validation and Failure Handling

The module rejects:

- Non-DataFrame inputs
- Empty datasets
- Duplicate column names
- Missing configured features
- Missing values in selected training columns
- Invalid continuous feature data types
- Boolean values used as continuous features
- Invalid or empty source identifiers
- Generation-unready decisions
- Training data larger than the assessed current dataset
- Affected-slice row counts inconsistent with readiness metadata

Input DataFrames are copied to prevent unintended mutation.

## 9. Automated Test Evidence

Focused test suite:

component3/tests/test_generator_data_preparer.py

Result:

17 passed in 1.11 seconds

Full Component 3 regression suite:

Command:

.\.venv\Scripts\python.exe -m pytest component3/tests/ -q

Result:

271 passed, 4 warnings in 13.56 seconds

Failures: 0

Warnings relate to existing PyTorch deprecation and
serialized-model compatibility notices.

## 10. Known Limitations and Follow-Up Work

1. Matching row counts does not independently verify the
   identity of affected-slice records.

2. Label provenance is not independently authenticated by
   the data preparer.

3. The prepared DataFrame retains its original feature
   representations. Existing generator adapters may infer
   categorical columns from pandas dtypes, while the central
   configuration identifies some numerically encoded columns
   as categorical.

4. Generator-specific categorical handling must be aligned
   with the preparation metadata during later Sprint 04 tasks.

5. This task does not establish synthetic-data fidelity,
   utility, privacy, or downstream repair effectiveness.

6. No generator training or synthetic sampling was performed
   as part of this verification.

7. The metadata object is frozen, but its contained DataFrame
   and feature_types dictionary remain mutable objects.

8. The preparer does not independently verify that a
   supervised-ready decision was produced by the official
   readiness engine.

## 11. Research Significance

This module establishes a reusable preparation boundary
between the readiness engine and synthetic generators.

It provides consistent feature ordering, explicit feature
metadata, basic input validation, and readiness-based
target handling.

It does not yet demonstrate improved model performance
or successful drift repair.

## 12. Completion Evidence

Production module created and import verified.

17 focused automated tests passed.

Real UCI Adult S04 preparation passed.

Full regression suite passed with 271 tests.

Technical documentation prepared.

Final status will be marked COMPLETE after documentation
verification.
