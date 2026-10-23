# Sprint 02 — Task 02: Controlled Drift Experiments

**Research Project:** Intelligent MLOps Platform\
**Component:** C3 — Adaptive Intelligent Synthetic Data Repair Engine\
**Task:** Implement and Test Controlled Drift Scenarios\
**Original Sprint:** 21–25 September 2026\
**Actual Completion:** 12 October 2026\
**Status:** Completed — 6/6 subtasks

## 1. Objective

The objective was to create reproducible drift scenarios using the existing shared drift injection framework. These scenarios will support later experiments involving drift diagnosis, synthetic data repair, and model performance recovery.

## 2. Experiment Setup

- Dataset: UCI Adult (preprocessed)
- Reference dataset: 26,048 records
- Evaluation batch: 6,000 records
- Model: Existing baseline XGBoost classifier
- Random seed: 42
- Drift scenarios: S00–S07
- Shared framework: `shared/injection/`
- Experiment configuration: `component3/configs/drift_scenarios.json`

The same reproducibly sampled batch was used as input for all scenarios. This avoided inconsistent batch sizes caused by the shared injection framework's handling of the `batch_size` parameter.

No shared framework files were modified.

## 3. Implemented Scenarios

| ID  | Drift Type                  | Affected Features                               | Magnitude |
| --- | --------------------------- | ----------------------------------------------- | --------- |
| S00 | None                        | None                                            | 0.0       |
| S01 | Numerical marginal          | age                                             | 0.4       |
| S02 | Numerical marginal          | capital_loss                                    | 0.4       |
| S03 | Categorical marginal        | marital_status                                  | 0.4       |
| S04 | Categorical marginal        | race                                            | 0.4       |
| S05 | Multiple numerical marginal | age, education_num                              | 0.6       |
| S06 | Feature interaction         | marital_status, relationship                    | 0.6       |
| S07 | Combined                    | age and marital_status–relationship interaction | 0.6       |

## 4. Experimental Results

The baseline accuracy on the selected reference batch was 88.83%.

| Scenario | Drifted Accuracy | Accuracy Drop (pp) | Severity | Malignant |
| -------- | ---------------- | ------------------ | -------- | --------- |
| S00      | 88.83%           | 0.00               | None     | No        |
| S01      | 88.02%           | 0.82               | None     | No        |
| S02      | 86.75%           | 2.08               | Low      | Yes       |
| S03      | 88.00%           | 0.83               | None     | No        |
| S04      | 88.70%           | 0.13               | None     | No        |
| S05      | 86.52%           | 2.32               | Low      | Yes       |
| S06      | 87.23%           | 1.60               | Low      | No        |
| S07      | 86.03%           | 2.80               | Low      | Yes       |

Three scenarios exceeded the framework's 2-percentage-point malignancy threshold: S02, S05, and S07.

## 5. Magnitude Sensitivity Experiment

Twelve additional experiments tested `age` and `capital_loss` at magnitudes 0.1, 0.2, 0.4, 0.6, 0.8, and 1.0.

For `age`, the observed accuracy drop increased from 0.25 to 2.23 percentage points as magnitude increased.

For `capital_loss`, the accuracy drop reached 2.08 percentage points at magnitude 0.4 but decreased at higher magnitudes.

These observations demonstrate that increasing the configured drift magnitude does not always increase model performance degradation.

The precise cause of the non-monotonic behavior has not yet been established.

## 6. Interaction Drift Verification

Scenario S06 was independently checked for disruption of the relationship between `marital_status` and `relationship`.

Results:

- Total records: 6,000
- Rows with changed `relationship` values: 2,633
- Joint distribution difference: 1,823
- `marital_status` marginal distribution preserved: Yes
- `relationship` marginal distribution preserved: Yes
- Model accuracy drop: 1.60 percentage points

This confirms that the experiment changed the joint feature distribution while preserving both individual categorical distributions.

## 7. Dataset Validation

All eight generated datasets passed the structural validation suite.

Verified properties:

- Dataset shape: 6,000 rows × 13 columns
- Correct column names and data types
- No missing values introduced
- Valid categorical values
- Intended feature isolation
- Target-label preservation for injected scenarios
- No-drift control generated successfully

Floating-point comparisons used numerical tolerance to account for CSV serialization differences.

## 8. Generated Artifacts

**Configuration**

`component3/configs/drift_scenarios.json`

**Experiment scripts**

`component3/src/experiments/run_drift_scenarios.py`

`component3/src/experiments/test_drift_magnitudes.py`

**Validation script**

`component3/tests/test_drift_datasets.py`

**Experimental datasets**

`component3/data/drifted/S00.csv` through `S07.csv`

**Sensitivity results**

`component3/data/drifted/magnitude_sensitivity.csv`

## 9. Limitations

The experiments used an evaluation batch sampled from the model's reference dataset rather than an independent held-out test dataset.

Injected feature changes retained original target labels, which can produce unrealistic feature-label combinations. Therefore, measured performance changes represent controlled stress-test behavior and should not automatically be interpreted as real-world concept drift.

The experiment also used one fixed seed. Further repetitions and statistical analysis would be needed to establish the generality of the observed patterns.

## 10. Conclusion

Sprint 02 Task 02 successfully established a reproducible controlled drift experiment suite for Component 3.

Eight scenarios were generated and structurally validated, twelve magnitude sensitivity experiments were completed, and interaction drift was independently verified.

These artifacts provide a foundation for subsequent drift performance analysis, simulated C1/C2 evidence generation, and synthetic data repair experiments.

**Final Status: Completed — 6/6 subtasks.**
