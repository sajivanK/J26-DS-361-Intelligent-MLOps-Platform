# Sprint 02 — Shared Drift Injection Framework Review

## 1. Document Information

- **Research Project:** J26-DS-361 — Intelligent MLOps Platform
- **Component:** C3 — Adaptive Intelligent Synthetic Data Repair Engine
- **Sprint:** 02 — Controlled Drift & Diagnosis Simulation
- **Task:** 01 — Review Shared Drift Injection Framework
- **Original Planned Sprint:** 21–25 September 2026
- **Review Date:** 12 October 2026
- **Status:** Code review completed; runtime validation pending

## 2. Purpose

The purpose of this task is to understand the existing shared drift injection framework and establish how Component 3 can reuse it for controlled research experiments.

The framework will be used to:

1. Generate reproducible drifted datasets.
2. Measure the baseline model's performance under drift.
3. Obtain experimental ground-truth information.
4. Prepare simulated evidence for Component 1 and Component 2.
5. Support later synthetic data repair experiments.

Component 3 will reuse the framework without modifying shared implementation files.

## 3. Shared Framework Files

| File | Responsibility |
|---|---|
| `shared/injection/drift_types.py` | Low-level drift transformations |
| `shared/injection/inject_drift.py` | Drift dispatcher, input validation and performance evaluation |
| `shared/injection/validate_injection.py` | Automated framework validation checks |
| `shared/injection/examples.py` | Example usage and integration patterns |

The reviewed shared files are maintained by the team and must remain unchanged during Component 3 development.

## 4. Supported Drift Types

The `inject_drift()` dispatcher accepts exactly five identifiers.

| Identifier | Description |
|---|---|
| `marginal_single` | Changes the distribution of one selected feature |
| `marginal_multi` | Changes the distributions of multiple selected features |
| `interaction` | Disrupts relationships between selected feature pairs |
| `combined` | Applies marginal and interaction drift together |
| `none` | Samples records without deliberately injecting drift |

### 4.1 Marginal Drift

Marginal drift changes the values or distribution of individual features.

For continuous features, the framework shifts values using the feature standard deviation and configured magnitude, with additional random noise.

For categorical features, it redistributes category probabilities toward a less frequent category.

### 4.2 Interaction Drift

Interaction drift disrupts relationships between features by shuffling one feature's values among selected rows.

This preserves the individual feature values and their marginal distributions while changing their pairing with other features.

The framework supports multiple feature pairs through the `interaction_pairs` parameter of the `interaction` drift type.

### 4.3 Combined Drift

Combined drift applies marginal changes and interaction changes in the same scenario.

### 4.4 No-Drift Scenario

The `none` scenario provides a comparison batch without deliberate feature perturbation.

It is useful for checking ordinary sampling variation and false-alarm behavior.

## 5. Main Injection Interface

The reviewed function signature is:

```python
def inject_drift(
    reference_df,
    model,
    drift_type,
    features,
    magnitude,
    feature_types=None,
    interaction_pairs=None,
    batch_size=None,
    target_col="target",
    seed=42
):
```

### 5.1 Parameter Summary

| Parameter | Purpose |
|---|---|
| `reference_df` | Source dataset containing features and target |
| `model` | Baseline model used for performance evaluation |
| `drift_type` | One of the five supported drift identifiers |
| `features` | Feature names selected for injection |
| `magnitude` | Configured perturbation strength, between 0 and 1 |
| `feature_types` | Mapping of continuous and categorical features |
| `interaction_pairs` | Feature pairs for interaction drift |
| `batch_size` | Number of records in the generated batch |
| `target_col` | Target label column; default is `target` |
| `seed` | Random seed; default is 42 |

When `feature_types` is omitted, the framework treats selected features as continuous by default.

For UCI Adult, categorical features must be explicitly identified because several categorical variables are stored as integer codes.

## 6. Framework Output

The dispatcher returns a dictionary containing:

```python
{
    "drifted_df": drifted_dataframe,
    "ground_truth": ground_truth_dictionary
}
```

`drifted_df` contains the generated drift scenario.

`ground_truth` provides experimental information such as the injected drift type, affected features, performance changes, severity and malignancy classification.

Exact metadata fields should be checked against the actual output before building downstream schema mappings.

## 7. Performance Degradation and Severity

The framework measures accuracy degradation as:

```text
accuracy_drop_pct =
    (baseline_accuracy - drifted_accuracy) * 100
```

This result is measured in percentage points.

Example:

- Baseline accuracy: 87%
- Drifted accuracy: 82%
- Accuracy drop: 5 percentage points

The framework also evaluates ROC-AUC where supported by the model and target data.

### 7.1 Severity Thresholds

| Severity | Accuracy drop |
|---|---|
| None | Below 1 percentage point |
| Low | 1 to below 3 percentage points |
| Moderate | 3 to below 7 percentage points |
| High | 7 to below 15 percentage points |
| Critical | 15 percentage points or higher |

These ranges summarize the reviewed threshold configuration. Exact boundary behavior should be confirmed through runtime boundary tests if required.

### 7.2 Malignancy Label

The framework defines:

```python
is_malignant = accuracy_drop_pct > 2.0
```

A scenario is labeled malignant when the measured accuracy drop exceeds two percentage points.

This is an experimental label, not a complete repair authorization decision.

Component 3 must separately evaluate repair readiness, label trust, affected-slice suitability and data sufficiency.

### 7.3 Magnitude Versus Performance Drop

Injection magnitude and measured accuracy degradation are different quantities.

- **Magnitude:** Configured strength of the perturbation.
- **Accuracy drop:** Observed change in model performance.

A high magnitude does not guarantee a high accuracy drop.

## 8. Existing Framework Validation Checks

The shared validation module contains four reviewed checks.

### Check 1 — No-Drift Baseline

Runs multiple no-drift samples.

Purpose:

- Assess sampling variation.
- Check that no deliberate drift produces minimal distribution differences.
- Check that model performance remains relatively stable.

Reviewed criteria:

- Mean KS statistic below 0.05.
- Mean absolute accuracy change below 2 percentage points.

### Check 2 — Interaction Marginal Preservation

Tests interaction drift at magnitudes 0.3, 0.6 and 1.0.

Purpose:

- Confirm that individual feature distributions remain unchanged.
- Confirm that interaction injection operates through feature re-pairing.

Reviewed criterion:

- KS statistics for selected features below 0.05.

Note: Marginal preservation alone does not establish how much the joint relationship changed. Separate relationship measurements may be useful during experiments.

### Check 3 — Magnitude Monotonicity

Tests a range of injection magnitudes.

Purpose:

- Compare performance degradation at weaker and stronger drift settings.

Reviewed criterion:

- Average degradation at high magnitude exceeds average degradation at low magnitude.

This is a framework test expectation, not a guarantee for every feature and random seed.

### Check 4 — Benign Versus Malignant Feature Comparison

Compares drift on high-importance and low-importance features at the same magnitude.

Purpose:

- Examine whether feature importance affects performance degradation.

Reviewed criterion:

- The high-importance feature produces a larger measured drop than the low-importance feature.

This behavior must be verified experimentally rather than assumed for all datasets.

### Validation Execution Status

The validation source code has been reviewed.

The four checks have **not yet been executed or independently verified** as part of this Sprint 02 review.

Runtime testing is planned for subsequent drift experiments.

## 9. UCI Adult Resources

The shared UCI Adult resources include:

```text
shared/uci_adult/
├── data/
│   ├── reference.csv
│   └── test.csv
├── models/
│   ├── base_model_adult.pkl
│   └── preprocessor.pkl
├── preprocessing.py
├── train_base_model.py
└── feature_importance_adult.json
```

Previously verified dataset information:

| Dataset | Rows | Columns |
|---|---:|---:|
| Reference | 26,048 | 13 |
| Test | 6,513 | 13 |

The 13 columns contain 12 model features and one binary target.

Previously verified baseline model performance on the non-drifted test set:

- Accuracy: approximately 87.44%
- F1 score: approximately 0.7195
- ROC-AUC: approximately 0.9298

These test-set results must not be confused with the framework's reference-versus-drifted evaluation.

## 10. Component 3 Integration Strategy

### 10.1 Proposed Experimental Workflow

```text
Shared UCI Adult Data
        |
        v
Shared Baseline XGBoost Model
        |
        v
Shared Drift Injection Framework
        |
        +--> Drifted Dataset
        |
        +--> Experimental Ground Truth
                    |
                    v
        Simulated C1 Drift Evidence
                    |
                    v
        Simulated C2 Diagnosis
                    |
                    v
        C3 Input Contract Validation
                    |
                    v
        C3 Repair Readiness Evaluation
                    |
                    v
        Later Synthetic Repair Experiments
```

### 10.2 Integration Rules

1. Import and reuse the existing shared injection functions.
2. Do not copy the shared framework into Component 3.
3. Do not edit shared implementation files.
4. Keep experiment scripts, configurations and outputs inside Component 3.
5. Record random seeds and scenario parameters.
6. Validate generated datasets before using them.
7. Map simulated C1 and C2 evidence to the actual shared JSON contracts.
8. Do not treat experimental ground truth as a real C1/C2 prediction.
9. Keep generated experimental artifacts separate from reference and test datasets.
10. Avoid training on held-out test labels or using test data to select repair strategies.

### 10.3 Example Integration Pattern

The following is an illustrative usage pattern, not an executed test:

```python
from shared.injection.inject_drift import inject_drift

result = inject_drift(
    reference_df=reference_df,
    model=baseline_model,
    drift_type="marginal_single",
    features=["education_num"],
    magnitude=0.4,
    feature_types={"education_num": "continuous"},
    target_col="target",
    seed=42,
)

drifted_df = result["drifted_df"]
ground_truth = result["ground_truth"]
```

The import path depends on the Python execution environment and should be verified during implementation.

## 11. Research Limitations and Risks

### 11.1 Feature Perturbation With Fixed Labels

The framework modifies feature values while preserving the original target labels.

This creates controlled model stress tests, but the resulting feature-label relationships may not always represent realistic production data.

### 11.2 Synthetic Repair Is Not Automatically Safe

A generated synthetic dataset must pass structural, fidelity, utility and privacy checks before acceptance.

Drift severity alone does not prove that synthetic generation will improve the model.

### 11.3 Ground Truth Versus Production Evidence

The injection framework knows which features were deliberately modified.

Real C1 and C2 components must infer drift and affected slices from observed data.

Simulated evidence must be clearly identified and should not be reported as measured C1/C2 detection accuracy.

### 11.4 Feature Importance Is Context-Dependent

Feature importance provides useful scenario selection guidance, but the actual performance impact must be measured.

### 11.5 Reproducibility

Each experiment should record:

- Scenario identifier.
- Dataset and model versions.
- Drift type.
- Selected features.
- Magnitude.
- Batch size.
- Random seed.
- Measured metrics.
- Output artifact paths.

## 12. Planned Follow-Up Work

The remaining Sprint 02 tasks will:

1. Implement controlled drift scenarios.
2. Measure model performance under drift.
3. Save reproducible scenario datasets.
4. Construct simulated C1 evidence.
5. Construct simulated C2 diagnosis.
6. Validate input contracts.
7. Document experimental results.

## 13. Task 01 Evidence and Completion Status

| Subtask | Evidence | Status |
|---|---|---|
| 01.1 — Inspect files | Reviewed `drift_types.py` and `inject_drift.py` | Complete |
| 01.2 — Identify drift types | Confirmed five dispatcher identifiers | Complete |
| 01.3 — Review parameters | Confirmed signature, thresholds and performance calculation | Complete |
| 01.4 — Review validation/examples | Inspected four checks and usage examples | Complete — code review |
| 01.5 — Document integration | This review document | Pending save and verification |

**Task 01 final status:** Pending documentation verification.

**Next planned task:** Sprint 02 — Task 02, Implement and Test Controlled Drift Scenarios.
