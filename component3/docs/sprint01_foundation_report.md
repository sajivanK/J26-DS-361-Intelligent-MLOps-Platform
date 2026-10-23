# Sprint 01 — Project Setup & System Foundation

**Research Project:** Intelligent MLOps Platform\
**Component:** C3 — Adaptive Intelligent Synthetic Data Repair Engine\
**Git Branch:** `c3-mathusan`\
**Planned Sprint:** 16–20 September 2026\
**Documentation Date:** 12 October 2026\
**Status:** Documentation in progress

## 1. Sprint Objective

The objective of Sprint 01 was to establish the development foundation for Component 3, review shared integration resources, understand the UCI Adult preprocessing and baseline model training pipeline, and verify that the shared datasets and trained baseline model are usable for subsequent drift and synthetic-data repair experiments.

## 2. Task Progress

| Task | Description                                            | Status                           |
| ---- | ------------------------------------------------------ | -------------------------------- |
| 01   | Review Component 3 Requirements and Architecture       | Completed — previously confirmed |
| 02   | Review Team Integration Contracts                      | Completed — previously confirmed |
| 03   | Set Up GitHub Repository and C3 Branch                 | Completed — previously confirmed |
| 04   | Configure VS Code and Python Development Environment   | Completed — previously confirmed |
| 05   | Review Shared Dataset and Model Resources              | Completed — verified             |
| 06   | Study UCI Adult Preprocessing and Baseline Training    | Completed — code reviewed        |
| 07   | Verify UCI Adult Baseline Model and Datasets           | Completed — execution verified   |
| 08   | Prepare Sprint 01 Technical Documentation and Evidence | In progress                      |

## 3. Development Environment

Component 3 development is performed in VS Code using Python and a project-specific virtual environment.

The verified Git branch is `c3-mathusan`.

Component 3 code is maintained separately under `component3/`, while shared datasets, trained models, integration contracts, and drift injection utilities are available under `shared/`.

## 4. Shared Resource Verification

The following UCI Adult resources were verified as available locally:

- `shared/uci_adult/data/reference.csv`
- `shared/uci_adult/data/test.csv`
- `shared/uci_adult/models/base_model_adult.pkl`
- `shared/uci_adult/models/preprocessor.pkl`
- `shared/uci_adult/feature_importance_adult.json`
- `shared/uci_adult/preprocessing.py`
- `shared/uci_adult/train_base_model.py`

Additional shared resources include:

- `shared/contracts/` — component integration schemas
- `shared/injection/` — controlled drift injection utilities
- `shared/ieee_cis/` — IEEE-CIS preprocessing and baseline training resources

The shared files were obtained from the team's Google Drive resources.

## 5. UCI Adult Dataset Verification

| Property                | Reference Dataset | Test Dataset |
| ----------------------- | ----------------- | ------------ |
| Records                 | 26,048            | 6,513        |
| Input features          | 12                | 12           |
| Target columns          | 1                 | 1            |
| Total columns           | 13                | 13           |
| Missing values detected | 0                 | 0            |

**Total records:** 32,561

The five numerical features are `age`, `education_num`, `capital_gain`, `capital_loss`, and `hours_per_week`.

The seven categorical features are `workclass`, `marital_status`, `occupation`, `relationship`, `race`, `sex`, and `native_country`.

The binary target column is `target`.

## 6. Preprocessing Pipeline Review

The shared preprocessing implementation was reviewed.

The pipeline cleans raw values, handles missing-value tokens, removes selected redundant features, converts income labels into binary targets, and divides the dataset into reference and test partitions.

The numerical features are transformed using `StandardScaler`, while categorical features are converted to integer representations using `LabelEncoder`.

The preprocessing transformations are fitted on the reference partition and applied to the test partition.

The fitted preprocessor is stored in `preprocessor.pkl`.

The saved CSV files contain already-transformed feature values and should not be treated as original raw Adult records.

## 7. Baseline Model Training Review

The shared training implementation uses `XGBClassifier` with the following configuration:

| Parameter        | Value |
| ---------------- | ----- |
| n_estimators     | 200   |
| max_depth        | 6     |
| learning_rate    | 0.1   |
| subsample        | 0.8   |
| colsample_bytree | 0.8   |
| random_state     | 42    |

The model is trained using the reference dataset and evaluated using the test dataset.

SHAP feature importance is calculated to identify influential input features for later drift experiments.

## 8. Saved Baseline Model Verification

The saved `base_model_adult.pkl` model was loaded and evaluated against the shared `test.csv`.

| Metric               | Verified Result |
| -------------------- | --------------- |
| Test records         | 6,513           |
| Input features       | 12              |
| Model class          | XGBClassifier   |
| Accuracy             | 0.8744          |
| F1 Score             | 0.7195          |
| ROC-AUC              | 0.9298          |
| Prediction execution | Successful      |

The verified accuracy and ROC-AUC matched the values reported in the shared feature-importance metadata.

These results provide the **original, non-drifted baseline** for later controlled-drift and repair experiments.

## 9. Relevance to Component 3

The verified datasets and baseline model will support subsequent Component 3 experiments.

The reference dataset provides the original training distribution, while the saved test dataset provides a baseline evaluation resource.

The shared drift injection utilities will support controlled experiments involving changes in feature distributions.

Component 3 will later evaluate whether targeted synthetic-data repair improves performance on affected slices while preserving overall and non-affected performance.

Synthetic data must pass structural, fidelity, utility, and privacy checks before being accepted for repair.

## 10. Evidence and Limitations

Evidence obtained during Sprint 01 verification includes:

- Git branch verification
- Local shared resource inventory
- Reference and test dataset inspection
- Review of preprocessing and baseline training implementations
- Successful saved baseline model evaluation
- Agreement between evaluated baseline metrics and recorded metadata

Tasks 01–04 were previously confirmed complete; their original setup evidence should be retained separately where available.

This report documents verification performed after the original planned Sprint 01 dates. It does not claim that all verification was performed during 16–20 September 2026.

## 11. Sprint Conclusion

The Component 3 development foundation and required shared UCI Adult resources have been established and verified.

The saved baseline model successfully processes the shared test dataset and provides reproducible performance measurements.

The next planned work is Sprint 02 — Controlled Drift & Diagnosis Simulation.
