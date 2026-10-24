# Sprint 03 — Configuration and Data Loading

**Project:** J26-DS-361 — Intelligent MLOps Platform\
**Component:** C3 — Adaptive Intelligent Synthetic Data Repair Engine\
**Task:** Sprint 03 — Task 02\
**Implementation Date:** 13 October 2026\
**Status:** Implementation and testing completed; documentation verification pending

## 1. Objective

Develop reusable configuration, dataset, evidence, and model-loading modules for Component 3 while preserving the existing Sprint 02 experiments and shared project resources.

## 2. Configuration Management

Created `component3/configs/c3_config.json`.

The configuration includes:

- UCI Adult dataset information and paths.
- Target column and feature types.
- Baseline model and preprocessor paths.
- C1 evidence and C2 diagnosis directories.
- Official C1/C2 contract locations.
- Initial repair-readiness settings.
- Logging level and random seed.

The existing `drift_scenarios.json` was preserved for reproducibility.

Readiness thresholds are preliminary engineering defaults and have not yet been scientifically validated.

## 3. Implemented Python Modules

### Configuration Loader

**File:** `component3/src/core/config_loader.py`

Implemented JSON configuration loading, required-section checks, configuration value retrieval, project-relative path resolution, and basic error handling.

### Data Loader

**File:** `component3/src/core/data_loader.py`

Implemented loading for reference datasets, current scenario datasets, simulated C1 evidence, simulated C2 diagnosis, and official integration contracts.

The loader performs basic checks for missing files, empty datasets, duplicate columns, and invalid JSON.

### Model Loader

**File:** `component3/src/core/model_loader.py`

Implemented loading for the existing trained model and fitted preprocessor.

Verified artifact types:

- Baseline model: `XGBClassifier`
- Preprocessor: `UCIAdultPreprocessor`

The loader requires explicit `trusted=True` approval before deserializing pickle artifacts. This approval flag does not sandbox deserialization; only trusted artifacts should be loaded.

No model retraining or modification was performed.

## 4. Dataset and Evidence Verification

Successfully loaded:

| Resource                 | Verified Result          |
| ------------------------ | ------------------------ |
| Reference dataset        | 26,048 rows × 13 columns |
| Current drift datasets   | 8 scenarios, S00–S07     |
| Each current dataset     | 6,000 rows × 13 columns  |
| Simulated C1 evidence    | 8 JSON files             |
| Simulated C2 diagnosis   | 8 JSON files             |
| Official C1/C2 contracts | Both loaded              |
| Baseline model           | Successfully loaded      |
| Fitted preprocessor      | Successfully loaded      |

The C1/C2 evidence is simulated research data, not output from the final production components.

## 5. Failure-Handling Verification

Seven manual negative checks passed, covering missing configuration files, empty paths, missing CSV and JSON files, unsupported scenario IDs, unapproved pickle loading, and missing trusted model files.

**Result: 7/7 PASS**

## 6. Automated Testing

Created `component3/tests/test_core_loaders.py`.

The automated loader test suite completed successfully:

**36/36 tests passed.**

A combined regression test was then performed with the existing input contract tests.

| Test Suite                 | Passed |
| -------------------------- | ------ |
| Core loader tests          | 36     |
| C1/C2 input contract tests | 18     |
| **Total**                  | **54** |

**Final regression result:** 54 passed, 3 warnings, 0 failures, execution time 3.82 seconds.

The warnings have not yet been investigated.

## 7. Architecture and Safety Decisions

- Reused existing datasets, trained model, preprocessor, and contract files.
- Kept Sprint 02 experiment configuration separate from central C3 configuration.
- Resolved project-relative paths consistently.
- Added basic loading errors to support future pipeline failure handling.
- Required explicit trust approval for pickle deserialization.
- Restricted benchmark scenario loading to S00–S07.
- Avoided retraining or changing shared model artifacts.

## 8. Current Limitations

The loaders currently perform basic structural checks, not full input schema validation.

The C1/C2 payloads are simulated. Their scientific reliability is not established by successful loading.

The repair-readiness thresholds require further design and evaluation.

Real C1/C2 integration, stronger input validation, safe slice-rule parsing, and repair-readiness decisions remain future tasks.

## 9. Evidence Files

- `component3/configs/c3_config.json`
- `component3/src/core/config_loader.py`
- `component3/src/core/data_loader.py`
- `component3/src/core/model_loader.py`
- `component3/tests/test_core_loaders.py`
- `component3/tests/test_input_contracts.py`
- `component3/configs/drift_scenarios.json`

## 10. Conclusion

The configuration and data-loading foundation has been implemented and tested successfully. Component 3 can load the existing research datasets, simulated integration evidence, official contracts, and trusted baseline model artifacts.

The next development task is **Sprint 03 — Task 03: Implement Input Schema Validation**.
