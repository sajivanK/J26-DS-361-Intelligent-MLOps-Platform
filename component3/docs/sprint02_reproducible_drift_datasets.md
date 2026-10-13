# Sprint 02 — Task 04: Reproducible Drift Scenario Dataset

**Research Project:** Intelligent MLOps Platform\
**Component:** C3 — Adaptive Intelligent Synthetic Data Repair Engine\
**Task:** Create Reproducible Drift Scenario Dataset\
**Original Sprint Schedule:** 21–25 September 2026\
**Actual Work Date:** 12 October 2026\
**Status:** Implementation and verification completed; documentation file verification pending.

## 1. Objective

The objective of this task was to prepare a reproducible, traceable, and verified collection of controlled drift datasets for Component 3 research experiments.

The datasets will be reused during later tasks to simulate drift detection, affected-slice diagnosis, synthetic data repair, and model performance evaluation.

## 2. Experimental Environment

- Dataset: Preprocessed UCI Adult
- Baseline model: XGBoost classifier
- Reference dataset: 26,048 records
- Evaluation batch: 6,000 records
- Dataset columns: 12 input features and 1 target
- Random seed: 42
- Number of drift scenarios: 8
- Drift injection framework: Existing shared team implementation

All scenario datasets were generated using the same configuration and controlled random seed.

## 3. Drift Scenario Dataset Collection

| Scenario | Drift Type                  | Affected Features                               | Magnitude |
| -------- | --------------------------- | ----------------------------------------------- | --------- |
| S00      | No drift                    | None                                            | 0.0       |
| S01      | Numerical marginal          | age                                             | 0.4       |
| S02      | Numerical marginal          | capital_loss                                    | 0.4       |
| S03      | Categorical marginal        | marital_status                                  | 0.4       |
| S04      | Categorical marginal        | race                                            | 0.4       |
| S05      | Multiple numerical marginal | age, education_num                              | 0.6       |
| S06      | Feature interaction         | marital_status, relationship                    | 0.6       |
| S07      | Combined drift              | age and marital_status–relationship interaction | 0.6       |

Each scenario dataset contains 6,000 records and 13 columns.

The datasets were saved as `S00.csv` through `S07.csv`.

## 4. Dataset Manifest

A dataset manifest was generated to record the provenance and structure of the experiment artifacts.

**Manifest file:**

`component3/data/drifted/drift_dataset_manifest.json`

The manifest records:

- Scenario identifiers and drift configurations
- Dataset paths and file sizes
- SHA-256 checksums
- Dataset dimensions
- Feature names and data types
- Target class distributions
- Reference dataset and model information
- Experiment configuration and random seed
- Performance evaluation and magnitude sensitivity result files

The manifest was generated using:

`component3/src/experiments/create_drift_manifest.py`

## 5. Data Integrity Validation

The recorded artifact information was compared against the actual files.

Validation covered:

1. File existence
2. Exact file size
3. SHA-256 checksum
4. Dataset dimensions
5. Feature column order
6. Target class distributions
7. Scenario identifier completeness

**Result:** All eight scenario datasets passed every performed integrity check.

The configuration file, reference dataset, baseline model, and both experiment result files also passed their file integrity checks.

## 6. Reproducibility Testing

The existing drift generation runner was updated to accept an optional output directory.

**Runner:**

`component3/src/experiments/run_drift_scenarios.py`

**New command-line option:**

`--output-dir`

This allows generated datasets to be saved separately without overwriting the original research artifacts.

The original eight scenarios were regenerated using the same reference data, model, configuration, and random seed.

**Reproduction command:**

```
.\.venv\Scripts\python.exe component3/src/experiments/run_drift_scenarios.py --output-dir component3/data/temp_repro
```

The regenerated datasets were compared with the original SHA-256 checksums stored in the manifest.

| Scenario | Reproducibility Result |
| -------- | ---------------------- |
| S00      | PASS                   |
| S01      | PASS                   |
| S02      | PASS                   |
| S03      | PASS                   |
| S04      | PASS                   |
| S05      | PASS                   |
| S06      | PASS                   |
| S07      | PASS                   |

**Final result: 8/8 PASS, 0 failures.**

All eight regenerated datasets were byte-for-byte identical to their corresponding original datasets.

## 7. Temporary Data Cleanup

After reproducibility testing, the temporary output directory was removed.

**Temporary directory:**

`component3/data/temp_repro/`

The following conditions were verified:

- Temporary folder no longer exists.
- Original dataset manifest still exists.
- All eight original scenario CSV files remain available.

The original research datasets were preserved.

## 8. Relevance to Component 3

These reproducible drift datasets provide controlled experimental inputs for later Component 3 tasks.

They can be used to:

- Simulate Component 1 drift detection evidence
- Simulate Component 2 affected-slice diagnosis
- Test repair readiness and data sufficiency
- Generate targeted synthetic repair candidates
- Compare CTGAN, TVAE, and TabDDPM
- Evaluate repair effectiveness
- Compare different repair quantities
- Train and evaluate intelligent repair prescription strategies

The same controlled scenarios can be reused to make experiments more consistent and easier to compare.

## 9. Research Limitations

Byte-for-byte reproducibility was verified in the current development environment.

The experiment has not yet established reproducibility across different operating systems, Python versions, or dependency versions.

The manifest records data and model checksums, but does not yet capture every source-code revision and installed dependency version.

The datasets represent controlled synthetic drift injections, not naturally occurring production drift.

Original target labels were retained during drift injection. These labels must not automatically be considered trustworthy for downstream repair training.

The reference-batch experiments should not be presented as independent held-out performance evaluations.

## 10. Research Artifacts

**Scenario configuration**

`component3/configs/drift_scenarios.json`

**Scenario generation script**

`component3/src/experiments/run_drift_scenarios.py`

**Manifest generation script**

`component3/src/experiments/create_drift_manifest.py`

**Dataset manifest**

`component3/data/drifted/drift_dataset_manifest.json`

**Scenario datasets**

`component3/data/drifted/S00.csv` through `S07.csv`

**Model performance results**

`component3/data/drifted/drift_performance_metrics.csv`

**Magnitude sensitivity results**

`component3/data/drifted/magnitude_sensitivity.csv`

**Related documentation**

`component3/docs/sprint02_controlled_drift_experiments.md`

`component3/docs/sprint02_drift_performance_evaluation.md`

## 11. Conclusion

Sprint 02 Task 04 established a reproducible and traceable collection of eight controlled drift scenario datasets.

All scenario files were indexed in a SHA-256 dataset manifest, validated against their recorded metadata, and independently regenerated in a separate output directory.

The regenerated files matched the originals byte-for-byte in all eight cases.

These artifacts provide a verified experimental foundation for the next stages of Component 3 development.

**Final Status:** Technical work completed; documentation file verification pending.
