# Sprint 05 — Task 11: Experimental Dataset Cleaning and Validation

## 1. Project Information

Project: Intelligent MLOps Platform  
Group: J26-DS-361  
Component: Component 3 — Adaptive Intelligent Synthetic Data Repair Engine  
Sprint: 05 — Validation Engine and Repair Experiments  
Task: 11 — Validate and Clean Experimental Results  

Date: October 18, 2026  

## 2. Objective

Establish an automated data quality and integrity cleaning pipeline for empirical synthetic data repair
results, producing an audited, validated dataset ready for training machine learning prescription models.

## 3. Implemented Files

- `component3/src/experiments/build_repair_dataset.py` (`clean_and_validate_dataset`)
- `component3/tests/test_repair_dataset.py`
- `component3/data/repair_experiments_clean.csv`
- `component3/data/repair_experiments_metadata.json`

## 4. Quality Rules and Integrity Checks

| Integrity Rule | Check Description | Action on Violation |
|---|---|---|
| **Primary Key Uniqueness** | Detect duplicate `experiment_id` records | Deduplicate, preserving latest run; record in audit log |
| **Metric Range Validity** | Check that `target_coverage` and `fidelity_score` $\in [0.0, 1.0]$ | Exclude corrupt row; flag in audit log |
| **Metric Non-Negativity** | Ensure runtime measurements (`total_seconds`, etc.) $\ge 0.0$ | Exclude invalid row |
| **Categorical Domain Integrity** | Ensure generator name $\in \{\text{CTGAN}, \text{TVAE}, \text{TABDDPM}\}$ | Validate against registry |
| **Label Authorization Check** | Ensure offline utility results are not claimed without authorized labels | Confirmed via governance gate |

## 5. Audit Summary Results

From `component3/data/repair_experiments_metadata.json`:
- **Raw Record Count**: 15
- **Clean Record Count**: 15
- **Duplicate Records**: 0
- **Excluded Records**: 0
- **Validation Status**: `PASSED`
- **Generators Represented**:
  - `CTGAN`: 7 experiments
  - `TVAE`: 6 experiments
  - `TABDDPM`: 2 experiments
- **Scenarios Represented**:
  - `S04` (Race categorical drift): 11 experiments
  - `S01` (Age continuous drift): 2 experiments
  - `S03` (Marital status categorical drift): 2 experiments
- **Viability Distribution (`prescriptive_viability_label`)**:
  - `1` (Viable repair candidate): 11 records (73.3%)
  - `0` (Non-viable / Degraded candidate): 4 records (26.7%)

## 6. Unit Testing Evidence

Test File: `component3/tests/test_repair_dataset.py`  
Command: `pytest component3/tests/test_repair_dataset.py -v`  
Result: 3 passed in 1.91 seconds (100% pass rate).

Test Scenarios Evaluated:
1. `test_clean_and_validate_valid_dataset`: Verifies clean pass on valid data.
2. `test_clean_and_validate_handles_duplicates`: Successfully detects duplicate IDs and deduplicates.
3. `test_clean_and_validate_filters_invalid_metrics`: Safely catches and excludes records with out-of-bounds metrics (e.g. coverage > 1.0).

## 7. Conclusion

Sprint 05 — Task 11 successfully implemented the cleaning and validation pipeline.  
The resulting dataset (`repair_experiments_clean.csv`) is verified, audited, and ready for model training.  
Proceeding to **Task 12: Document Validation and Repair Experiment Evidence**.
