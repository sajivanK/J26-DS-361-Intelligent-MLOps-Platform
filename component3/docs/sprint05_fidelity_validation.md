# Sprint 05 — Task 03: Statistical Fidelity and Diversity Validation

## 1. Project Information

Project: Intelligent MLOps Platform  
Group: J26-DS-361  
Component: Component 3 — Adaptive Intelligent Synthetic Data Repair Engine  
Sprint: 05 — Validation Engine and Repair Experiments  
Task: 03 — Implement Fidelity Validation  

Date: October 18, 2026  

## 2. Objective

Design, implement, and empirically evaluate a statistical fidelity and diversity validation
engine for mixed-type synthetic tabular data.

The module evaluates how closely generated synthetic records preserve the statistical
properties, univariate distributions, multivariate dependencies, and diversity of the
real affected training slice without mode collapse or excessive record duplication.

## 3. Implemented Files

- `component3/src/validation/fidelity_validator.py`
- `component3/src/validation/__init__.py`
- `component3/tests/test_fidelity_validator.py`
- `component3/src/experiments/run_fidelity_validation_s04.py`
- `component3/results/fidelity_validation/S04_ctgan_fidelity_e1_n100_seed42.json`

## 4. Reused Components and Standards

- `component3/src/validation/validation_models.py` (`ValidationStatus`, `ValidationStageResult`)
- `component3/src/generators/data_preparer.py` (`PreparedGeneratorData`)
- `component3/src/generators/ctgan_generator.py` (`CTGANGenerator`)
- `shared/contracts/c3_output_schema.json` (`fidelity_score` contract specification)

## 5. Statistical Methodology

### 5.1 Continuous Feature Evaluation
For each declared continuous feature:
- Absolute differences in means and standard deviations.
- Two-sample Kolmogorov-Smirnov (KS) test (`scipy.stats.ks_2samp`) measuring maximum distance between empirical CDFs.
- Normalized Wasserstein distance (`scipy.stats.wasserstein_distance` divided by reference standard deviation).
- Quantile differences at the 25th, 50th, and 75th percentiles.
- Boundary concentration measuring accumulation of values at extreme reference limits.

### 5.2 Categorical Feature Evaluation
For each declared categorical feature:
- Total Variation Distance (TVD): $0.5 \times \sum_{k} |p_k - q_k|$.
- Jensen-Shannon (JS) divergence: squared Jensen-Shannon distance (`scipy.spatial.distance.jensenshannon` with base 2).
- Category domain coverage and detection of unseen categories.

### 5.3 Multivariate Dependence
- Continuous correlation matrix difference: mean absolute difference, maximum absolute difference, and Frobenius norm of $(Corr_{syn} - Corr_{ref})$.

### 5.4 Diversity & Mode Collapse Indicators
- Duplicate record count and duplicate ratio ($\frac{n_{duplicates}}{n_{total}}$).
- Unique record ratio.
- Configurable maximum allowable duplicate ratio (provisional default: 0.15).

### 5.5 Composite Fidelity Score
A bounded composite metric $[0.0, 1.0]$ computed as $1 - \text{mean}(\text{KS}, \text{JS}, \text{CorrDiff})$, aligning with the `fidelity_score` expected by the C3/C4 contract.

## 6. Unit Testing Evidence

Test File: `component3/tests/test_fidelity_validator.py`  
Command: `pytest component3/tests/test_fidelity_validator.py -v`  
Result: 9 passed in 4.85 seconds (100% pass rate).

Test Scenarios Evaluated:
1. `test_identical_distributions_pass`: Synthetic and reference from identical distributions return `PASS` with high fidelity score.
2. `test_strong_distribution_shift_fails`: Distorted distributions trigger `FAIL` with explicit reason logging.
3. `test_duplicate_heavy_synthetic_data_fails`: Synthetics with 90% duplicated rows fail the diversity threshold.
4. `test_empty_datasets`: Empty synthetic or reference inputs are rejected safely.
5. `test_missing_columns`: Schema column mismatch returns `FAIL`.
6. `test_constant_columns`: Features with zero variance are handled gracefully without division-by-zero crashes.
7. `test_input_immutability`: Input DataFrames remain strictly unmodified.
8. `test_validator_class_interface`: OOP wrapper executes correctly and outputs valid dictionaries.
9. `test_invalid_thresholds`: Out-of-bounds or invalid threshold parameters raise `ValueError`.

## 7. Real S04 Controlled Experiment Evidence

Scenario: S04 (`race == 2`), 360 affected training records, 100 generated records via CTGAN (1 epoch, seed 42).  
Experiment Script: `component3/src/experiments/run_fidelity_validation_s04.py`  
Evidence File: `component3/results/fidelity_validation/S04_ctgan_fidelity_e1_n100_seed42.json` (verified, 4,370 bytes).

### Experimental Results:
- **Validation Status**: `FAIL`
- **Rejection Reason**: `Mean KS statistic (0.407) exceeds threshold (0.350)`
- **Composite Fidelity Score**: `0.7688`
- **Continuous Metrics Summary**:
  - `mean_ks_statistic`: `0.4067`
  - `mean_normalized_wasserstein`: `1.6659`
- **Categorical Metrics Summary**:
  - `mean_js_divergence`: `0.1938` (PASS under 0.350 threshold)
  - `mean_total_variation_distance`: `0.3670`
- **Multivariate Correlation Difference**:
  - `mean_absolute_difference`: `0.0931`
  - `frobenius_norm`: `0.6758`
- **Diversity**:
  - `duplicate_count`: `0`
  - `duplicate_ratio`: `0.00%`
  - `unique_record_ratio`: `100.0%`

## 8. Research Interpretation

In Task 02, the same 1-epoch CTGAN output achieved 100% target coverage and passed structural validation.  
However, the fidelity validation engine revealed that 1 training epoch is insufficient for continuous features (`mean_ks = 0.407 > 0.350`), correctly rejecting the sample.  
This demonstrates that high target coverage does not imply statistical fidelity and validates the need for independent validation stages.

## 9. Research Limitations

- Thresholds (0.35 KS, 0.35 JS, 0.15 Duplicate ratio) are provisional and require downstream empirical calibration across repair quantities.
- Categorical association (Cramer's V) across high-cardinality discrete pairs is computationally omitted in favor of univariate TVD and JS divergence.
- Only a single 1-epoch smoke test was conducted for Task 03 verification; multi-epoch and multi-generator comparisons are scheduled for Tasks 07–09.

## 10. Conclusion

Sprint 05 — Task 03 is fully implemented, verified with 9 unit tests, and demonstrated on real S04 CTGAN data.  
Proceeding to **Task 04: Implement Privacy Validation**.
