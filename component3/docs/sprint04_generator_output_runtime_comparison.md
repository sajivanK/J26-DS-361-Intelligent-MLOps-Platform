# Sprint 04 — Task 10: Generator Output and Runtime Comparison

## 1. Task Overview

**Project:** Intelligent MLOps Platform  
**Component:** Component 3 — Adaptive Intelligent Synthetic Data Repair Engine  
**Sprint:** Sprint 04 — Synthetic Data Generator Implementation  
**Task:** 10 — Compare Generator Output and Runtime  
**Planned sprint period:** October 1–5, 2026  
**Actual implementation date:** October 17, 2026  
**Status:** Implementation and testing completed; Git delivery pending.

## 2. Objective

Compare CTGAN, TVAE, and TabDDPM using previously recorded controlled-drift experiments.

The comparison focuses on:

- Target-slice coverage
- Generated record counts
- Missing values
- Training runtime
- Sampling runtime where recorded
- Observed numerical-range violations where recorded

The objective is to produce a reproducible descriptive comparison, not to establish which generator has the highest overall synthetic-data quality.

## 3. Existing Experimental Evidence

The comparison uses 63 existing JSON experiment reports:

| Generator | Experiments |
|---|---:|
| CTGAN | 21 |
| TVAE | 21 |
| TabDDPM | 21 |
| **Total** | **63** |

Each generator was evaluated across seven controlled drift scenarios (S01–S07) using three random seeds:

- 42
- 123
- 456

Each experiment generated 100 synthetic records.

All experiments used one training epoch.

TabDDPM additionally used ten diffusion timesteps and a batch size of 128.

## 4. Experiment Data Sources

The comparison reads:

- `component3/results/ctgan_targeted/`
- `component3/results/tvae_targeted/`
- `component3/results/tabddpm_targeted/`

The existing JSON reports contain experiment metadata and diagnostic measurements.

The comparison does not independently re-evaluate raw synthetic records because raw generated datasets were not retained in these experiment artifacts.

## 5. Comparison Engine Implementation

**Implementation file:**

`component3/src/experiments/compare_generators.py`

The comparison engine:

1. Loads existing JSON experiment reports.
2. Checks experiment identities and expected combinations.
3. Validates numerical values and target-coverage arithmetic.
4. Checks experimental metadata.
5. Normalizes measurements into a common tabular structure.
6. Calculates generator-level summaries.
7. Calculates scenario-level summaries.
8. Exports CSV files and a validation JSON report.

No generator training or synthetic-data sampling occurs during comparison.

## 6. Common and Generator-Specific Metrics

### Common metrics

All three generator experiment reports record:

- Scenario
- Seed
- Training rows
- Generated rows
- Matching and non-matching records
- Target-slice coverage
- Training seconds
- Missing values

### Generator-specific metrics

Only TabDDPM reports contain:

- Sampling seconds
- Diffusion timesteps
- Batch size
- Observed numerical-range diagnostics

Unrecorded CTGAN and TVAE measurements are represented as missing values rather than zero.

## 7. Aggregate Experimental Results

| Metric | CTGAN | TVAE | TabDDPM |
|---|---:|---:|---:|
| Experiments | 21 | 21 | 21 |
| Generated records | 2,100 | 2,100 | 2,100 |
| Target-matching records | 2,066 | 2,099 | 2,100 |
| Non-matching records | 34 | 1 | 0 |
| Aggregate target coverage | 98.38% | 99.95% | 100.00% |
| Mean training seconds | 12.1822 | 10.5620 | 4.0694 |
| Mean sampling seconds | Not recorded | Not recorded | 0.0852 |
| Observed-range violations | Not recorded | Not recorded | 0 |

The coverage percentage is calculated as:

`matching_rows / generated_rows × 100`

All three generators produced 2,100 records across their experiments.

## 8. Target Coverage Analysis

CTGAN produced 2,066 target-matching records from 2,100 generated records.

TVAE produced 2,099 target-matching records from 2,100 generated records.

TabDDPM produced 2,100 target-matching records from 2,100 generated records.

TabDDPM achieved the highest observed target coverage in these experiments.

However, target coverage only measures whether generated records satisfy the specified affected-slice conditions.

It does not establish:

- Statistical fidelity
- Feature diversity
- Privacy protection
- Downstream predictive utility
- Model performance recovery

## 9. Runtime Analysis

Recorded mean training times:

- CTGAN: 12.1822 seconds
- TVAE: 10.5620 seconds
- TabDDPM: 4.0694 seconds

TabDDPM had the lowest recorded mean training time in this experiment set.

However, these results are descriptive.

The models have different internal training algorithms and hyperparameters.

One epoch does not represent equivalent computational work or equivalent training quality across generator families.

The experiments were not designed as a controlled equal-quality runtime benchmark.

Therefore, the observed runtime differences must not be interpreted as proof of general computational superiority.

## 10. Sampling Time Availability

TabDDPM recorded a mean sampling time of approximately 0.0852 seconds per experiment.

Separate sampling times were not recorded for CTGAN or TVAE.

Therefore, a three-generator sampling-speed ranking cannot be produced from the existing evidence.

Missing sampling-time measurements must not be treated as zero.

## 11. Numerical-Range Diagnostics

TabDDPM recorded zero generated values outside the observed training ranges in the 21 experiments.

Equivalent automated numerical-range diagnostics were not recorded for CTGAN and TVAE in this experiment set.

This prevents a fully comparable three-generator numerical-range analysis.

Additionally, a generated value within the observed training range is not necessarily statistically realistic or useful.

Previous TabDDPM smoke tests identified numerical boundary concentration.

Consequently, zero observed-range violations do not establish high fidelity or sufficient diversity.

## 12. Comparison Output Artifacts

The engine generated four output files:

1. `component3/results/generator_comparison/generator_comparison_detailed.csv`
2. `component3/results/generator_comparison/generator_comparison_summary.csv`
3. `component3/results/generator_comparison/generator_comparison_scenario.csv`
4. `component3/results/generator_comparison/generator_comparison_validation.json`

The detailed CSV contains 63 experiment rows.

The summary CSV contains three generator-level rows.

The scenario CSV contains 21 generator-scenario rows.

The validation JSON records the comparison evidence checks and limitations.

## 13. Automated Comparison Engine Testing

**Test file:**

`component3/tests/test_generator_comparison.py`

The test suite contains 12 automated tests covering:

- Numeric value validation
- Invalid and non-finite values
- Complete experiment loading
- Missing directories
- Incomplete experiment matrices
- Corrupted JSON files
- Generator-level aggregation
- Scenario-level aggregation
- Output artifact creation
- Validation report contents

**Test result:**

- 12 passed
- 0 failed
- Execution time: 1.42 seconds

## 14. Full Component 3 Regression Testing

Command:

`.\.venv\Scripts\python.exe -m pytest component3/tests/ -q`

Results:

- 361 passed
- 0 failed
- 4 non-failing warnings
- Execution time: 12.93 seconds

Previous baseline: 349 passing tests.

New comparison tests: 12.

The regression suite confirms that the new comparison implementation did not break the functionality covered by the existing Component 3 tests.

## 15. Research Findings

The experiment results support the following observations:

1. All three generators successfully produced targeted synthetic records in the tested controlled-drift scenarios.
2. CTGAN achieved 98.38% aggregate target coverage.
3. TVAE achieved 99.95% aggregate target coverage.
4. TabDDPM achieved 100.00% aggregate target coverage.
5. TabDDPM had the lowest recorded mean training time in this specific experiment set.
6. Sampling-time and numerical-range measurements were not consistently available across all three generators.
7. The comparison pipeline successfully processed and summarized all 63 existing experiment reports.

These are preliminary descriptive findings.

## 16. Research Limitations

The comparison has important limitations:

- All experiments used only one training epoch.
- The generators have different training procedures and hyperparameters.
- The experiments used simulated drift scenarios.
- Readiness decisions were synthetic experiment decisions.
- Generated data was exploratory, not approved for supervised retraining.
- No independent fidelity evaluation was performed.
- No downstream utility evaluation was performed.
- No privacy validation was performed.
- No statistically controlled equal-quality runtime benchmark was performed.
- Raw generated datasets were not retained for independent sample-level auditing.
- TabDDPM boundary concentration remains a potential synthetic-data quality problem.

These limitations must be addressed in subsequent validation and research experiments.

## 17. Contribution to the Research System

The comparison engine provides a reusable evidence-processing component for Component 3.

It supports:

- Consistent comparison of generator experiment records
- Automated evidence completeness checks
- Reproducible aggregate metrics
- Scenario-level result analysis
- Explicit handling of unrecorded measurements
- Technical evidence for subsequent validation work

This work supports future intelligent generator selection but does not itself implement a learned recommendation mechanism.

## 18. Evidence and Deliverables

**Source code:**

`component3/src/experiments/compare_generators.py`

**Automated tests:**

`component3/tests/test_generator_comparison.py`

**Experimental outputs:**

`component3/results/generator_comparison/`

**Technical report:**

`component3/docs/sprint04_generator_output_runtime_comparison.md`

**Verification:**

- Python compilation succeeded.
- 63/63 experiment reports processed.
- Four comparison artifacts generated.
- 12/12 comparison tests passed.
- 361/361 Component 3 tests passed.

## 19. Conclusion

Sprint 04 Task 10 successfully implemented and tested a descriptive comparison engine for CTGAN, TVAE, and TabDDPM.

The recorded experiments showed aggregate target coverage of 98.38%, 99.95%, and 100.00%, respectively.

TabDDPM also had the lowest recorded mean training time in this particular experiment set.

However, these measurements do not establish overall generator superiority.

Fidelity, privacy, utility, diversity, and downstream model recovery require further evaluation.

**Task 10 remains pending documentation verification and Git commit/push.**
