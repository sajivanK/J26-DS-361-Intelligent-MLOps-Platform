# Sprint 04 — Task 04: Test CTGAN Targeted Data Generation

**Research Project:** J26-DS-361 — Intelligent MLOps Platform  
**Component:** C3 — Adaptive Intelligent Synthetic Data Repair Engine  
**Sprint:** 04 — Synthetic Data Generator Implementation  
**Task:** 04 — Test CTGAN Targeted Data Generation  
**Execution Date:** October 13, 2026  
**Status:** Experiments completed and verified; regression suite pending system restart  

---

## 1. Objective

Systematically evaluate the targeted generation capabilities of the metadata-aware CTGAN generator when trained on diagnosed affected slices across multiple drift scenarios and random seeds.

Key research questions:
1. Does CTGAN successfully concentrate its generated distribution within the diagnosed affected slice?
2. Does target coverage vary between discrete (categorical) versus continuous boundary conditions?
3. How much stochastic variation in target coverage occurs across independent random seeds?
4. What architectural safeguards (e.g., post-generation target coverage evaluation and filtering) are necessary before synthetic data can be considered for repair?

---

## 2. Architecture and Experiment Workflow

The experiment pipeline connects the Component 3 modules in an end-to-end automated workflow:

```
Drift Scenario (S01–S07)
         │
         ▼
Affected Slice Extraction (safe_slice_evaluator.py)
         │
         ▼
Generator Data Preparation (data_preparer.py)
         │
         ▼
Metadata-Aware CTGAN Training & Sampling (ctgan_generator.py)
         │
         ▼
Target-Slice Coverage Evaluation (target_coverage.py)
         │
         ▼
Structured JSON Persistence (component3/results/ctgan_targeted/)
```

### Core Modules Implemented and Reused
- **Safe Slice Evaluator** (`component3/src/contracts/safe_slice_evaluator.py`): Restricted AST-based evaluation of C2 slice expressions without `eval()`, `exec()`, or `query()`.
- **Generator Data Preparer** (`component3/src/generators/data_preparer.py`): Enforces 12-feature schema order, categorical/continuous typing, and label safety gating (`unsupervised_exploratory` mode).
- **CTGAN Generator** (`component3/src/generators/ctgan_generator.py`): Metadata-aware CTGAN adapter handling explicit discrete column specifications.
- **Target Coverage Evaluator** (`component3/src/generators/target_coverage.py`): Calculates `total_rows`, `matching_rows`, `non_matching_rows`, and `coverage` ratio.
- **Automated Experiment Runner** (`component3/src/experiments/run_ctgan_targeted.py`): CLI runner automating multi-scenario, multi-seed training, sampling, and structured artifact persistence.

---

## 3. Evaluated Drift Scenarios and Rules

All seven drift scenarios (S01–S07) from the verified Component 2 diagnosis contract were evaluated:

| Scenario | Drift Category | Diagnosed Affected Slice Rule | Condition Type |
|---|---|---|---|
| **S01** | Marginal (Single) | `age > 0.69` | Continuous threshold |
| **S02** | Marginal (Single) | `capital_loss > -0.20` | Continuous threshold |
| **S03** | Marginal (Single) | `marital_status == 2` | Discrete equality |
| **S04** | Marginal (Single) | `race == 2` | Discrete equality |
| **S05** | Marginal (Multi) | `age > 0.69 and education_num < -0.03` | Continuous compound (AND) |
| **S06** | Interaction | `marital_status == 2 and relationship == 0` | Discrete compound (AND) |
| **S07** | Combined | `age > 0.69 and marital_status == 2` | Hybrid (Continuous + Discrete) |

---

## 4. Comprehensive Experimental Results (21 Runs)

Each scenario was executed across three independent seeds (`42`, `123`, `456`) with 1 training epoch and 100 generated synthetic records per run (2,100 total records generated).

| Scenario | Seed | Training Rows | Generated | Matching Rows | Coverage | Training Duration (s) | Missing Values |
|---|---|---|---|---|---|---|---|
| **S01** | 42 | 2,159 | 100 | 98 | 98.00% | 22.59 | 0 |
| **S01** | 123 | 2,159 | 100 | 99 | 99.00% | 11.13 | 0 |
| **S01** | 456 | 2,159 | 100 | 98 | 98.00% | 10.22 | 0 |
| **S02** | 42 | 6,000 | 100 | 100 | 100.00% | 27.56 | 0 |
| **S02** | 123 | 6,000 | 100 | 100 | 100.00% | 13.54 | 0 |
| **S02** | 456 | 6,000 | 100 | 100 | 100.00% | 14.14 | 0 |
| **S03** | 42 | 1,723 | 100 | 100 | 100.00% | 19.15 | 0 |
| **S03** | 123 | 1,723 | 100 | 100 | 100.00% | 12.71 | 0 |
| **S03** | 456 | 1,723 | 100 | 100 | 100.00% | 12.83 | 0 |
| **S04** | 42 | 360 | 100 | 100 | 100.00% | 4.67 | 0 |
| **S04** | 123 | 360 | 100 | 100 | 100.00% | 2.04 | 0 |
| **S04** | 456 | 360 | 100 | 100 | 100.00% | 1.97 | 0 |
| **S05** | 42 | 342 | 100 | 93 | 93.00% | 4.40 | 0 |
| **S05** | 123 | 342 | 100 | 89 | 89.00% | 1.82 | 0 |
| **S05** | 456 | 342 | 100 | 99 | 99.00% | 2.16 | 0 |
| **S06** | 42 | 1,674 | 100 | 100 | 100.00% | 19.88 | 0 |
| **S06** | 123 | 1,674 | 100 | 100 | 100.00% | 13.88 | 0 |
| **S06** | 456 | 1,674 | 100 | 100 | 100.00% | 14.05 | 0 |
| **S07** | 42 | 1,550 | 100 | 96 | 96.00% | 17.58 | 0 |
| **S07** | 123 | 1,550 | 100 | 100 | 100.00% | 14.05 | 0 |
| **S07** | 456 | 1,550 | 100 | 94 | 94.00% | 15.46 | 0 |

---

## 5. Statistical Aggregation Across Scenarios

| Scenario | Total Runs | Total Generated | Total Matching | Mean Coverage | Min Coverage | Max Coverage | Coverage Spread |
|---|---|---|---|---|---|---|---|
| **S01** | 3 | 300 | 295 | **98.33%** | 98.00% | 99.00% | 1.00% |
| **S02** | 3 | 300 | 300 | **100.00%** | 100.00% | 100.00% | 0.00% |
| **S03** | 3 | 300 | 300 | **100.00%** | 100.00% | 100.00% | 0.00% |
| **S04** | 3 | 300 | 300 | **100.00%** | 100.00% | 100.00% | 0.00% |
| **S05** | 3 | 300 | 281 | **93.67%** | 89.00% | 99.00% | 10.00% |
| **S06** | 3 | 300 | 300 | **100.00%** | 100.00% | 100.00% | 0.00% |
| **S07** | 3 | 300 | 290 | **96.67%** | 94.00% | 100.00% | 6.00% |
| **All Scenarios** | **21** | **2,100** | **2,066** | **98.38%** | **89.00%** | **100.00%** | **11.00%** |

---

## 6. Empirical Findings and Analysis

1. **Discrete Invariance**: Pure categorical scenarios (S03, S04, S06) maintained 100% target coverage across all seeds. When training slices contain isolated discrete categories, CTGAN's softmax conditional representations faithfully reproduce the category without leakage.
2. **Continuous Boundary Leakage**: Boundary leakage occurred whenever continuous threshold conditions were present:
   - S01 (`age > 0.69`): 5 leaked records across 300 generated rows (98.33% mean).
   - S07 (`age > 0.69 and marital_status == 2`): 10 leaked records across 300 generated rows (96.67% mean). Leakage was entirely driven by the continuous `age` condition.
   - S05 (`age > 0.69 and education_num < -0.03`): 19 leaked records across 300 generated rows (93.67% mean), with the lowest single run dropping to 89.00% (Seed 123).
3. **Stochastic Sensitivity**: Continuous slices exhibited substantial sensitivity to random seeds (S05 had a 10 percentage point spread between Seed 123 and Seed 456). Single-run evaluations are insufficient for characterizing generator behavior.
4. **Data Isolation & Integrity**: Zero missing values occurred across all 2,100 generated records. All 12 configured features were preserved in canonical order.

---

## 7. Artifact Integrity Verification

Automated integrity verification of all 21 JSON artifacts in `component3/results/ctgan_targeted/` confirmed:
- Expected experiments: 21 / 21 present.
- Total records verified: 2,100.
- Total matching records: 2,066 (98.38% aggregate coverage).
- Missing values: 0.
- Column count: Exactly 12 features per record.
- Generation mode: `unsupervised_exploratory` across all runs.
- **Verification Status**: `ARTIFACT VERIFICATION: PASS`.

---

## 8. Unit Testing Evidence

Dedicated unit test suites created or verified in this task:
- `component3/tests/test_target_coverage.py`: **10 / 10 PASSED** (1.03s).
  - Tests categorical equality, continuous ranges, conjunction (AND), disjunction (OR), 100% full coverage, 0% zero coverage, dataclass serialization, input immutability, empty DataFrame rejection, unknown column rejection, and unsafe expression rejection.
- `component3/tests/test_ctgan_metadata.py`: **8 / 8 PASSED** (10.38s).
- `component3/tests/test_generator_data_preparer.py`: **17 / 17 PASSED** (1.11s).

---

## 9. Environment Incident: Windows Smart App Control

### Incident Summary
During Step 04.13, execution of the full regression test suite (`pytest component3/tests/ -q`) was interrupted during test collection:
```
ImportError: DLL load failed while importing _check_build: An Application Control policy has blocked this file.
```

### Diagnostic Trace
Examination of the system log (`Microsoft-Windows-CodeIntegrity/Operational`) identified that Windows 11 Smart App Control (SAC) transitioned to enforced mode at 8:19:29 PM:
```
Message: Smart App Control Block Details
Code Integrity determined that a process (python.exe) attempted to load 
\sklearn\__check_build\_check_build.cp311-win_amd64.pyd that did not meet the 
Enterprise signing level requirements or violated code integrity policy.
```

### Technical Root Cause & Remediation
- Smart App Control blocks unsigned native C-extension (`.pyd`) binaries that lack Microsoft Authenticode digital signatures.
- The setting was changed to **Off** in Windows Security (`windowsdefender://smartappcontrol`).
- Under Windows 11 architecture, unloading the active kernel Code Integrity policy driver requires a **system restart**.
- Upon system reboot, `scikit-learn` imports normally, and the regression test command (`pytest component3/tests/ -q`) will execute to complete the expected 289-test verification.

---

## 10. Research Significance & Next Steps

This task establishes the first empirical benchmark for slice-targeted synthetic tabular generation within Component 3:
- Demonstrates that training tabular generative models directly on diagnosed slices produces high target concentration (98.38% overall).
- Quantifies the phenomenon of continuous boundary leakage, providing direct empirical motivation for Sprint 05's planned target-coverage validation and post-sampling filtering mechanisms.
- Prepares Component 3 for the remaining generator implementations in Sprint 04:
  - Task 05: TVAE Generator Implementation.
  - Task 06: TVAE Targeted Generation Testing.
  - Task 07: TabDDPM Implementation & Testing.
  - Tasks 08–11: Multi-Generator Comparison, Selection Strategy, and Sprint 04 Synthesis.
