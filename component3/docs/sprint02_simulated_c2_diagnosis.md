# Sprint 02 — Simulated Component 2 Diagnosis

## 1. Task Information

- **Research Project:** Intelligent MLOps Platform
- **Research Component:** Component 3 — Adaptive Intelligent Synthetic Data Repair Engine
- **Sprint:** 02 — Controlled Drift & Diagnosis Simulation
- **Task:** 06 — Implement Simulated Component 2 Diagnosis
- **Original Planned Sprint:** 21–25 September 2026
- **Actual Implementation and Verification:** 12 October 2026
- **Status:** Implementation and independent verification completed

## 2. Objective

The objective was to implement a simulated Component 2 diagnosis generator so Component 3 can be developed and tested independently before the real Component 2 integration is available.

The simulator produces repair specifications using controlled drift scenario information, simulated Component 1 evidence, and empirical affected-slice statistics.

## 3. Official Integration Contract

The simulator follows:

`shared/contracts/c2_output_to_c3_schema.json`

Each diagnosis contains nine required fields:

1. `root_cause_feature`
2. `root_cause_category`
3. `affected_slice_rule`
4. `affected_features_ranked`
5. `affected_value_ranges`
6. `slice_support`
7. `diagnosis_stability`
8. `drift_impact_score`
9. `temporal_window`

The root-cause categories are `marginal`, `interaction`, `combined`, and `unknown`.

## 4. Implementation

**Main script:**

`component3/src/experiments/simulate_c2_diagnosis.py`

The implementation includes:

- Loading controlled drift scenario configurations.
- Reading simulated C1 evidence.
- Mapping drift types to C2 root-cause categories.
- Creating scenario-specific affected-slice rules.
- Measuring slice support against current drifted data.
- Extracting affected feature ranges from reference data.
- Assigning deterministic simulation parameters.
- Validating generated diagnoses against the official contract.
- Recording simulation provenance separately from the C2 payload.

## 5. Experimental Results

| Scenario | Root-Cause Category | Current Slice Support | Available Reference Records |
| -------- | ------------------- | --------------------- | --------------------------- |
| S00      | Unknown             | 0.00%                 | 0                           |
| S01      | Marginal            | 35.98%                | 6,097                       |
| S02      | Marginal            | 100.00%               | 1,223                       |
| S03      | Marginal            | 28.72%                | 11,999                      |
| S04      | Marginal            | 6.00%                 | 2,462                       |
| S05      | Marginal            | 5.70%                 | 4,190                       |
| S06      | Interaction         | 27.90%                | 10,579                      |
| S07      | Combined            | 25.83%                | 3,792                       |

Each scenario uses a 6,000-record current batch.

## 6. Independent Validation

An independent validation script was executed after generating the JSON files.

The verification covered:

- Official C2 contract conformance.
- Exact nine-field payload structure.
- Correct root-cause category mapping.
- Executable affected-slice rules.
- Current-data slice support within a tolerance of 0.000001.
- Reference-derived affected feature ranges.
- Consistency of reference and current slice counts.
- Presence of corresponding simulation provenance.

**Result: 8/8 scenarios passed.**

## 7. Generated Artifacts

**Diagnosis files:**

`component3/data/simulated_c2/c2_diagnosis_S00.json` through `c2_diagnosis_S07.json`

**Provenance file:**

`component3/data/simulated_c2/c2_simulation_provenance.json`

The provenance records the simulation method, slice statistics, and configured diagnosis parameters.

## 8. Research Limitations

This implementation is a simulated C2 integration-testing fixture.

It does not execute Temporal SHAP or Sliceline.

Root-cause features and rankings are derived from known drift injection configurations rather than independently discovered explanations.

Affected-slice rules are predefined experimental fixtures, not verified worst-performing subgroups.

Slice support and reference feature ranges are measured from the datasets.

Diagnosis stability, drift impact scores, and temporal windows are deterministic simulation settings, not measured C2 outputs.

S00 is a no-drift negative test with an intentionally empty slice and a technical root-cause placeholder.

S02 represents a global shift whose affected-slice rule matches the entire current batch.

These limitations must be considered when interpreting subsequent Component 3 experiments.

## 9. Outcome

Component 3 now has eight independently verified simulated C2 repair specifications covering no drift, marginal drift, multi-feature drift, interaction drift, and combined drift.

These artifacts are ready for subsequent input-contract and repair-readiness testing.

**Task 06 conclusion:** Implementation and independent validation completed successfully.
