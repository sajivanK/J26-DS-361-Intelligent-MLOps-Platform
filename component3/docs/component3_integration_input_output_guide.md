# Component 3 — Integration Input/Output Specification Guide

## Collaborative Research Project: J26-DS-361 Intelligent MLOps Platform
**Component 3:** Adaptive Intelligent Synthetic Data Repair Engine  
**Document Purpose:** Exact Schema Reference, JSON Fixture Examples, and Platform Orchestration Context  

---

## 1. Architectural Overview & Context Injection

A critical finding from auditing the shared contract specifications (`shared/contracts/README.md`) is that **Component 1 (Drift Detection) and Component 2 (Root Cause Diagnosis) do NOT transmit physical datasets, models, or preprocessor artifacts**. They transmit statistical reports (drifted features, harm estimates, slice rules).

Therefore, Component 3 **cannot execute a repair on C1 and C2 outputs alone**.

A complete repair invocation requires a two-part payload:
1. **Upstream Statistical Telemetry:** C1 Drift Event + C2 Diagnostic Slice Rule.
2. **Platform Injected Data Context (`data_context`):** Accessible URIs for the reference dataset, drifted current window dataset, baseline model artifact, preprocessor artifact, target column name, and adaptation budget.

```
       [Component 1]                [Component 2]
    (Drift Report JSON)        (Diagnosis Slice JSON)
             \                           /
              \                         /
         +-------------------------------------+
         |      Platform Layer Orchestrator    |
         | (Injects Data Context & Model URIs) |
         +-------------------------------------+
                            |
                            v
                     [Component 3]
       (Adaptive Synthetic Data Repair Engine)
                            |
                            v
               [Component 3 Output Package]
              (Matches shared/contracts/
                 c3_output_schema.json)
                            |
                            v
                      [Component 4]
                (Release Controller / C4)
```

---

## 2. Minimum Valid Repair Request Checklist

Before Component 3 can begin execution, the platform orchestrator must verify that all items on this checklist are available:

- [ ] **Request ID:** Non-empty unique tracking string (e.g. `"REQ-20261021-001"`).
- [ ] **Drift Signal:** `drift_detected == True` (or `drift_type != "none"`).
- [ ] **Severity Score / Harm:** Valid float in `[0.0, 1.0]`.
- [ ] **Affected Features:** Non-empty list of feature names identified by C1/C2 (e.g. `["tenure", "MonthlyCharges"]`).
- [ ] **Slice Predicate / Rule:** Well-formed boolean condition string (e.g. `"tenure < 12 and Contract == 'Month-to-month'"`).
- [ ] **Reference Dataset:** Accessible CSV/Parquet file containing clean reference training data.
- [ ] **Adaptation Dataset:** Accessible CSV/Parquet file containing current window drifted data.
- [ ] **Source Model Artifact:** Accessible serialized model (`.pkl` / `.joblib`) capable of `.predict()` on the dataset schema.
- [ ] **Target Column:** Name of the prediction column (e.g. `"Churn"`).
- [ ] **Adaptation Budget:** Explicit integer budget specifying sample size limit (e.g. `200` rows).
- [ ] **Retention Tolerance:** Allowable source performance degradation margin (e.g. `0.05` F1 drop).

---

## 3. Boundary 1: Component 1 (Drift Detection) Output Contract

### Schema Reference: `shared/contracts/c1_output_schema.json`
This JSON payload is produced by Component 1 when drift monitoring triggers.

### Valid C1 Output Example
```json
{
  "schema_version": "1.0.0",
  "detection_id": "DET-TELCO-20261021-001",
  "timestamp": "2026-10-21T01:15:00Z",
  "dataset_id": "telco_churn",
  "model_id": "telco_rf_experimental_source",
  "drift_type": "combined",
  "drift_severity": "high",
  "predicted_harm": 0.725,
  "features_drifted": [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
  ],
  "p_values": {
    "tenure": 0.00012,
    "MonthlyCharges": 0.00340,
    "TotalCharges": 0.00085
  },
  "window_metadata": {
    "reference_window_id": "WIN-REF-2026Q1",
    "current_window_id": "WIN-CUR-2026Q3",
    "sample_count": 500
  }
}
```

### Key Field Descriptions
- `detection_id` (string, required): Unique identifier for this drift detection run.
- `drift_type` (string, required): One of `["marginal_single", "marginal_multi", "interaction", "combined", "none"]`.
- `drift_severity` (string, required): One of `["none", "low", "medium", "high", "critical"]`.
- `predicted_harm` (float, required): Estimated impact on model performance `[0.0, 1.0]`.
- `features_drifted` (array of strings, required): Feature columns displaying significant statistical divergence.
- `p_values` (object, optional): Per-feature hypothesis test p-values.

---

## 4. Boundary 2: Component 2 (Root Cause Diagnosis) Output Contract

### Schema Reference: `shared/contracts/c2_output_to_c3_schema.json`
This JSON payload is produced by Component 2 to specify the diagnosed root cause and problematic data slice.

### Valid C2 Output Example
```json
{
  "schema_version": "1.0.0",
  "diagnosis_id": "DIAG-TELCO-20261021-001",
  "detection_id": "DET-TELCO-20261021-001",
  "timestamp": "2026-10-21T01:16:30Z",
  "root_cause_type": "concept_shift",
  "affected_features_ranked": [
    "tenure",
    "Contract",
    "MonthlyCharges"
  ],
  "affected_slice_rule": "tenure < 12 and Contract == 'Month-to-month'",
  "slice_sample_count": 142,
  "diagnosis_confidence": 0.88,
  "diagnosis_stability": 0.92,
  "drift_impact_score": 0.78
}
```

### Key Field Descriptions
- `diagnosis_id` (string, required): Unique identifier for the diagnosis.
- `detection_id` (string, required): Foreign key linking back to C1's `detection_id`.
- `root_cause_type` (string, required): One of `["data_quality", "concept_shift", "covariate_shift", "sampling_bias", "external_factor", "unknown"]`.
- `affected_features_ranked` (array of strings, required): Ranked list of features contributing to drift.
- `affected_slice_rule` (string, required): Pandas/SQL query predicate isolating the underperforming data subpopulation.
- `slice_sample_count` (integer, required): Observed record count inside the identified slice.
- `diagnosis_confidence` (float, required): Confidence score of the diagnosis `[0.0, 1.0]`.

---

## 5. Platform Injected Data Context (`data_context`)

Because C1 and C2 pass metadata rather than binary data, the platform orchestrator merges C1, C2, and storage pointers into the complete execution context consumed by Component 3.

### Complete Orchestrated Repair Invocation Payload
```json
{
  "request_id": "REQ-TELCO-20261021-001",
  "schema_version": "1.0.0",
  "dataset_id": "telco_churn",
  "model_id": "telco_rf_experimental_source",
  "drift_event": {
    "detection_id": "DET-TELCO-20261021-001",
    "drift_type": "combined",
    "drift_severity": "high",
    "predicted_harm": 0.725,
    "features_drifted": ["tenure", "MonthlyCharges"]
  },
  "diagnosis": {
    "diagnosis_id": "DIAG-TELCO-20261021-001",
    "root_cause_type": "concept_shift",
    "affected_features_ranked": ["tenure", "MonthlyCharges"],
    "affected_slice_rule": "tenure < 12 and Contract == 'Month-to-month'",
    "slice_sample_count": 142,
    "diagnosis_confidence": 0.88
  },
  "data_context": {
    "reference_dataset_uri": "artifacts/datasets/telco_churn_reference_clean.csv",
    "current_dataset_uri": "artifacts/datasets/telco_churn_drifted_current.csv",
    "baseline_model_uri": "component3/models/telco_churn/rf_source.pkl",
    "target_column": "Churn",
    "adaptation_budget_size": 200,
    "source_retention_tolerance": 0.05,
    "max_runtime_seconds": 300.0,
    "candidate_generators": ["tvae", "ctgan", "tabddpm"]
  }
}
```

---

## 6. Boundary 3: Component 3 $\to$ Component 4 Output Contract

### Schema Reference: `shared/contracts/c3_output_schema.json`
This is the **flat 12-field contract** expected by Component 4 (Release Controller / Retraining Engine).

### Valid C3 Output Package Example
```json
{
  "repair_dataset_uri": "component3/results/runs/run_req-telco-20261021-001/synthetic_repair_k25.csv",
  "repair_package_status": "APPROVED_FOR_REVIEW",
  "generator": "tvae",
  "n_synthetic_rows": 25,
  "augmentation_ratio": 0.125,
  "fidelity_score": 0.142,
  "utility_score": 0.0123,
  "privacy_status": "PASS",
  "target_coverage": 1.0,
  "prescription_confidence": 0.85,
  "repair_version": "1.0.0",
  "repair_timestamp": "2026-10-21T01:36:42Z"
}
```

### Field Definitions for Component 4
1. `repair_dataset_uri` (string, required): Relative or absolute URI to the validated synthetic dataset artifact ready for retraining.
2. `repair_package_status` (string, required): One of `["APPROVED_FOR_REVIEW", "REJECTED", "ABSTAINED_NO_REPAIR"]`.
3. `generator` (string, required): Generator algorithm selected (`"tvae"`, `"ctgan"`, `"tabddpm"`, or `"none"`).
4. `n_synthetic_rows` (integer, required): Number of generated synthetic rows ($\ge 0$).
5. `augmentation_ratio` (float, required): Ratio of synthetic samples to authentic adaptation samples ($K / N_{\text{adapt}}$).
6. `fidelity_score` (float, required): Average multivariate Kolmogorov-Smirnov distance to reference distribution (`[0.0, 1.0]`, lower is better fidelity).
7. `utility_score` (float, required): Measured F1 gain on the target validation partition ($\Delta F_1$).
8. `privacy_status` (string, required): Privacy audit outcome: `"PASS"` (no verbatim copies), `"FAIL"` (privacy risk), or `"NOT_EVALUATED"`.
9. `target_coverage` (float, required): Fraction of diagnosed slice conditions satisfied by the generated data `[0.0, 1.0]`.
10. `prescription_confidence` (float, required): System confidence in the repair strategy `[0.0, 1.0]`.
11. `repair_version` (string, required): Semantic version of the repair engine protocol.
12. `repair_timestamp` (string, required): ISO 8601 UTC timestamp of repair package completion.
