# Component 3: PP1 Contract Mapping & Schema Specification

**Project:** Intelligent MLOps Platform (J26-DS-361)  
**Author:** Mathusan K (IT23174580), Component 3 Lead  
**Document Version:** 1.0.0 (PP1 Baseline)  
**Authoritative Schemas:**
- `shared/contracts/c1_output_schema.json` (v1.1)
- `shared/contracts/c2_output_to_c3_schema.json` (v1.0)
- `shared/contracts/c3_output_schema.json` (v1.0)
- `shared/contracts/c4_output_schema.json` (v1.0)

---

## 1. Upstream Input Contracts Received by C3

Component 3 receives input from three distinct sources:
1. **Component 1 (Drift Evidence)**: Quantitative drift magnitude, severity, and suspected features.
2. **Component 2 (Repair Specification)**: Diagnosed root cause, affected subgroup rule, and impact metrics.
3. **Platform Orchestrator (Data Context)**: Dataset and model identifiers, storage URIs, and metadata.

### 1.1 From Component 1 (`c1_output_schema.json` v1.1)

| Field Name | Type | Req | Allowed Values / Range | Meaning & C3 Consumption |
|---|---|---|---|---|
| `evidence_version` | `string` | Yes | e.g. `"c1-536b3839"` | Audit traceability identifier recorded in C3 package. |
| `model_id` | `string` | Yes | e.g. `"adult-xgb-v1"` | Identifier of monitored model; validated against model registry. |
| `predicted_harm` | `float` | Yes | -1.0 to 1.0 (or pp) | Predicted performance degradation (accuracy or F1 drop). Higher means more severe. |
| `harm_interval` | `array[float, float]` | Yes | `[lo, hi]` | 95% conformal prediction interval on harm. C3 uses width to gauge drift certainty. |
| `harm_severity` | `string` | Yes | `"none"`, `"low"`, `"moderate"`, `"high"`, `"critical"` | Determines whether repair gate passes (`!= "none"`). Scales repair quantity. |
| `drift_detected` | `boolean` | Yes | `true`, `false` | Binary drift flag. If false and severity is none, pipeline halts at repair gate. |
| `marginal_extent` | `string` | Yes | `"none"`, `"single"`, `"multi"` | Extent of marginal distribution shift. |
| `interaction_suspected` | `float` | Yes | `0.0` to `1.0` | Probability/score of feature interactions. Higher values indicate complex drift. |
| `affected_features` | `array[string]` | Yes | Feature names list | Features exhibiting significant drift. Used as generator priority conditioning. |
| `batch_size` | `integer` | Yes | >= 100 | Number of production rows in the analyzed batch. |

### 1.2 From Component 2 (`c2_output_to_c3_schema.json` v1.0)

| Field Name | Type | Req | Allowed Values / Range | Meaning & C3 Consumption |
|---|---|---|---|---|
| `root_cause_feature` | `string` | Yes | Feature name | Primary feature driving performance degradation. |
| `root_cause_category` | `string` | Yes | `"marginal"`, `"interaction"`, `"combined"`, `"unknown"` | Category of drift. Informs C3 generator choice (e.g., CTGAN vs TVAE vs TabDDPM). |
| `affected_slice_rule` | `string` | Yes | Valid pandas `.query()` string | Predicate isolating the failing subgroup (e.g., `"age > 45 and hours_per_week < 25"`). |
| `affected_features_ranked`| `array[string]` | Yes | Ordered feature names | Features ranked by Drift Impact Score contribution. |
| `affected_value_ranges` | `object` | Yes | `{feat: [min, max]}` | Concrete bounding ranges for continuous features or categories for discrete ones. |
| `slice_support` | `float` | Yes | `0.0` to `1.0` | Prevalence of affected slice in production data. Multiplied by reference size to determine required repair rows. |
| `diagnosis_stability` | `float` | Yes | `0.0` to `1.0` | Consistency of diagnosis over monitoring cycles. Values < 0.5 trigger conservative augmentation. |
| `drift_impact_score` | `float` | Yes | `0.0` to `1.0` | Fused score scaling repair quantity via C3 Prescription Engine. |
| `temporal_window` | `integer` | Yes | >= 1 | Number of cycles diagnosis remained stable. |

### 1.3 From Platform Orchestrator (`data_context`)

| Field Name | Type | Req | Example | Meaning & C3 Consumption |
|---|---|---|---|---|
| `model_id` | `string` | Yes | `"telco_rf_experimental_source"` | Model registry identifier. |
| `dataset_id` | `string` | Yes | `"telco_churn"` or `"uci"` | Dataset registry identifier. |
| `reference_dataset_uri` | `string` | Yes | `"shared/uci_adult/data/reference.csv"` | Clean baseline reference data path. |
| `current_dataset_uri` | `string` | Yes | `"data/production/batch_01.csv"` | Current production batch (labels removed). |
| `target_column` | `string` | Yes | `"Churn"` or `"target"` | Target label column for downstream evaluation. |
| `label_status` | `string` | No | `"AVAILABLE"` or `"UNAVAILABLE"` | Declares whether ground-truth labels exist for adaptation. |
| `preprocessing_uri` | `string` | No | `"shared/uci_adult/models/preprocessor.pkl"` | Preprocessor pipeline path for feature transformation. |
| `adaptation_budget_size`| `integer` | No | `200` | Budget of real samples available for fine-tuning/adaptation. |
| `source_retention_tolerance`| `float`| No | `0.05` | Maximum allowable drop on non-drifted source data. |

---

## 2. Downstream Output Contract (`c3_output_schema.json` v1.0)

Component 3 produces the authoritative `c3` contract payload sent to Component 4 (Release Control).  
Every field is required, strictly validated against types and allowed values.

| # | Field Name | Type | Req | Range / Allowed Values | Description & Calculation |
|---|---|---|---|---|---|
| 1 | `repair_dataset_uri` | `string` | Yes | Valid relative path or URI | Path to the validated synthetic repair CSV dataset. C4 retrains candidate model with this file. |
| 2 | `repair_package_status` | `string` | Yes | `["APPROVED", "REJECTED"]` | Overall validation gate decision. Only `APPROVED` packages authorize retraining. |
| 3 | `generator` | `string` | Yes | `["CTGAN", "TVAE", "TabDDPM"]` | Generative architecture selected by the Prescription Engine for this slice and drift condition. |
| 4 | `n_synthetic_rows` | `integer` | Yes | >= 0 | Number of validated synthetic rows generated. Set to 0 if package is rejected. |
| 5 | `augmentation_ratio` | `float` | Yes | `0.0` to `1.0` | Ratio of synthetic rows to original reference training data (`n_synthetic / n_training`). |
| 6 | `fidelity_score` | `float` | Yes | `0.0` to `1.0` | Statistical fidelity (derived from 1 - mean KS statistic across numeric features and TVD on categoricals). Higher is better. |
| 7 | `utility_score` | `float` | Yes | `0.0` to `1.0` | Downstream model F1 score on the affected slice when retrained with synthetic repair data. |
| 8 | `privacy_status` | `string` | Yes | `["PASS", "FAIL"]` | Distance to Closest Record (DCR) check. Hard gate: if 5th percentile DCR < threshold, fails. |
| 9 | `target_coverage` | `float` | Yes | `0.0` to `1.0` | Fraction of generated synthetic rows that satisfy `affected_slice_rule`. Higher is better. |
| 10 | `prescription_confidence` | `float` | Yes | `0.0` to `1.0` | Confidence of the Prescription Engine in the prescribed generator and quantity. |
| 11 | `repair_version` | `string` | Yes | e.g. `"c3-001"` | Version identifier for audit tracing in C4. |
| 12 | `repair_timestamp` | `string` | Yes | ISO-8601 UTC string | Timestamp when repair package was generated. |

---

## 3. Status Mapping & Acceptance Gates

### Internal Status vs Contract Status
- **`APPROVED_FOR_REVIEW`** -> `repair_package_status: "APPROVED"`. All validation gates passed (structural, fidelity >= 0.70, privacy == PASS, utility delta >= 0).
- **`REJECTED`** -> `repair_package_status: "REJECTED"`. One or more gates failed (e.g. DCR privacy leak, fidelity collapse, or utility degradation).
- **`ABSTAINED_NO_REPAIR`** -> `repair_package_status: "REJECTED"`. No repair necessary (e.g. drift severity was negligible or slice was empty).
- **`FAILED`** -> `repair_package_status: "REJECTED"`. System error or missing dependencies.

### Contract Compliance Guarantee
1. When `APPROVED`: `repair_dataset_uri` points to an authentic generated CSV file on disk with verified SHA256.
2. When `REJECTED`: `n_synthetic_rows` is 0, `repair_dataset_uri` is `"NONE"`, and all scores accurately reflect why the package was rejected.
3. Every outgoing dictionary is validated by `contracts.validate("c3", out)` to guarantee zero platform schema violations.
