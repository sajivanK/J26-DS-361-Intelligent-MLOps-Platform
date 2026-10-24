# Component 3 — Integration Contract Compatibility Matrix

## Collaborative Research Project: J26-DS-361 Intelligent MLOps Platform
**Component 3:** Adaptive Intelligent Synthetic Data Repair Engine  
**Branch:** `c3-mathusan` | **Committed Revision:** `fb74cdcf0ca4ea47a3cc5e1dedf7d3c1eadd2c45`  
**Audit Date:** October 2026 | **Auditor Role:** Senior MLOps Architect & Quality Assurance Auditor  

---

## 1. Executive Summary of Contract Status

This matrix details the exact contract compatibility across all three integration boundaries for Component 3:
1. **Boundary 1 (C1 $\to$ C3):** Component 1 Drift Detection output consumed by Component 3.
2. **Boundary 2 (C2 $\to$ C3):** Component 2 Root Cause Diagnosis output consumed by Component 3.
3. **Boundary 3 (C3 $\to$ C4):** Component 3 Repair Package output delivered to Component 4 (Release Controller / Retraining).

### Overall Boundary Status
- **Boundary 1 (C1 $\to$ C3):** **PARTIALLY SUPPORTED / REQUIRES ADAPTER UPDATE**. Component 3 possesses `C1DriftAdapter`, but the frozen contract `shared/contracts/c1_output_schema.json` names the severity metric `predicted_harm` (float `[0, 1]`) and does not supply a boolean `drift_detected` field. Enums for `drift_type` also differ (`marginal_single` / `marginal_multi` / `combined` vs. `CONCEPT` / `COVARIATE`).
- **Boundary 2 (C2 $\to$ C3):** **PARTIALLY SUPPORTED / REQUIRES ADAPTER UPDATE**. Component 3 has `C2DiagnosisAdapter`, but the frozen contract `shared/contracts/c2_output_to_c3_schema.json` uses the key `affected_slice_rule` (while C3's adapter currently looks for `diagnosed_problematic_slice` or `slice_predicate`). Unadapted real C2 JSON causes a `ValueError` in `c2_adapter.adapt()`. Furthermore, C2 outputs `affected_features_ranked` as a list of strings, while C3 expects a list of dictionaries.
- **Boundary 3 (C3 $\to$ C4):** **MISMATCHED / REQUIRES ADAPTER UPDATE**. Component 3 emits a nested, governance-rich `C4ReviewPayload` (`component3/src/adapters/c4_retraining_adapter.py`), but the frozen contract `shared/contracts/c3_output_schema.json` specifies a **flat 12-field schema** (`repair_dataset_uri`, `repair_package_status`, `generator`, `n_synthetic_rows`, `augmentation_ratio`, `fidelity_score`, `utility_score`, `privacy_status`, `target_coverage`, `prescription_confidence`, `repair_version`, `repair_timestamp`). None of the 12 flat top-level fields are present in the current `C4ReviewPayload`.

---

## 2. Boundary 1: C1 (Drift Detection) $\to$ C3 (Synthetic Repair)

### Source Contract Reference: `shared/contracts/c1_output_schema.json`
### Consumer Implementation: `component3/src/adapters/c1_drift_adapter.py` (`C1DriftAdapter`)

| Boundary | Required Field | Producer Format (C1 Contract) | Consumer Format (C3 Adapter) | Compatibility | Evidence / Location | Problem & Remediation |
|---|---|---|---|---|---|---|
| C1 $\to$ C3 | `schema_version` | String (optional, e.g. `"1.0.0"`) | String (optional, defaults `"1.0.0"`) | **MATCHED** | `c1_drift_adapter.py:L116` | No issue. |
| C1 $\to$ C3 | `detection_id` / `request_id` | String (`"DET-..."`) | String (parsed as `source_event_id` / `request_id`) | **MATCHED** | `c1_drift_adapter.py:L118` | Successfully captured and stored. |
| C1 $\to$ C3 | `timestamp` | String (ISO 8601 UTC) | String (recorded in adapter metadata) | **MATCHED** | `c1_drift_adapter.py:L119` | Preserved in telemetry. |
| C1 $\to$ C3 | `drift_detected` | **Missing in C1 JSON** | Boolean (`True`/`False`) | **MISMATCHED** | `c1_drift_adapter.py:L124` | C1 schema does not output `drift_detected`; it provides `drift_type` (`"none"` vs. other) and `drift_severity`. C3 adapter defaults to `False` if missing, causing false abstentions if raw C1 JSON is consumed. **Remediation:** In adapter, derive `drift_detected = (drift_type != "none" and drift_severity != "none")`. |
| C1 $\to$ C3 | `drift_type` | Enum: `["marginal_single", "marginal_multi", "interaction", "combined", "none"]` | Enum: `["CONCEPT", "COVARIATE", "SUBPOPULATION", "LABEL", "PRIOR", "UNKNOWN"]` | **PARTIALLY SUPPORTED** | `c1_drift_adapter.py:L70-L86` | C1 types indicate distribution dimensionality (`marginal_single`, etc.), whereas C3 classifies statistical nature. Currently maps to `DriftType.UNKNOWN` for `interaction` and `combined`. **Remediation:** Add explicit mappings in `_DRIFT_TYPE_MAP` (`combined` $\to$ `COVARIATE`, `interaction` $\to$ `CONCEPT`, `none` $\to$ `NONE`). |
| C1 $\to$ C3 | `drift_severity` | Enum: `["none", "low", "medium", "high", "critical"]` | Enum: `["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]` | **MATCHED** | `c1_drift_adapter.py:L88-L98` | Case-insensitive mapping works correctly. |
| C1 $\to$ C3 | `severity_score` / `predicted_harm` | Float `[0.0, 1.0]` under key `predicted_harm` | Float `[0.0, 1.0]` under key `severity_score` | **MISMATCHED** | `c1_drift_adapter.py:L126` | C1 contract uses `predicted_harm`. C3 adapter expects `severity_score` and defaults to `0.0` if not found. **Remediation:** Check `raw.get("predicted_harm", raw.get("severity_score", 0.0))`. |
| C1 $\to$ C3 | `features_drifted` / `affected_features` | Array of strings (`["tenure", "MonthlyCharges"]`) | Array of strings | **MATCHED** | `c1_drift_adapter.py:L127` | Key aliases `features_drifted` and `affected_features` are supported. |
| C1 $\to$ C3 | `p_values` / `feature_p_values` | Object (key: feature, value: float) | Object (stored in `feature_p_values`) | **MATCHED** | `c1_drift_adapter.py:L129` | Stored in normalized drift event. |
| C1 $\to$ C3 | `window_metadata` | Object (`reference_window_id`, `current_window_id`, `sample_count`) | Stored in telemetry / adapter log | **MATCHED** | `c1_drift_adapter.py:L130` | Preserved in metadata. |

---

## 3. Boundary 2: C2 (Diagnosis) $\to$ C3 (Synthetic Repair)

### Source Contract Reference: `shared/contracts/c2_output_to_c3_schema.json`
### Consumer Implementation: `component3/src/adapters/c2_diagnosis_adapter.py` (`C2DiagnosisAdapter`)

| Boundary | Required Field | Producer Format (C2 Contract) | Consumer Format (C3 Adapter) | Compatibility | Evidence / Location | Problem & Remediation |
|---|---|---|---|---|---|---|
| C2 $\to$ C3 | `schema_version` | String (optional, e.g. `"1.0.0"`) | String (optional, defaults `"1.0.0"`) | **MATCHED** | `c2_diagnosis_adapter.py:L141` | Supported. |
| C2 $\to$ C3 | `diagnosis_id` | String (`"DIAG-..."`) | String (`diagnosis_id`) | **MATCHED** | `c2_diagnosis_adapter.py:L142` | Captured. |
| C2 $\to$ C3 | `detection_id` | String (`"DET-..."`) | String (mapped to `c1_drift_event_id`) | **MATCHED** | `c2_diagnosis_adapter.py:L143` | Traceability maintained. |
| C2 $\to$ C3 | `timestamp` | String (ISO 8601 UTC) | String | **MATCHED** | `c2_diagnosis_adapter.py:L144` | Preserved. |
| C2 $\to$ C3 | `root_cause_type` | Enum: `["data_quality", "concept_shift", "covariate_shift", "sampling_bias", "external_factor", "unknown"]` | String | **MATCHED** | `c2_diagnosis_adapter.py:L145` | Preserved as string attribute. |
| C2 $\to$ C3 | `affected_features_ranked` | Array of strings (`["tenure", "Contract"]`) | Array of objects (`affected_feature_details`) | **PARTIALLY SUPPORTED** | `c2_diagnosis_adapter.py:L100-L113` | C3 expects list of dicts with `feature_name`, `importance_rank`, `attribution_score`. C2 provides list of strings. C3 adapter currently creates synthetic entries if missing, but raises warnings. |
| C2 $\to$ C3 | `affected_slice_rule` | String (e.g. `"tenure < 12 and Contract == 'Month-to-month'"`) | String under key `slice_predicate` or `diagnosed_problematic_slice` | **MISMATCHED (CRITICAL)** | `c2_diagnosis_adapter.py:L147-L158` | C2 frozen schema defines `affected_slice_rule`. C3 looks for `diagnosed_problematic_slice`, `slice_predicate`, or `subpopulation_filter`. Raw C2 JSON causes `ValueError: Diagnostic slice_predicate must be a non-empty string`. **Remediation:** Add alias `raw.get("affected_slice_rule")`. |
| C2 $\to$ C3 | `slice_sample_count` / `support` | Integer (`slice_sample_count`) | Float (`reported_slice_support`) | **PARTIALLY SUPPORTED** | `c2_diagnosis_adapter.py:L152` | C2 outputs absolute row count; C3 expects relative support `[0, 1]`. Defaulted to `0.25` if not normalized. |
| C2 $\to$ C3 | `diagnosis_confidence` | Float `[0.0, 1.0]` | Float `[0.0, 1.0]` | **MATCHED** | `c2_diagnosis_adapter.py:L153` | Directly compatible. |
| C2 $\to$ C3 | `diagnosis_stability` | Float `[0.0, 1.0]` | Not parsed | **MISSING IN C3** | `c2_diagnosis_adapter.py` | C2 provides stability score; C3 adapter currently ignores it. Does not cause crash. |
| C2 $\to$ C3 | `drift_impact_score` | Float `[0.0, 1.0]` | Not parsed | **MISSING IN C3** | `c2_diagnosis_adapter.py` | C2 provides impact score; C3 adapter currently ignores it. |

---

## 4. Boundary 3: C3 (Synthetic Repair) $\to$ C4 (Release Controller / Retraining)

### Target Contract Reference: `shared/contracts/c3_output_schema.json`
### Producer Implementation: `component3/src/adapters/c4_retraining_adapter.py` (`C4RetrainingAdapter`)

The team has frozen `shared/contracts/c3_output_schema.json` as a **flat 12-field schema**. Below is the field-by-field audit against C3's current `C4RetrainingAdapter`:

| Boundary | Required Field | Target Format (C3 Output Contract) | Current Producer Format (C3 C4 Adapter) | Compatibility | Evidence / Location | Problem & Remediation |
|---|---|---|---|---|---|---|
| C3 $\to$ C4 | `repair_dataset_uri` | String (path/URI to synthetic CSV) | Nested: `artifacts.synthetic_data_path` | **MISMATCHED (CRITICAL)** | `c4_retraining_adapter.py:L114` | Top-level key missing in root JSON. |
| C3 $\to$ C4 | `repair_package_status` | Enum: `["APPROVED_FOR_REVIEW", "REJECTED", "ABSTAINED_NO_REPAIR"]` | Nested: `metadata.package_status` | **MISMATCHED (CRITICAL)** | `c4_retraining_adapter.py:L108` | Top-level key missing in root JSON. |
| C3 $\to$ C4 | `generator` | Enum: `["tvae", "ctgan", "tabddpm", "none"]` | Nested: `prescription.recommended_generator` | **MISMATCHED (CRITICAL)** | `c4_retraining_adapter.py:L110` | Top-level key missing in root JSON. |
| C3 $\to$ C4 | `n_synthetic_rows` | Integer ($\ge 0$) | Nested: `artifacts.n_synthetic_rows` | **MISMATCHED (CRITICAL)** | `c4_retraining_adapter.py:L115` | Top-level key missing in root JSON. |
| C3 $\to$ C4 | `augmentation_ratio` | Float ($\ge 0.0$) | Not computed at top level | **MISSING IN PRODUCER** | `c4_retraining_adapter.py` | Required by schema validator. |
| C3 $\to$ C4 | `fidelity_score` | Float `[0.0, 1.0]` | Nested: `validation.fidelity_ks_statistic` | **MISMATCHED (CRITICAL)** | `c4_retraining_adapter.py:L119` | Top-level key missing in root JSON. |
| C3 $\to$ C4 | `utility_score` | Float `[0.0, 1.0]` | Nested: `evaluation.target_delta_f1` | **MISMATCHED (CRITICAL)** | `c4_retraining_adapter.py:L123` | Top-level key missing in root JSON. |
| C3 $\to$ C4 | `privacy_status` | Enum: `["PASS", "FAIL", "NOT_EVALUATED"]` | Nested: `validation.privacy_status` | **MISMATCHED (CRITICAL)** | `c4_retraining_adapter.py:L121` | Top-level key missing in root JSON. |
| C3 $\to$ C4 | `target_coverage` | Float `[0.0, 1.0]` | Not exposed at top level | **MISSING IN PRODUCER** | `c4_retraining_adapter.py` | Required by schema validator. |
| C3 $\to$ C4 | `prescription_confidence` | Float `[0.0, 1.0]` | Nested: `prescription.evidence_strength` | **MISMATCHED (CRITICAL)** | `c4_retraining_adapter.py:L112` | String enum vs float. |
| C3 $\to$ C4 | `repair_version` | String (e.g. `"1.0.0"`) | Nested: `metadata.schema_version` | **MISMATCHED (CRITICAL)** | `c4_retraining_adapter.py:L106` | Top-level key missing in root JSON. |
| C3 $\to$ C4 | `repair_timestamp` | String (ISO 8601 UTC) | Nested: `metadata.created_at_utc` | **MISMATCHED (CRITICAL)** | `c4_retraining_adapter.py:L107` | Top-level key missing in root JSON. |

---

## 5. Summary of Blocking Contract Gaps

1. **Gap B1-1 (Field Name):** `c1_drift_adapter.py` looks for `severity_score` instead of `predicted_harm`.
2. **Gap B1-2 (Missing Drift Detected):** `c1_drift_adapter.py` defaults `drift_detected=False` if not present in C1 output, even when drift is detected.
3. **Gap B2-1 (Slice Rule Key Mismatch):** `c2_diagnosis_adapter.py` raises `ValueError` on real C2 JSON because it expects `slice_predicate` rather than `affected_slice_rule`.
4. **Gap B3-1 (Schema Structure Incompatibility):** `c4_retraining_adapter.py` emits an internal hierarchical review structure instead of the flat 12-field schema mandated by `shared/contracts/c3_output_schema.json`.

---

## 6. Verdict on Contract Readiness

**Status: LEVEL 1 — INTERNAL ENGINE WORKING / CONTRACT ALIGNMENT REQUIRED**

Component 3's core data repair engine is fully functional internally. However, until the adapters are updated to align with the frozen contracts in `shared/contracts/`, passing real JSON payloads between C1 $\to$ C3, C2 $\to$ C3, and C3 $\to$ C4 will fail schema validation or raise runtime parsing exceptions.
