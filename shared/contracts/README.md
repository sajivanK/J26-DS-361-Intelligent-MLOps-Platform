# API Contracts - J26-DS-361
## Intelligent MLOps Platform: v1.1 (C1 contract updated)

**Maintained by:** Sajivan K (IT23172296)  
**Status:** v1.0 frozen and agreed by all four members (October 2026). v1.1 changes the C1 contract only (proposed by Sajivan, 11 October 2026, to be agreed by the team). C2, C3 and C4 output fields are unchanged; only their references to C1 fields were updated.

---

## Files

| File | Produced By | Consumed By |
|------|-------------|-------------|
| `c1_output_schema.json` | C1 Sajivan | C2, C3 (evidence), C4 |
| `c2_output_to_c3_schema.json` | C2 Tharmika | C3 |
| `c2_output_to_c4_schema.json` | C2 Tharmika | C4 |
| `c3_output_schema.json` | C3 Mathusan | C4 |
| `c4_output_schema.json` | C4 Kesigan | Platform |

---

## Data Flow

```
Production batch arrives
        |
        v
  C1: Drift Detection (Sajivan)
        |
        |-----> C2: XAI Diagnosis (Tharmika)
        |              |
        |              |-----> C3: Synthetic Repair (Mathusan)
        |              |              |
        |              |              v
        |              |        C4: Release Controller (Kesigan)
        |              |              ^
        |              |______________|
        |                             ^
        |_____________________________|

C1 output goes to: C2 and C4
C2 output goes to: C3 (repair spec) and C4 (diagnostic report)
C3 output goes to: C4
C4 produces: final release decision
```

---

## C1 Contract v1.1 (11 October 2026)

Built from the tested C1 engine 1.2.1 (real outputs, not placeholders). Full details and examples are in `c1_output_schema.json` (`changes_from_v1_0`, `consumer_guidance`, `more_examples`).

| v1.0 | v1.1 | What consumers change |
|------|------|-----------------------|
| `drift_type` | removed; use `marginal_extent` (none/single/multi) + `interaction_suspected` (0 to 1) | C2: derive `root_cause_category` (rule in `consumer_guidance`). C3, C4: use the two fields as model inputs |
| `drift_severity` | `harm_severity` (same values, same rule) | Rename only |
| (none) | `drift_detected` (bool) | New. Do not gate on it alone: interaction-only drift can be harmful with `drift_detected = false` |
| `detector_scores.{ks, chi2, psi, mmd, lsdd, classifier_auc}` | `ks_mean, ks_max, chi2_mean, chi2_max, psi_mean, psi_max, mmd, lsdd, clf_auc` | C2: new key names |
| `confidence_stats` (3 values) | `prediction_stats` (6 values) | Rename keys |
| `harm_interval` lower bound >= 0 in examples | lower bound can be negative (not clipped) | Width = upper minus lower, still positive |
| (none) | `detectors_fired_list`, `warnings`, `engine_version`, `batch_size`, `runtime_sec` | Optional to use |
| request sent reference + production data | request sends production data only | C2 integration call |

Also updated for consistency: `c3_output_schema.json` (`input_context.from_c1`), `c4_output_schema.json` (`upstream_evidence_summary.c1`), and the `root_cause_category` descriptions in both C2 schemas.

---

## Resolved Contradictions (October 2026)

These were resolved before freezing the contracts:

| Issue | Resolution |
|-------|------------|
| C1 field names vs Tharmika's expectations | Keep C1 names: predicted_harm, harm_interval, detector_scores, confidence_stats (confidence_stats renamed prediction_stats in v1.1) |
| drift_severity type (string vs float) | String everywhere: none/low/moderate/high/critical (renamed harm_severity in v1.1) |
| detectors_fired missing from C1 | Added to C1 output |
| affected_features_ranked vs important_features | Use affected_features_ranked everywhere. Kesigan updated C4 |
| C2-to-C4 missing fields | Follow Tharmika's version. Add diagnosis_timestamp. Remove business_report and drift_impact_score |
| privacy_status casing | PASS/FAIL uppercase everywhere |
| data_context in C3 | NOT passed through C1 or C2. Injected by platform orchestrator from onboarding config |
| model_id, reference_dataset_uri missing from C1 | Added to C1 output as pass-through from onboarding config |

---

## How Each Member Uses These Contracts

### Sajivan (C1)
You produce `c1_output_schema.json`. Your `api.py` must return a dict matching this exactly. Every required field must be present.

### Tharmika (C2)
- You consume `c1_output_schema.json` as input
- You produce `c2_output_to_c3_schema.json` for Mathusan
- You produce `c2_output_to_c4_schema.json` for Kesigan
- During independent development: derive C1 values from injection ground truth (v1.1 fake builder: see your research guide)

### Mathusan (C3)
- You consume `c2_output_to_c3_schema.json` as primary input
- You also receive C1 evidence (c1_output_schema.json) for the Repair Prescription Engine
- You receive `data_context` from the platform orchestrator (NOT from C1 or C2)
- You produce `c3_output_schema.json` for Kesigan
- During independent development: vary diagnosis_stability as training parameter (0.3 to 0.95)

### Kesigan (C4)
- You consume: c1_output_schema.json + c2_output_to_c4_schema.json + c3_output_schema.json
- You produce `c4_output_schema.json` as final platform output
- During independent development: construct full scenarios using injection ground truth for C1, controlled values for C2 stability, self-generated repair data for C3

---

## Architectural Note: data_context

`data_context` (model_id, dataset_id, reference_dataset_uri, current_dataset_uri, target_column, label_status, preprocessing_uri) is NOT passed through the C1-C2 pipeline.

It is injected directly by the platform orchestrator into each component that needs it (C3 primarily) from the onboarding configuration stored during client setup. This keeps the C1 and C2 schemas clean and avoids duplicating large config blobs in every message.

---

## Validation Helper

```python
import json

def validate_output(output_dict, schema_path):
    with open(schema_path) as f:
        schema = json.load(f)
    
    # For C1, C2, C4 schemas
    if 'fields' in schema:
        required = [f for f, v in schema['fields'].items()
                    if v.get('required', False)]
    # For C3 schema
    elif 'output_fields' in schema:
        required = [f for f, v in schema['output_fields'].items()
                    if v.get('required', False)]
    else:
        required = []
    
    missing = [f for f in required if f not in output_dict]
    
    if missing:
        print(f"MISSING REQUIRED FIELDS: {missing}")
        return False
    
    print(f"All {len(required)} required fields present")
    return True

# Example
validate_output(your_c1_output, 'contracts/c1_output_schema.json')
```

---

## Integration Week Schedule (Week 3, October 2026)

```
Monday-Tuesday:   C1 + C2 connection and test
Wednesday:        C2 + C3 connection and test
Thursday:         C2 + C4 and C3 + C4 connections
Friday:           Full pipeline C1 -> C2 -> C3 -> C4 demo
```

---

## C4 Contract v1.1 (24 October 2026)

Proposed by Kesigan, agreed by Sajivan. Consumer: the platform (and the C4 page).

| Change | Why |
|---|---|
| New required field `evaluation_status`: `EVALUATED`, `FORECAST_UNAVAILABLE`, `UPSTREAM_GATE_FAILED` | Says what C4 actually evaluated. When C3 rejects, privacy fails, the generator is `NONE` or diagnosis_stability < 0.5, C4 does not train a candidate or forecast, and the decision is ESCALATE |
| `candidate_model_id`, `forecast_model_version`, `conformal_config_version`, `mlflow_run_id` may be `null` | Null when not created or executed, instead of placeholder ids |
| Values inside `future_safety_margins`, `future_safety_intervals`, `pre_release_evidence` may be `null` | Unmeasured values are null, never zeros or example forecasts |
| `hard_gate_status.schema_compatible`, `candidate_artifact_valid`, `rollback_ready` may be `null` | Runtime gates that were not checked stay null; `overall` is `FAIL` when an upstream gate failed |

The platform validator (`webapp/backend/app/contracts.py`) now accepts `null` only for fields marked `"nullable": true`. It checks top-level fields; values inside nested objects are checked by C4 itself. Example: `example_outputs.upstream_gate_failed` in `c4_output_schema.json`.
