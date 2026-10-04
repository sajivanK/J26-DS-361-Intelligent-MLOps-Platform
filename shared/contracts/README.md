# API Contracts - J26-DS-361
## Intelligent MLOps Platform — Frozen v1.0

**Maintained by:** Sajivan K (IT23172296)  
**Status:** FROZEN — all four team members agreed October 2026

---

## Files

| File | Produced By | Consumed By |
|------|-------------|-------------|
| `c1_output_schema.json` | C1 Sajivan | C2, C4 |
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

## Resolved Contradictions (October 2026)

These were resolved before freezing the contracts:

| Issue | Resolution |
|-------|------------|
| C1 field names vs Tharmika's expectations | Keep C1 names: predicted_harm, harm_interval, detector_scores, confidence_stats |
| drift_severity type (string vs float) | String everywhere: none/low/moderate/high/critical |
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
- During independent development: derive C1 values from injection ground truth

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
