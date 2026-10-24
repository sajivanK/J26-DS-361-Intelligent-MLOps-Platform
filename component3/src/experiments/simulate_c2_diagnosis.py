"""Simulated Component 2 (Root Cause Diagnosis Engine) evidence generator.

Generates contract-shaped, explicitly simulated C2 diagnosis specifications
for C3 testing and integration.
"""

import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

CONFIG_PATH = ROOT / "component3/configs/drift_scenarios.json"
CONTRACT_PATH = ROOT / "shared/contracts/c2_output_to_c3_schema.json"
REFERENCE_PATH = ROOT / "shared/uci_adult/data/reference.csv"
C1_DIRECTORY = ROOT / "component3/data/simulated_c1"
DRIFT_DIRECTORY = ROOT / "component3/data/drifted"
OUTPUT_DIRECTORY = ROOT / "component3/data/simulated_c2"

DRIFT_CATEGORY_MAPPING = {
    "none": "unknown",
    "marginal_single": "marginal",
    "marginal_multi": "marginal",
    "interaction": "interaction",
    "combined": "combined",
}

SLICE_RULES = {
    "S00": "age > 100",
    "S01": "age > 0.69",
    "S02": "capital_loss > -0.20",
    "S03": "marital_status == 2",
    "S04": "race == 2",
    "S05": "age > 0.69 and education_num < -0.03",
    "S06": "marital_status == 2 and relationship == 0",
    "S07": "age > 0.69 and marital_status == 2",
}


def load_json(path: Path) -> dict:
    """Load a JSON document."""
    return json.loads(path.read_text(encoding="utf-8"))


def validate_c2_payload(
    payload: dict,
    contract: dict,
    reference_df: pd.DataFrame,
) -> None:
    """Validate a simulated C2 payload against the official contract."""
    fields = contract["fields"]
    required = {
        name for name, specification in fields.items()
        if specification.get("required", False)
    }
    # Check exact field names.
    missing = required - payload.keys()
    extra = payload.keys() - fields.keys()
    assert not missing, f"Missing required fields: {sorted(missing)}"
    assert not extra, f"Unexpected fields: {sorted(extra)}"

    # Check declared field types and constraints.
    for name, specification in fields.items():
        if name not in payload:
            continue
        value = payload[name]
        declared_type = specification["type"]

        if declared_type == "string":
            assert isinstance(value, str), f"{name} must be a string"
            if "allowed_values" in specification:
                assert value in specification["allowed_values"], (
                    f"{name} must be one of {specification['allowed_values']}"
                )
        elif declared_type == "float":
            assert isinstance(value, (int, float)) and not isinstance(
                value, bool
            ), f"{name} must be numeric"
            if specification.get("range") == "0.0 to 1.0":
                assert 0.0 <= float(value) <= 1.0, f"{name} must be in range 0.0 to 1.0"
        elif declared_type == "integer":
            assert isinstance(value, int) and not isinstance(
                value, bool
            ), f"{name} must be an integer"
            assert value >= 1, f"{name} must be >= 1"
        elif declared_type == "array[string]":
            assert isinstance(value, list), f"{name} must be a list"
            assert all(isinstance(item, str) for item in value), (
                f"{name} items must all be strings"
            )
        elif declared_type == "object":
            assert isinstance(value, dict), f"{name} must be an object (dict)"

    # Verify affected_slice_rule executes cleanly on reference_df
    slice_rule = payload["affected_slice_rule"]
    try:
        reference_df.query(slice_rule)
    except Exception as exc:
        raise AssertionError(
            f"affected_slice_rule failed execution on reference_df: {exc}"
        ) from exc


def analyze_affected_slice(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    rule: str,
    affected_features: list,
    feature_types: dict,
) -> dict:
    """Analyze the affected slice across reference and current datasets."""
    reference_slice = reference_df.query(rule)
    current_slice = current_df.query(rule)
    reference_count = len(reference_slice)
    current_count = len(current_slice)
    slice_support = (
        current_count / len(current_df)
        if len(current_df) > 0
        else 0.0
    )
    affected_value_ranges = {}
    for feature in affected_features:
        if feature not in reference_df.columns:
            raise ValueError(f"Unknown affected feature: {feature}")
        if reference_slice.empty:
            affected_value_ranges[feature] = []
            continue
        if feature_types[feature] == "continuous":
            affected_value_ranges[feature] = [
                float(reference_slice[feature].min()),
                float(reference_slice[feature].max()),
            ]
        elif feature_types[feature] == "categorical":
            affected_value_ranges[feature] = sorted(
                reference_slice[feature].unique().tolist()
            )
        else:
            raise ValueError(
                f"Unsupported feature type: {feature_types[feature]}"
            )
    return {
        "slice_support": round(slice_support, 6),
        "reference_count": reference_count,
        "current_count": current_count,
        "affected_value_ranges": affected_value_ranges,
    }


SCENARIO_DIAGNOSIS_PARAMS = {
    "S00": {"stability": 0.30, "impact": 0.00, "window": 1},
    "S01": {"stability": 0.85, "impact": 0.45, "window": 3},
    "S02": {"stability": 0.75, "impact": 0.65, "window": 3},
    "S03": {"stability": 0.80, "impact": 0.42, "window": 3},
    "S04": {"stability": 0.70, "impact": 0.35, "window": 3},
    "S05": {"stability": 0.88, "impact": 0.72, "window": 4},
    "S06": {"stability": 0.84, "impact": 0.68, "window": 4},
    "S07": {"stability": 0.92, "impact": 0.82, "window": 5},
}


def build_c2_diagnosis(
    scenario: dict,
    c1_evidence: dict,
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_types: dict,
    contract: dict,
) -> tuple[dict, dict]:
    """Build a simulated C2 repair specification.

    Returns:
        payload: Official nine-field C2-to-C3 output.
        provenance: Separate simulation metadata.
    """
    sid = scenario["id"]
    rule = SLICE_RULES[sid]

    # Use the official C1 evidence as the starting point.
    drift_type = c1_evidence["drift_type"]
    category = DRIFT_CATEGORY_MAPPING[drift_type]

    # Use the known injected features as simulated rankings.
    affected_features = list(c1_evidence["affected_features"])

    # The no-drift case needs a technical placeholder because
    # the C2 contract requires a root_cause_feature string.
    if sid == "S00":
        root_cause_feature = "age"
        ranked_features = []
    else:
        if not affected_features:
            raise ValueError(f"{sid}: No affected features supplied")
        root_cause_feature = affected_features[0]
        ranked_features = list(affected_features)

    params = SCENARIO_DIAGNOSIS_PARAMS.get(
        sid, {"stability": 0.80, "impact": 0.50, "window": 3}
    )

    analysis = analyze_affected_slice(
        reference_df=reference_df,
        current_df=current_df,
        rule=rule,
        affected_features=affected_features,
        feature_types=feature_types,
    )

    payload = {
        "root_cause_feature": root_cause_feature,
        "root_cause_category": category,
        "affected_slice_rule": rule,
        "affected_features_ranked": ranked_features,
        "affected_value_ranges": analysis["affected_value_ranges"],
        "slice_support": analysis["slice_support"],
        "diagnosis_stability": params["stability"],
        "drift_impact_score": params["impact"],
        "temporal_window": params["window"],
    }

    validate_c2_payload(payload, contract, reference_df)

    provenance = {
        "scenario_id": sid,
        "uses_actual_temporal_shap": False,
        "uses_actual_sliceline": False,
        "simulation_method": "Heuristic ground-truth mapping and empirical slice support measurement",
        "slice_rule": rule,
        "current_count": analysis["current_count"],
        "reference_count": analysis["reference_count"],
        "measured_slice_support": analysis["slice_support"],
        "c1_predicted_harm": c1_evidence.get("predicted_harm", 0.0),
        "diagnosis_params": params,
        "notes": (
            "Simulated diagnosis specification for C3 interface testing. "
            "Root causes, slice definitions, and impact scores are heuristic stand-ins."
        ),
    }

    return payload, provenance


def main():
    """Entry point for simulated C2 diagnosis generation."""
    print("=" * 70)
    print("SIMULATED COMPONENT 2 DIAGNOSIS GENERATOR (S00 – S07)")
    print(f"Contract: {CONTRACT_PATH}")
    print(f"Output directory: {OUTPUT_DIRECTORY}")
    print("=" * 70)

    config = load_json(CONFIG_PATH)
    contract = load_json(CONTRACT_PATH)
    reference_df = pd.read_csv(REFERENCE_PATH)

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    provenance_records = {}

    for scenario in config["scenarios"]:
        sid = scenario["id"]
        c1_path = C1_DIRECTORY / f"c1_evidence_{sid}.json"
        c1_evidence = load_json(c1_path)

        current_path = DRIFT_DIRECTORY / f"{sid}.csv"
        current_df = pd.read_csv(current_path)

        payload, provenance = build_c2_diagnosis(
            scenario=scenario,
            c1_evidence=c1_evidence,
            reference_df=reference_df,
            current_df=current_df,
            feature_types=config["feature_types"],
            contract=contract,
        )

        output_path = OUTPUT_DIRECTORY / f"c2_diagnosis_{sid}.json"
        output_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

        provenance_records[sid] = provenance

        print(
            f"{sid} | "
            f"category={payload['root_cause_category']:<12} | "
            f"support={payload['slice_support']:.2%} | "
            f"stability={payload['diagnosis_stability']:.2f} | "
            f"impact={payload['drift_impact_score']:.2f}"
        )

    provenance_path = OUTPUT_DIRECTORY / "c2_simulation_provenance.json"
    provenance_document = {
        "component": "C2",
        "simulation": True,
        "purpose": "Independent C3 integration testing",
        "official_contract": str(CONTRACT_PATH.relative_to(ROOT)),
        "uses_actual_temporal_shap": False,
        "uses_actual_sliceline": False,
        "scenarios": provenance_records,
    }
    provenance_path.write_text(
        json.dumps(provenance_document, indent=2),
        encoding="utf-8",
    )

    print("=" * 70)
    print(f"Generated diagnoses: {len(provenance_records)}")
    print(f"Provenance: {provenance_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
