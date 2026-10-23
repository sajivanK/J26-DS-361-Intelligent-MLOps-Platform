"""Explicit UCI research scenarios. Real integration uses live_repair directly."""
from pathlib import Path
import json
import sys
import uuid

import pandas as pd

from component3.src.core.model_loader import load_trusted_pickle
from component3.src.contracts.safe_slice_evaluator import evaluate_slice_rule
from component3.src.live_repair import execute_live_repair, response_for_package

REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_DIR = REPO_ROOT / "shared"
REF_CSV_PATH = SHARED_DIR / "uci_adult/data/reference.csv"
TEST_CSV_PATH = SHARED_DIR / "uci_adult/data/test.csv"
MODEL_PATH = SHARED_DIR / "uci_adult/models/base_model_adult.pkl"
FEAT_JSON_PATH = SHARED_DIR / "uci_adult/feature_importance_adult.json"

UCI_SCENARIOS = {
    key: {"id": key, "name": name, "drift_type": kind, "features": features,
          "magnitude": magnitude, "diagnosis_stability": stability,
          "description": "Controlled simulation; C1/C2 evidence is simulated, not live monitoring."}
    for key, name, kind, features, magnitude, stability in [
        ("education_num_marginal", "Education shift", "marginal_single", ["education_num"], .85, .88),
        ("age_marginal", "Age shift", "marginal_single", ["age"], .8, .85),
        ("capital_gain_marginal", "Capital gain shift", "marginal_single", ["capital_gain"], .75, .9),
        ("education_occupation_interaction", "Education and occupation interaction", "interaction", ["education_num", "occupation"], .65, .82),
        ("multi_feature_severe", "Education and age shift", "marginal_multi", ["education_num", "age"], .7, .92),
        ("benign_native_country", "Native country gate example", "marginal_single", ["native_country"], .3, .4),
    ]
}


def load_shared_uci_artifacts():
    return (pd.read_csv(REF_CSV_PATH), load_trusted_pickle(MODEL_PATH, trusted=True),
            json.loads(FEAT_JSON_PATH.read_text(encoding="utf-8")))


def get_scenario_evidence(scenario_key="education_num_marginal", seed=42):
    if scenario_key not in UCI_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_key}")
    if str(SHARED_DIR) not in sys.path:
        sys.path.insert(0, str(SHARED_DIR))
    from injection.inject_drift import inject_drift
    from shared.contracts.c1_fake_output import build_fake_c1_output

    scenario = UCI_SCENARIOS[scenario_key]
    reference, model, metadata = load_shared_uci_artifacts()
    # Use a distinct test pool rather than corrupting reference training rows.
    test = pd.read_csv(TEST_CSV_PATH)
    injected = inject_drift(reference_df=reference, model=model, drift_type=scenario["drift_type"],
        features=scenario["features"], magnitude=scenario["magnitude"],
        feature_types=metadata.get("feature_types", {}), seed=seed,
        source_df=test, batch_size=min(1000, len(test)), paired=True)
    current, truth = injected["drifted_df"], injected["ground_truth"]
    feature = scenario["features"][0]
    threshold = float(current[feature].median())
    rule = f"{feature} > {threshold!r}"
    support = float(evaluate_slice_rule(current, rule).mean())
    c1 = build_fake_c1_output(truth, seed=seed)
    # The true measured drop is preserved; no preset overwrites.
    c2 = {"root_cause_feature": feature, "root_cause_category": "interaction" if scenario["drift_type"] == "interaction" else "marginal",
        "affected_slice_rule": rule, "affected_features_ranked": scenario["features"],
        "affected_value_ranges": {}, "slice_support": support,
        "diagnosis_stability": scenario["diagnosis_stability"],
        "drift_impact_score": max(0., min(1., abs(float(truth["actual_accuracy_drop"])) / 100.)),
        "temporal_window": 1}
    return {"scenario": scenario, "c1": c1, "c2": c2, "ground_truth": truth, "drifted_df": current}


def execute_c3_repair_flow(scenario_key="education_num_marginal", run_id=None, epochs=15,
                           custom_c1=None, custom_c2=None):
    """Simulation only; the same strict engine evaluates its actual artifacts."""
    evidence = get_scenario_evidence(scenario_key)
    directory = REPO_ROOT / "component3/data/scenarios" / uuid.uuid4().hex
    directory.mkdir(parents=True, exist_ok=True)
    current = evidence["drifted_df"]
    current_path = directory / "current.csv"
    current.to_csv(current_path, index=False)
    c1 = dict(custom_c1) if custom_c1 is not None else evidence["c1"]
    # Simulation provenance is explicit, including the materialized batch URI.
    c1.update(model_id="adult-shared-demo", current_dataset_uri=str(current_path),
              reference_dataset_uri=str(REF_CSV_PATH))
    features = [f for f in current.columns if f != "target"]
    payload = {"c1_evidence": c1, "c2_diagnosis": custom_c2 if custom_c2 is not None else evidence["c2"],
        "data_context": {"model_id": "adult-shared-demo", "model_uri": str(MODEL_PATH),
            "dataset_id": "uci", "reference_dataset_uri": str(REF_CSV_PATH),
            "current_dataset_uri": str(current_path), "target_column": "target", "label_status": "AVAILABLE",
            "input_representation": "model_ready", "feature_columns": features,
            "categorical_features": [f for f in ("workclass", "marital_status", "occupation", "relationship", "race", "sex", "native_country") if f in features],
            "generator_epochs": epochs}}
    package, log = execute_live_repair(payload)
    result = response_for_package(package, log)
    result["mode"] = "simulation"
    result["upstream_evidence"]["scenario"] = evidence["scenario"]
    return result
