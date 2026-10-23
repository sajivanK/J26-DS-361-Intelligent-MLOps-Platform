"""
Generate contract-shaped, explicitly simulated C1 evidence for C3 tests.
These payloads are NOT predictions from the real Component 1 engine.
Synthetic detector values and harm intervals are placeholders.
Run from repository root:
    .\.venv\Scripts\python.exe component3/src/experiments/simulate_c1_evidence.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "component3/configs/drift_scenarios.json"
CONTRACT_PATH = ROOT / "shared/contracts/c1_output_schema.json"
METRICS_PATH = ROOT / "component3/data/drifted/drift_performance_metrics.csv"
OUTPUT_DIR = ROOT / "component3/data/simulated_c1"
SCENARIO_IDS = [f"S{i:02d}" for i in range(8)]

DETECTOR_NAMES = ("ks", "chi2", "psi", "mmd", "lsdd", "classifier_auc")
CONFIDENCE_NAMES = ("mean_confidence", "confidence_std", "pct_low_confidence")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def round4(value):
    return round(float(value), 4)


def get_scenario_config(config, scenario_id):
    """Find scenario configuration without assuming its container name."""
    for key in ("scenarios", "drift_scenarios"):
        scenarios = config.get(key, [])
        if isinstance(scenarios, dict):
            scenarios = list(scenarios.values())
        for scenario in scenarios:
            if not isinstance(scenario, dict):
                continue
            candidate = (
                scenario.get("scenario_id")
                or scenario.get("id")
                or scenario.get("name")
            )
            if candidate == scenario_id:
                return scenario
    return None


def get_severity_label(drop_pp):
    """Determine severity label matching shared framework."""
    if drop_pp < 1.0:
        return "none"
    elif drop_pp < 3.0:
        return "low"
    elif drop_pp < 7.0:
        return "moderate"
    elif drop_pp < 15.0:
        return "high"
    else:
        return "critical"


def validate_c1_payload(payload, contract_schema):
    """Strictly validate generated payload against the official C1 schema."""
    fields_def = contract_schema.get("fields", {})
    required_fields = [k for k, v in fields_def.items() if v.get("required") is True]

    for req in required_fields:
        assert req in payload, f"Missing required field: {req}"

    # Validate allowed values
    drift_types = fields_def["drift_type"].get("allowed_values", [])
    assert payload["drift_type"] in drift_types, (
        f"Invalid drift_type: {payload['drift_type']} not in {drift_types}"
    )

    severities = fields_def["drift_severity"].get("allowed_values", [])
    assert payload["drift_severity"] in severities, (
        f"Invalid drift_severity: {payload['drift_severity']} not in {severities}"
    )

    harm_metrics = fields_def["harm_metric"].get("allowed_values", [])
    assert payload["harm_metric"] in harm_metrics, (
        f"Invalid harm_metric: {payload['harm_metric']} not in {harm_metrics}"
    )

    assert 0.0 <= payload["predicted_harm"] <= 1.0, "predicted_harm out of range"
    assert len(payload["harm_interval"]) == 2, "harm_interval must be length 2"
    assert payload["harm_interval"][0] <= payload["harm_interval"][1], (
        "harm_interval lower > upper"
    )

    # Validate detector_scores object
    det = payload["detector_scores"]
    for d_name in DETECTOR_NAMES:
        assert d_name in det, f"Missing detector score: {d_name}"
        assert isinstance(det[d_name], float), f"Detector {d_name} not a float"

    # Validate confidence_stats object
    conf = payload["confidence_stats"]
    for c_name in CONFIDENCE_NAMES:
        assert c_name in conf, f"Missing confidence stat: {c_name}"
        assert isinstance(conf[c_name], float), f"Confidence stat {c_name} not a float"

    assert isinstance(payload["detectors_fired"], int), "detectors_fired not integer"
    assert 0 <= payload["detectors_fired"] <= len(DETECTOR_NAMES), (
        f"detectors_fired must be between 0 and {len(DETECTOR_NAMES)}"
    )
    assert isinstance(payload["affected_features"], list), "affected_features not list"


def main():
    config = load_json(CONFIG_PATH)
    contract = load_json(CONTRACT_PATH)
    metrics_df = pd.read_csv(METRICS_PATH).set_index("scenario")

    reference_df = pd.read_csv(ROOT / config["reference_path"])
    model = joblib.load(ROOT / config["model_path"])
    target_col = config.get("target_col", "target")
    feature_types = config.get("feature_types", {})

    continuous_cols = [
        c for c, t in feature_types.items()
        if t == "continuous" and c in reference_df.columns
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("GENERATING SIMULATED COMPONENT 1 EVIDENCE (S00 – S07)")
    print(f"Output directory: {OUTPUT_DIR.relative_to(ROOT)}")
    print("=" * 75)

    provenance_records = {}

    for s_id in SCENARIO_IDS:
        s_config = get_scenario_config(config, s_id)
        if s_config is None:
            raise KeyError(f"Configuration not found for scenario: {s_id}")

        dataset_path = ROOT / f"component3/data/drifted/{s_id}.csv"
        drifted_df = pd.read_csv(dataset_path)

        X = drifted_df.drop(columns=[target_col])

        # 1. Real computation: Model confidence statistics on drifted batch
        probs = model.predict_proba(X)
        max_probs = np.max(probs, axis=1)
        confidence_stats = {
            "mean_confidence": round4(np.mean(max_probs)),
            "confidence_std": round4(np.std(max_probs)),
            "pct_low_confidence": round4(np.mean(max_probs < 0.6)),
        }

        # 2. Real computation: Mean 2-sample KS statistic on continuous features
        ks_stats = []
        for col in continuous_cols:
            stat, _ = ks_2samp(reference_df[col].values, drifted_df[col].values)
            ks_stats.append(stat)
        mean_ks = round4(np.mean(ks_stats)) if ks_stats else 0.0

        # 3. Ground-truth lookup: Performance degradation proxy
        metric_row = metrics_df.loc[s_id]
        acc_drop_pp = float(metric_row["accuracy_drop_pp"])
        predicted_harm = round4(max(0.0, acc_drop_pp / 100.0))

        # Explicitly simulated interval (±0.015 proxy placeholder)
        harm_interval = [
            round4(max(0.0, predicted_harm - 0.015)),
            round4(predicted_harm + 0.015),
        ]

        # Severity classification
        severity = get_severity_label(acc_drop_pp)

        # 4. Detector scores simulation
        if s_id == "S00":
            chi2_score = 0.0050
            psi_score = 0.0100
            mmd_score = 0.0050
            lsdd_score = 0.0050
            c2st_auc = 0.5000
            detectors_fired = 0
        else:
            # Heuristic mock detector scores for interface testing only
            is_cat_or_interaction = s_config["drift_type"] in (
                "marginal_single", "interaction", "combined"
            ) and any(feature_types.get(f) == "categorical" for f in s_config.get("features", []))
            chi2_score = round4(0.12 if is_cat_or_interaction else 0.02)
            psi_score = round4(max(0.05, mean_ks * 1.4))
            mmd_score = round4(max(0.04, mean_ks * 0.7 + 0.03))
            lsdd_score = round4(max(0.03, mean_ks * 0.6 + 0.02))
            c2st_auc = round4(min(0.95, 0.52 + mean_ks * 0.5 + (0.15 if s_config["drift_type"] in ("interaction", "combined") else 0.05)))

            # Count detectors fired against standard alarms
            fired = 0
            if mean_ks > 0.05: fired += 1
            if chi2_score > 0.05: fired += 1
            if psi_score > 0.10: fired += 1
            if mmd_score > 0.05: fired += 1
            if lsdd_score > 0.05: fired += 1
            if c2st_auc > 0.55: fired += 1
            detectors_fired = fired

        detector_scores = {
            "ks": mean_ks,
            "chi2": chi2_score,
            "psi": psi_score,
            "mmd": mmd_score,
            "lsdd": lsdd_score,
            "classifier_auc": c2st_auc,
        }

        # Affected features from scenario definition
        affected_features = list(s_config.get("features", []))
        for pair in s_config.get("interaction_pairs", []):
            for feat in pair:
                if feat not in affected_features:
                    affected_features.append(feat)

        payload = {
            "evidence_version": f"c1-sim-{s_id.lower()}",
            "model_id": "adult-xgb-v1",
            "reference_dataset_uri": config["reference_path"],
            "current_dataset_uri": f"component3/data/drifted/{s_id}.csv",
            "drift_type": s_config["drift_type"],
            "drift_severity": severity,
            "predicted_harm": predicted_harm,
            "harm_interval": harm_interval,
            "harm_metric": "accuracy_drop",
            "affected_features": affected_features,
            "detector_scores": detector_scores,
            "detectors_fired": detectors_fired,
            "confidence_stats": confidence_stats,
        }

        # Validate before writing
        validate_c1_payload(payload, contract)

        # Write to JSON
        out_file = OUTPUT_DIR / f"c1_evidence_{s_id}.json"
        out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        provenance_records[s_id] = {
            "measured_fields": [
                "confidence_stats (computed from model probabilities)",
                "detector_scores.ks (mean two-sample KS statistic)",
            ],
            "known_experiment_metadata": [
                "affected_features (injected scenario features)",
                "drift_type (scenario configuration)",
                "actual_accuracy_drop_pp (observed performance degradation)",
            ],
            "simulation_proxies": [
                "predicted_harm (derived from observed accuracy drop; not C1 prediction)",
                "drift_severity (derived from observed accuracy drop)",
                "detectors_fired (count using simulated detector scores)",
            ],
            "simulated_placeholders": [
                "harm_interval (uncalibrated +/-0.015 margin)",
                "detector_scores.chi2",
                "detector_scores.psi",
                "detector_scores.mmd",
                "detector_scores.lsdd",
                "detector_scores.classifier_auc",
            ],
            "actual_accuracy_drop_pp": acc_drop_pp,
            "predicted_harm_proportion": predicted_harm,
            "severity": severity,
            "detectors_fired": detectors_fired,
        }

        print(
            f"Generated {s_id}: severity={severity:8s} | "
            f"harm={predicted_harm:.4f} | "
            f"fired={detectors_fired}/6 | "
            f"file={out_file.name}"
        )

    # Save provenance metadata
    provenance_doc = {
        "provenance_metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "generator_script": "component3/src/experiments/simulate_c1_evidence.py",
            "target_schema": "shared/contracts/c1_output_schema.json",
            "schema_version": contract.get("schema_version", "1.0"),
            "disclaimer": (
                "These payloads are explicitly SIMULATED stand-ins for Component 1 output. "
                "Harm intervals and non-KS detector scores are synthetic placeholders. "
                "Confidence stats and KS values were computed on the actual batches."
            ),
        },
        "scenarios": provenance_records,
    }

    prov_file = OUTPUT_DIR / "c1_simulation_provenance.json"
    prov_file.write_text(json.dumps(provenance_doc, indent=2), encoding="utf-8")

    print("\n" + "=" * 75)
    print(f"Saved Provenance: {prov_file.relative_to(ROOT)}")
    print(f"Total simulated C1 evidence payloads generated: {len(SCENARIO_IDS)}")
    print("=" * 75)


if __name__ == "__main__":
    main()
