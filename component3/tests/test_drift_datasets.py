"""
Sprint 02 — Subtask 02.6: Validate Generated Drifted Datasets.
Checks structural integrity, schema, label preservation, and isolated perturbations.
Run from repository root:
    .\.venv\Scripts\python.exe component3/tests/test_drift_datasets.py
"""

import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

config_path = ROOT / "component3/configs/drift_scenarios.json"
config = json.loads(config_path.read_text(encoding="utf-8"))

reference = pd.read_csv(ROOT / config["reference_path"])
batch = reference.sample(
    n=config["batch_size"], random_state=config["seed"]
).reset_index(drop=True)

output_dir = ROOT / "component3/data/drifted"
categorical_features = {
    feature
    for feature, feature_type in config["feature_types"].items()
    if feature_type == "categorical"
}

print("=" * 70)
print("VALIDATING GENERATED DRIFT DATASETS")
print(f"Expected batch shape: {batch.shape}")
print("=" * 70)

for scenario in config["scenarios"]:
    scenario_id = scenario["id"]
    path = output_dir / f"{scenario_id}.csv"

    if not path.exists():
        if scenario_id == "S00":
            print("S00: Not generated yet — pending no-drift test.")
            continue
        raise FileNotFoundError(path)

    drifted = pd.read_csv(path)

    assert drifted.shape == batch.shape, (
        f"{scenario_id}: unexpected dataset shape {drifted.shape} vs {batch.shape}"
    )
    assert list(drifted.columns) == list(batch.columns), (
        f"{scenario_id}: column mismatch"
    )
    assert drifted.dtypes.equals(batch.dtypes), (
        f"{scenario_id}: data type mismatch"
    )
    assert not drifted.isna().any().any(), (
        f"{scenario_id}: missing values found"
    )

    if scenario_id != "S00":
        assert drifted["target"].equals(batch["target"]), (
            f"{scenario_id}: target labels changed"
        )

    allowed_to_change = set(scenario["features"])
    for feature_a, feature_b in scenario.get("interaction_pairs", []):
        # The shared interaction injector shuffles feature_b.
        allowed_to_change.add(feature_b)

    if scenario_id != "S00":
        for column in batch.columns:
            if column not in allowed_to_change:
                if pd.api.types.is_float_dtype(batch[column]):
                    import numpy as np
                    assert np.allclose(drifted[column], batch[column], atol=1e-7), (
                        f"{scenario_id}: unexpected change in {column}"
                    )
                else:
                    assert drifted[column].equals(batch[column]), (
                        f"{scenario_id}: unexpected change in {column}"
                    )

    for column in categorical_features:
        original_categories = set(reference[column].unique())
        drifted_categories = set(drifted[column].unique())
        assert drifted_categories.issubset(original_categories), (
            f"{scenario_id}: invalid category in {column}"
        )

    print(f"PASS: {scenario_id} — shape, schema, labels and features verified")

print("\nDataset validation completed.")
