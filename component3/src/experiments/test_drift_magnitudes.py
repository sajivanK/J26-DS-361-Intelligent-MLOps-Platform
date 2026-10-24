"""
Sprint 02 — Subtask 02.5: Test Different Drift Magnitudes
Tests magnitude sensitivity across high- and low-importance features.
Run from repository root:
    .\.venv\Scripts\python.exe component3/src/experiments/test_drift_magnitudes.py
"""

import json
import sys
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "shared"))

from injection.inject_drift import inject_drift  # noqa: E402

CONFIG_PATH = ROOT / "component3/configs/drift_scenarios.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

reference_df = pd.read_csv(ROOT / CONFIG["reference_path"])
model = joblib.load(ROOT / CONFIG["model_path"])

# Sample reproducible evaluation batch
batch = reference_df.sample(
    n=CONFIG["batch_size"], random_state=CONFIG["seed"]
).reset_index(drop=True)

features = ["age", "capital_loss"]
magnitudes = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

print("=" * 75)
print("SPRINT 02 — SUBTASK 02.5: MAGNITUDE SENSITIVITY SWEEP")
print(f"Features: {features} | Magnitudes: {magnitudes}")
print(f"Batch size: {len(batch)} | Seed: {CONFIG['seed']}")
print("=" * 75)

records = []
for feature in features:
    print(f"\n--- Feature: {feature} ---")
    for magnitude in magnitudes:
        result = inject_drift(
            reference_df=batch.copy(),
            model=model,
            drift_type="marginal_single",
            features=[feature],
            magnitude=magnitude,
            feature_types=CONFIG["feature_types"],
            batch_size=len(batch),
            target_col=CONFIG["target_col"],
            seed=CONFIG["seed"],
        )
        truth = result["ground_truth"]
        # In inject_drift, the key is 'actual_accuracy_drop'
        acc_drop = truth.get("actual_accuracy_drop", truth.get("accuracy_drop_pct", 0.0))

        record = {
            "feature": feature,
            "magnitude": magnitude,
            "accuracy_drop_pct": acc_drop,
            "baseline_accuracy": truth["baseline_accuracy"],
            "drifted_accuracy": truth["drifted_accuracy"],
            "severity": truth["severity"],
            "is_malignant": truth["is_malignant"],
        }
        records.append(record)
        print(
            f"{feature:15s} | "
            f"magnitude={magnitude:.1f} | "
            f"drop={record['accuracy_drop_pct']:.2f} pp | "
            f"severity={record['severity']:8s} | "
            f"malignant={record['is_malignant']}"
        )

output_dir = ROOT / "component3/data/drifted"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "magnitude_sensitivity.csv"
pd.DataFrame(records).to_csv(output_path, index=False)

print("\n" + "=" * 75)
print(f"Saved: {output_path.relative_to(ROOT)}")
print(f"Total experiments: {len(records)}")
print("=" * 75)
