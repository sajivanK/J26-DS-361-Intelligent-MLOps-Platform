"""
Sprint 02: Controlled UCI Adult drift experiments.
Run from the repository root:
    .\.venv\Scripts\python.exe component3/src/experiments/run_drift_scenarios.py
"""

import argparse
import json
import sys
from pathlib import Path
import joblib
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "shared"))

from injection.inject_drift import inject_drift  # noqa: E402

CONFIG_PATH = REPO_ROOT / "component3/configs/drift_scenarios.json"
OUTPUT_DIR = REPO_ROOT / "component3/data/drifted"
NUMERICAL_SCENARIOS = {
    "S00", "S01", "S02", "S03",
    "S04", "S05", "S06", "S07"
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate controlled drift scenario datasets"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for generated scenario CSV files",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    reference_df = pd.read_csv(REPO_ROOT / config["reference_path"])
    model = joblib.load(REPO_ROOT / config["model_path"])
    batch_size = config["batch_size"]
    seed = config["seed"]
    feature_types = config.get("feature_types", {})
    target_col = config.get("target_col", "target")

    # Sample a reproducible, consistent evaluation reference batch
    eval_ref_batch = reference_df.sample(
        n=batch_size, random_state=seed
    ).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SPRINT 02 — NUMERICAL DRIFT SCENARIOS (S01, S02, S05)")
    print(f"Reference evaluation batch size: {len(eval_ref_batch)} rows, seed={seed}")
    print("=" * 70)

    results = []

    for scenario in config["scenarios"]:
        s_id = scenario["id"]
        if s_id not in NUMERICAL_SCENARIOS:
            continue

        print(f"\n--- Running Scenario {s_id}: {scenario['name']} ---")
        print(f"  Drift type: {scenario['drift_type']}")
        print(f"  Features:   {scenario['features']}")
        print(f"  Magnitude:  {scenario['magnitude']}")

        result = inject_drift(
            reference_df=eval_ref_batch,
            model=model,
            drift_type=scenario["drift_type"],
            features=scenario["features"],
            magnitude=scenario["magnitude"],
            feature_types=feature_types,
            interaction_pairs=scenario.get("interaction_pairs") or None,
            batch_size=batch_size,
            target_col=target_col,
            seed=seed,
        )

        drifted_df = result["drifted_df"]
        gt = result["ground_truth"]

        # Save drifted dataset
        out_csv = output_dir / f"{s_id}.csv"
        drifted_df.to_csv(out_csv, index=False)

        print(f"  Saved to:            {out_csv.relative_to(REPO_ROOT)}")
        print(f"  Drifted batch shape: {drifted_df.shape}")
        print(f"  Baseline Accuracy:   {gt['baseline_accuracy'] * 100:.2f}%")
        print(f"  Drifted Accuracy:    {gt['drifted_accuracy'] * 100:.2f}%")
        print(f"  Accuracy Drop:       {gt['actual_accuracy_drop']:.2f}%")
        print(f"  Baseline AUC:        {gt['baseline_auc']}")
        print(f"  Drifted AUC:         {gt['drifted_auc']}")
        print(f"  Severity:            {gt['severity']}")
        print(f"  Is Malignant:        {gt['is_malignant']}")
        if gt.get("per_feature_ks"):
            print(f"  Per-feature KS:      {gt['per_feature_ks']}")

        results.append({
            "scenario_id": s_id,
            "name": scenario["name"],
            "shape": list(drifted_df.shape),
            "ground_truth": gt,
        })

    print("\n" + "=" * 70)
    print("NUMERICAL SCENARIOS COMPLETED")
    print(f"Generated datasets: {[r['scenario_id'] for r in results]}")
    print("=" * 70)


if __name__ == "__main__":
    main()
