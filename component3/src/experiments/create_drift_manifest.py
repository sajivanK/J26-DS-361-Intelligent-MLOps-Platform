"""
Create a reproducible manifest for Sprint 02 drift datasets.
Calculates SHA-256 hashes, file sizes, schema signatures, and metadata.
Run from repository root:
    .\.venv\Scripts\python.exe component3/src/experiments/create_drift_manifest.py
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "component3/configs/drift_scenarios.json"
DATA_DIR = ROOT / "component3/data/drifted"
OUTPUT_PATH = DATA_DIR / "drift_dataset_manifest.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_file(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def main():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    target_col = config.get("target_col", "target")

    print("=" * 70)
    print("CREATING REPRODUCIBLE DRIFT DATASET MANIFEST")
    print(f"Data directory: {DATA_DIR.relative_to(ROOT)}")
    print("=" * 70)

    # Reference artifacts
    ref_desc = describe_file(ROOT / config["reference_path"])
    model_desc = describe_file(ROOT / config["model_path"])
    config_desc = describe_file(CONFIG_PATH)

    scenario_manifests = []
    for scenario in config["scenarios"]:
        s_id = scenario["id"]
        csv_path = DATA_DIR / f"{s_id}.csv"
        file_desc = describe_file(csv_path)

        # Inspect dataset schema and row distribution
        df = pd.read_csv(csv_path)
        feature_cols = [c for c in df.columns if c != target_col]
        target_dist = {
            str(k): int(v)
            for k, v in df[target_col].value_counts().items()
        }

        s_entry = {
            "scenario_id": s_id,
            "name": scenario["name"],
            "drift_type": scenario["drift_type"],
            "features": scenario["features"],
            "magnitude": scenario["magnitude"],
            "interaction_pairs": scenario.get("interaction_pairs", []),
            "dataset": file_desc,
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "feature_columns": feature_cols,
            "target_distribution": target_dist,
        }
        scenario_manifests.append(s_entry)
        print(f"Hashed {s_id:4s}: {file_desc['sha256'][:16]}... ({file_desc['size_bytes']} bytes)")

    # Supporting result files as a list of entries
    results_list = []
    for extra in ["drift_performance_metrics.csv", "magnitude_sensitivity.csv"]:
        extra_path = DATA_DIR / extra
        if extra_path.exists():
            desc = describe_file(extra_path)
            results_list.append(desc)
            print(f"Hashed {extra}: {desc['sha256'][:16]}...")

    manifest = {
        "manifest_version": "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": config["experiment_name"],
        "dataset_name": config["dataset"],
        "batch_size": config["batch_size"],
        "seed": config["seed"],
        "target_column": target_col,
        "feature_types": config["feature_types"],
        "configuration": config_desc,
        "reference_dataset": ref_desc,
        "baseline_model": model_desc,
        "scenarios": scenario_manifests,
        "experimental_results": results_list,
    }

    OUTPUT_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"Saved Manifest: {OUTPUT_PATH.relative_to(ROOT)}")
    print(f"Total scenario datasets indexed: {len(scenario_manifests)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
