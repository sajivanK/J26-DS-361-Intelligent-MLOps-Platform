"""Build, validate, and clean the repair experiment dataset for Sprint 06 prescription engine."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "component3" / "results" / "repair_experiments"
DATA_DIR = ROOT / "component3" / "data"
C1_DIR = DATA_DIR / "simulated_c1"
C2_DIR = DATA_DIR / "simulated_c2"


def load_c1_c2_context(scenario: str) -> dict[str, Any]:
    """Load simulated C1 and C2 context for a drift scenario."""
    context: dict[str, Any] = {
        "drift_type": "unknown",
        "drift_severity": "unknown",
        "affected_features": "unknown",
        "slice_support": 0.05,
        "diagnosis_stability": 0.5,
    }
    c1_path = C1_DIR / f"c1_evidence_{scenario}.json"
    c2_path = C2_DIR / f"c2_diagnosis_{scenario}.json"

    if c1_path.is_file():
        try:
            with c1_path.open("r", encoding="utf-8") as f:
                c1_data = json.load(f)
                context["drift_type"] = c1_data.get("drift_type", "unknown")
                context["drift_severity"] = c1_data.get("drift_severity", "unknown")
                context["affected_features"] = ",".join(c1_data.get("affected_features", []))
        except Exception:
            pass

    if c2_path.is_file():
        try:
            with c2_path.open("r", encoding="utf-8") as f:
                c2_data = json.load(f)
                context["slice_support"] = float(c2_data.get("slice_support", 0.05))
                context["diagnosis_stability"] = float(c2_data.get("diagnosis_stability", 0.5))
        except Exception:
            pass

    return context


def build_raw_dataset(output_dir: Path | None = DATA_DIR) -> pd.DataFrame:
    """Extract and aggregate all JSON experiment artifacts into a tabular DataFrame."""
    json_files = list(RESULTS_DIR.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No experiment JSON files found in {RESULTS_DIR}")

    rows: list[dict[str, Any]] = []

    for file_path in sorted(json_files):
        try:
            with file_path.open("r", encoding="utf-8") as f:
                exp = json.load(f)

            scenario = exp.get("scenario", "S04")
            c_ctx = load_c1_c2_context(scenario)
            scores = exp.get("summary_scores", {})
            val_decision = exp.get("acceptance_decision", {})

            row = {
                "experiment_id": exp.get("experiment_id", file_path.stem),
                "dataset_id": exp.get("dataset", "uci_adult"),
                "drift_scenario": scenario,
                "drift_type": c_ctx["drift_type"],
                "drift_severity": c_ctx["drift_severity"],
                "affected_features": c_ctx["affected_features"],
                "slice_support": c_ctx["slice_support"],
                "diagnosis_stability": c_ctx["diagnosis_stability"],
                "slice_rule": exp.get("slice_rule", ""),
                "generator": exp.get("generator", "UNKNOWN"),
                "epochs": int(exp.get("epochs", 1)),
                "repair_quantity": int(exp.get("requested_samples", 0)),
                "generated_quantity": int(exp.get("generated_samples", 0)),
                "seed": int(exp.get("seed", 42)),
                "label_authorized": bool(exp.get("label_authorized", False)),
                "label_source": str(exp.get("label_source") or ("teacher_pseudo_label" if exp.get("label_authorized") else "unlabeled")),
                "target_coverage": float(scores.get("target_coverage", 0.0) or 0.0),
                "fidelity_score": float(scores.get("fidelity_score", 0.0) or 0.0),
                "privacy_status": str(scores.get("privacy_status", "UNKNOWN")),
                "dcr_p05": float(scores.get("dcr_p05", 0.0) or 0.0),
                "exact_match_count": int(scores.get("exact_match_count", 0) or 0),
                "utility_score": float(scores.get("utility_score", 0.0) or 0.0),
                "delta_overall_f1": float(scores.get("delta_overall_f1", 0.0) or 0.0),
                "delta_affected_slice_f1": float(scores.get("delta_affected_slice_f1", 0.0) or 0.0),
                "delta_non_affected_f1": float(scores.get("delta_non_affected_f1", 0.0) or 0.0),
                "delta_baseline_slice_f1": float(scores.get("delta_baseline_slice_f1", 0.0) or 0.0),
                "training_seconds": float(exp.get("training_seconds", 0.0)),
                "sampling_seconds": float(exp.get("sampling_seconds", 0.0)),
                "total_seconds": float(exp.get("total_seconds", 0.0)),
                "decision": str(scores.get("decision", val_decision.get("decision", "UNKNOWN"))),
                "package_status": str(scores.get("package_status", val_decision.get("package_status", "REJECTED"))),
            }

            # Prescriptive Viability Label for Sprint 06:
            # Delineates algorithmic candidate viability for meta-learner training
            # from strict production deployment acceptance (package_status).
            # A repair is viable for prescriptive recommendation if:
            # 1. Target coverage >= 80% (structural fidelity to drifted slice)
            # 2. Privacy is PASS (zero exact copies, healthy DCR, zero near-duplicates)
            # 3. Controlled affected slice delta F1 >= -0.035 (non-degrading utility)
            # 4. Composite fidelity score >= 0.70
            is_viable = int(
                row["target_coverage"] >= 0.80
                and row["privacy_status"] == "PASS"
                and row["delta_affected_slice_f1"] >= -0.035
                and row["fidelity_score"] >= 0.70
            )
            row["prescriptive_viability_label"] = is_viable

            rows.append(row)
        except Exception as exc:
            print(f"Skipping corrupt artifact {file_path.name}: {exc}")

    df_raw = pd.DataFrame(rows)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_path = output_dir / "repair_experiments_raw.csv"
        raw_json_path = output_dir / "repair_experiments_raw.json"
        df_raw.to_csv(raw_path, index=False)
        df_raw.to_json(raw_json_path, orient="records", indent=2)
        print(f"Raw repair experiments dataset saved: {raw_path} & {raw_json_path} ({len(df_raw)} records)")
    return df_raw


def clean_and_validate_dataset(
    df_raw: pd.DataFrame,
    output_dir: Path | None = DATA_DIR,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply data quality checks, remove invalid records, and produce audit metadata."""
    audit: dict[str, Any] = {
        "raw_record_count": int(len(df_raw)),
        "excluded_records": [],
        "duplicate_experiment_ids": int(df_raw["experiment_id"].duplicated().sum()),
        "validation_passed": True,
    }

    df_clean = df_raw.copy()

    # 1. Deduplicate by experiment_id
    if audit["duplicate_experiment_ids"] > 0:
        dup_ids = df_clean[df_clean["experiment_id"].duplicated(keep="last")]["experiment_id"].tolist()
        audit["excluded_records"].append({"reason": "duplicate_experiment_id", "ids": dup_ids})
        df_clean = df_clean.drop_duplicates(subset=["experiment_id"], keep="last")

    # 2. Check metric bounds
    invalid_coverage = df_clean[(df_clean["target_coverage"] < 0.0) | (df_clean["target_coverage"] > 1.0)]
    if not invalid_coverage.empty:
        audit["excluded_records"].append({"reason": "invalid_coverage_range", "count": len(invalid_coverage)})
        df_clean = df_clean.drop(index=invalid_coverage.index)

    invalid_fidelity = df_clean[(df_clean["fidelity_score"] < 0.0) | (df_clean["fidelity_score"] > 1.0)]
    if not invalid_fidelity.empty:
        audit["excluded_records"].append({"reason": "invalid_fidelity_range", "count": len(invalid_fidelity)})
        df_clean = df_clean.drop(index=invalid_fidelity.index)

    # 3. Check impossible negative runtimes
    invalid_runtime = df_clean[df_clean["total_seconds"] < 0.0]
    if not invalid_runtime.empty:
        audit["excluded_records"].append({"reason": "negative_runtime", "count": len(invalid_runtime)})
        df_clean = df_clean.drop(index=invalid_runtime.index)

    audit["clean_record_count"] = int(len(df_clean))
    audit["viability_distribution"] = df_clean["prescriptive_viability_label"].value_counts().to_dict()
    audit["generator_distribution"] = df_clean["generator"].value_counts().to_dict()
    audit["scenario_distribution"] = df_clean["drift_scenario"].value_counts().to_dict()
    audit["timestamp_utc"] = datetime.now(timezone.utc).isoformat()

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        clean_path = output_dir / "repair_experiments_clean.csv"
        meta_path = output_dir / "repair_experiments_metadata.json"
        clean_json_path = output_dir / "repair_experiments_clean.json"

        df_clean.to_csv(clean_path, index=False)
        df_clean.to_json(clean_json_path, orient="records", indent=2)

        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2)

        print(f"Clean repair experiments dataset saved: {clean_path} & {clean_json_path} ({len(df_clean)} records)")
        print(f"Dataset audit metadata saved: {meta_path}")

    return df_clean, audit


if __name__ == "__main__":
    df_raw = build_raw_dataset()
    df_clean, audit = clean_and_validate_dataset(df_raw)
    print("\n=== DATASET CREATION AND CLEANING COMPLETE ===")
    print(f"Raw rows: {audit['raw_record_count']} | Clean rows: {audit['clean_record_count']}")
    print(f"Viability labels: {audit['viability_distribution']}")
    print(f"Generators: {audit['generator_distribution']}")
    print(f"Scenarios: {audit['scenario_distribution']}\n")
