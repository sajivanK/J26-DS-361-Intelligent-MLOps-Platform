"""Inspect frozen training-group progress without reading future outcomes."""

from collections import Counter

from _common import checked_json, parser, require, resolve_data_root


def main():
    args = parser(__doc__).parse_args()
    root = resolve_data_root(args.data_root)
    frozen_dir = root / "c4_artifacts/benchmark_protocols/adult/adult-simulation-pp1-v2"
    freeze, freeze_hash = checked_json(frozen_dir, "freeze")
    manifest, manifest_hash = checked_json(frozen_dir, "scenario_manifest")
    require(manifest_hash == freeze["scenario_manifest_sha256"], "Manifest link differs.")
    require(freeze["full_execution_enabled"] is False, "Unexpected execution scope.")
    groups = manifest["groups"]
    require(len({g["group_id"] for g in groups}) == len(groups), "Duplicate groups.")
    print("FROZEN MANIFEST CHECKSUM: PASS")
    print("PLANNED GROUPS BY FOLD:", dict(Counter(g["split_role"] for g in groups)))
    runs = root / "c4_artifacts/benchmark_runs/adult/adult-simulation-pp1-v2"
    for group in sorted(groups, key=lambda g: g["group_id"]):
        if group["split_role"] != "forecast_train":
            continue
        group_id = group["group_id"]
        rows = [r for r in manifest["scenarios"] if r["group_id"] == group_id]
        require(len(rows) == 37, "Training group must contain 37 cases.")
        require(all(r["split_role"] == "forecast_train" for r in rows), "Mixed folds.")
        path = runs / "group_admissions" / group_id / "v1"
        status = "NOT_STARTED"
        if (path / "group_admissions.json").is_file():
            report, _ = checked_json(path, "group_admissions")
            require(report["group_id"] == group_id and report["freeze_sha256"] ==
                    freeze_hash, "Admission protocol/group differs.")
            records = report["records"]
            require(len(records) == 37 and {r["scenario_id"] for r in records} ==
                    {r["scenario_id"] for r in rows}, "Admission inventory differs.")
            admitted = sum(r["simulation_admission"] ==
                           "READY_FOR_SCOPED_CANDIDATE_VALIDATION" for r in records)
            status = f"ADMISSION_REPORT_PRESENT admitted={admitted} blocked={37-admitted}"
        elif path.exists() or any((runs / r["scenario_id"]).exists() for r in rows):
            status = "PARTIAL_OR_EXISTING_CASES inspect_before_execution"
        print("GROUP:", group_id, "CASES:", len(rows), "STATUS:", status)
    print("SORT ORDER: display only; no new execution order or fold assignment.")
    print("NO TRAINING, FUTURE OUTCOME READS, OR ARTIFACT WRITES EXECUTED.")


if __name__ == "__main__":
    main()
