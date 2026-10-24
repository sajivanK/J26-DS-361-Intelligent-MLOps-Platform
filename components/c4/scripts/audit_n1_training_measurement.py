"""Read-only paired-outcome audit of the original forecast_train scenarios.

Paired row standard errors are descriptive IID-row diagnostics, not an R2 ceiling,
a conformal guarantee, or proof that finite-sample noise causes weak forecasts.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd

import diagnose_n1_development as d

TRAINING_HASH = "43dd85d45a94b7209a86481378b6eba5e48a61d4bde43a170a7e380adb065e30"


def paired_statistics(predictions, membership, targets, source_ids):
    d.require(predictions.source_row_id.is_unique and membership.source_row_id.is_unique,
              "Duplicate source-row IDs.")
    d.require(predictions.source_row_id.tolist() == membership.source_row_id.tolist() == source_ids,
              "Paired source-row identities/order differ.")
    d.require(pd.api.types.is_bool_dtype(membership.true_slice_member)
              and not membership.true_slice_member.isna().any(), "Invalid true-slice membership.")
    for column in ("target", "champion_prediction", "candidate_prediction"):
        d.require(predictions[column].isin([0, 1]).all(), "Nonbinary paired outcome.")
    truth = predictions.target.to_numpy()
    difference = (predictions.candidate_prediction.to_numpy() == truth).astype(float) - \
                 (predictions.champion_prediction.to_numpy() == truth).astype(float)
    mask = membership.true_slice_member.to_numpy()
    rows = []
    for target, population in zip(d.TARGETS, (np.ones(len(mask), bool), mask, ~mask)):
        values = difference[population]
        d.require(len(values) >= 2, "Insufficient population rows.")
        margin = float(values.mean())
        d.require(abs(margin - targets[target]) < 1e-14, "Saved training target does not match paired predictions.")
        rows.append({"target": target, "population_rows": len(values), "margin": margin,
                     "candidate_only_correct": int((values == 1).sum()),
                     "champion_only_correct": int((values == -1).sum()),
                     "paired_iid_standard_error": float(values.std(ddof=1) / np.sqrt(len(values)))})
    return rows


def load_training(root):
    folder = root / "c4_artifacts/meta_datasets/adult" / d.PROTOCOL / "forecast-training-all32-audit-v1"
    metadata = json.loads(d.checked(folder / "metadata.json", TRAINING_HASH))
    d.require(metadata["fold"] == "forecast_train" and metadata["rows"] == 106,
              "Wrong training snapshot scope.")
    frames = {}
    for name in ("features.csv", "targets.csv", "scenario_index.csv"):
        frames[name] = pd.read_csv(io.BytesIO(d.checked(folder / name, metadata["files"][name])),
                                   float_precision="round_trip")
    x = frames["features.csv"].set_index("scenario_id")
    y = frames["targets.csv"].set_index("scenario_id")
    index = frames["scenario_index.csv"]
    d.validate_tables(x, y, index)
    d.require(set(index.group_id) <= set(metadata["groups"]), "Unexpected training group.")
    return y, index


def audit(root, progress=print):
    targets, index = load_training(root)
    benchmark = root / "c4_artifacts/benchmark_runs/adult" / d.PROTOCOL
    rows = []
    source_sets = {}
    for number, row in enumerate(index.itertuples(index=False), 1):
        sid = row.scenario_id
        d.require(re.fullmatch(r"[A-Za-z0-9_-]+", sid) is not None
                  and sid.startswith(row.group_id + "-"), "Unsafe/unbound training scenario ID.")
        directory = benchmark / sid / "future_research-v1"
        contents = d.checked(directory / "outcomes.json", row.outcomes_sha256)
        outcome = json.loads(contents)
        d.require(outcome["evaluation_role"] == "sealed_future"
                  and outcome["primary_slice_basis"] == "true_injected_slice"
                  and outcome["target_units"] == "accuracy_margin_fraction", "Outcome target definition differs.")
        for target in d.TARGETS:
            d.require(abs(outcome["primary_targets"][target] - targets.loc[sid, target]) < 1e-14,
                      "Training outcome binding differs.")
        def read(name):
            return pd.read_csv(io.BytesIO(d.checked(directory / name, outcome["files"][name])),
                               float_precision="round_trip")
        predictions, membership = read("future_predictions.csv"), read("future_membership.csv")
        statistics = paired_statistics(predictions, membership, targets.loc[sid], outcome["source_row_ids"])
        rows.extend(dict(scenario_id=sid, group_id=row.group_id, **s) for s in statistics)
        source_sets[sid] = frozenset(predictions.source_row_id)
        if number % 10 == 0 or number == len(index):
            progress(f"TRAINING PAIRED RECORDS: {number}/{len(index)} verified", flush=True)
    frame = pd.DataFrame(rows)
    print("\nTRAINING MEASUREMENT DIAGNOSTIC — accuracy percentage points")
    for target in d.TARGETS:
        part = frame[frame.target == target]
        w = d.group_weights(part.group_id)
        target_sd = np.sqrt(np.average((part.margin - np.average(part.margin, weights=w))**2, weights=w))
        se = np.sqrt(np.average(part.paired_iid_standard_error**2, weights=w))
        print(f"{target}: target SD={100*target_sd:.5f} pp; root-mean-square paired IID SE={100*se:.5f} pp; "
              f"population rows={part.population_rows.min()}..{part.population_rows.max()}")
    print("Distinct saved source-row sets:", len(set(source_sets.values())))
    print("106 verified training cases only. No conformal/policy/outer cases accessed.")
    print("IID-row SE describes a hypothetical resampling diagnostic. Source pools are shared.")
    print("It is not a measured noise floor, an achievable-R2 estimate, or proof of the cause of weak forecasts.")
    print("No fitting, prediction regeneration, file writes, or Git commands executed.")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    root = args.data_root or os.environ.get("C4_DATA_ROOT")
    d.require(bool(root), "Set C4_DATA_ROOT or --data-root.")
    audit(Path(root))


if __name__ == "__main__":
    main()
