"""Diagnose frozen N1 training OOF predictions; no outer/calibration reads or commits.

Constant references are fitted separately inside each original training fold.
This script never loads a joblib model, refits N1, or changes frozen artifacts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

TARGETS = ("future_global_margin", "future_slice_margin", "future_nonaffected_margin")
QUANTILES = ((0.05, "lower"), (0.5, "median"), (0.95, "upper"))
PROTOCOL = "adult-simulation-pp1-v2"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def checked(path, expected):
    data = Path(path).read_bytes()
    require(sha(data) == expected, f"Checksum mismatch: {path}")
    return data


def checked_json(directory, stem="metadata"):
    expected = (directory / (stem + ".sha256")).read_text(encoding="utf-8").strip()
    return json.loads(checked(directory / (stem + ".json"), expected)), expected


def group_weights(groups):
    groups = pd.Series(list(groups), dtype="object")
    require(len(groups) > 0 and not groups.isna().any(), "Missing/empty groups.")
    counts = groups.value_counts()
    return np.array([1.0 / counts[g] for g in groups], dtype=float)


def weighted_quantile(values, weights, quantile):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    require(values.ndim == weights.ndim == 1 and len(values) == len(weights) > 0,
            "Invalid weighted quantile arrays.")
    require(np.isfinite(values).all() and np.isfinite(weights).all() and (weights > 0).all(),
            "Invalid weighted quantile values/weights.")
    require(0 <= quantile <= 1, "Invalid quantile.")
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    location = min(np.searchsorted(cumulative, quantile * cumulative[-1], side="left"), len(values)-1)
    return float(values[order[location]])


def constant_references(targets, index, splits=5):
    """Only each fold's TRAIN labels determine its constant predictions."""
    require(targets.index.tolist() == index.scenario_id.tolist(), "Reference row alignment differs.")
    require(index.split_role.eq("forecast_train").all(), "References require forecast_train only.")
    quantile = pd.DataFrame(index=targets.index, dtype=float)
    mean = pd.DataFrame(index=targets.index, dtype=float)
    records = []
    for number, (train, valid) in enumerate(GroupKFold(splits).split(index, groups=index.group_id), 1):
        require(not (set(index.iloc[train].group_id) & set(index.iloc[valid].group_id)), "Group leakage.")
        weights = group_weights(index.iloc[train].group_id)
        row = {"fold": number, "training_rows": len(train), "validation_rows": len(valid),
               "training_group_ids": sorted(index.iloc[train].group_id.unique().tolist()),
               "validation_group_ids": sorted(index.iloc[valid].group_id.unique().tolist()),
               "train_ids": index.iloc[train].scenario_id.tolist(),
               "validation_ids": index.iloc[valid].scenario_id.tolist()}
        records.append(row)
        for target in TARGETS:
            values = targets.iloc[train][target].to_numpy(float)
            mean.loc[targets.index[valid], target + "__median"] = float(np.average(values, weights=weights))
            for q, label in QUANTILES:
                quantile.loc[targets.index[valid], target + "__" + label] = weighted_quantile(values, weights, q)
    require(not quantile.isna().any().any() and not mean.isna().any().any(), "Incomplete references.")
    return quantile, mean, records


def metrics(y, point, weights):
    y, point, weights = map(lambda x: np.asarray(x, float), (y, point, weights))
    require(len(y) > 0 and y.shape == point.shape == weights.shape, "Metric alignment differs.")
    require(all(np.isfinite(a).all() for a in (y, point, weights)), "Nonfinite metrics.")
    require((weights > 0).all(), "Nonpositive metric weights.")
    error = point - y
    mse = float(np.average(error**2, weights=weights))
    variance = float(np.average((y - np.average(y, weights=weights))**2, weights=weights))
    return {"mae": float(np.average(abs(error), weights=weights)), "rmse": float(np.sqrt(mse)),
            "r2": None if variance <= 0 else float(1 - mse / variance),
            "bias_prediction_minus_actual": float(np.average(error, weights=weights)),
            "target_mean": float(np.average(y, weights=weights)), "target_std": float(np.sqrt(variance)),
            "prediction_std": float(np.sqrt(np.average((point - np.average(point, weights=weights))**2, weights=weights)))}


def validate_tables(features, targets, index):
    require(features.shape == (106, 75), "Expected original 106 x 75 training features.")
    require(index.group_id.nunique() == 29 and len(index) == 106, "Original group inventory differs.")
    require(index.split_role.eq("forecast_train").all(), "Non-training role rejected.")
    require(features.index.is_unique and features.columns.is_unique and targets.index.is_unique,
            "Duplicate features/scenario identities.")
    require(index.scenario_id.is_unique and features.index.tolist() == index.scenario_id.tolist()
            and features.index.equals(targets.index), "Scenario IDs/order differ.")
    require(targets.columns.tolist() == list(TARGETS), "Target definition differs.")
    require(not (set(TARGETS) & set(features.columns)), "Target leakage in features.")
    for frame in (features, targets):
        require(all(pd.api.types.is_numeric_dtype(frame[c]) for c in frame), "Nonnumeric table.")
        require(np.isfinite(frame.to_numpy(float)).all(), "Nonfinite input.")
    require((np.abs(targets.to_numpy(float)) <= 1).all(), "Margins outside physical range.")
    require(index.group_id.map(lambda x: isinstance(x, str) and bool(x)).all(), "Invalid groups.")


def load_inputs(root, c4_root):
    """Read ONLY exact allowlisted forecast_train and original N1 OOF files."""
    config_path = c4_root / "configs/n1_benchmark_config.json"
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    require(config["specification_version"] == "adult-n1-training-v1", "Unexpected N1 version.")
    require(config["training_snapshot_version"] == "forecast-training-all32-audit-v1", "Unexpected training snapshot.")
    require(config["raw_quantiles"] == [0.05, 0.5, 0.95] and config["parameters"]["cv_splits"] == 5,
            "Original quantiles/folds differ.")
    training = root / "c4_artifacts/meta_datasets/adult" / PROTOCOL / config["training_snapshot_version"]
    meta, meta_hash = checked_json(training)
    require(meta_hash == config["training_snapshot_metadata_sha256"], "Training snapshot is not pinned by N1 config.")
    require(meta["fold"] == "forecast_train" and meta["rows"] == 106 and len(meta["groups"]) == 32,
            "Wrong training snapshot scope.")
    fit = root / "c4_artifacts/forecasting/adult" / PROTOCOL / config["specification_version"]
    specification, specification_hash = checked_json(fit, "specification")
    require(specification["binding"]["configuration_sha256"] == sha(config_bytes), "Original config binding differs.")
    require(specification["binding"]["training_metadata_sha256"] == meta_hash, "Training binding differs.")
    model_meta, model_hash = checked_json(fit / "n1-fit-v1")
    require(model_meta["specification_sha256"] == specification_hash
            and model_meta["training_metadata_sha256"] == meta_hash, "N1 OOF dependency differs.")
    pinned = {str(config_path): sha(config_bytes), str(training / "metadata.json"): meta_hash,
              str(fit / "specification.json"): specification_hash, str(fit / "n1-fit-v1/metadata.json"): model_hash}
    def csv(directory, name, hashes):
        data = checked(directory / name, hashes[name])
        pinned[str(directory / name)] = sha(data)
        return pd.read_csv(io.BytesIO(data), float_precision="round_trip")
    features = csv(training, "features.csv", meta["files"]).set_index("scenario_id")
    targets = csv(training, "targets.csv", meta["files"]).set_index("scenario_id")
    index = csv(training, "scenario_index.csv", meta["files"])
    validate_tables(features, targets, index)
    require(features.columns.tolist() == meta["feature_columns"], "Feature metadata differs.")
    require(set(index.group_id) <= set(meta["groups"]), "Group assignment differs.")
    forecasts = {}
    for method in ("lightgbm", "linear"):
        frame = csv(fit / "n1-fit-v1", method + "_grouped_oof.csv", model_meta["files"]).set_index("scenario_id")
        require(frame.index.equals(features.index), "Saved OOF order/identity differs.")
        for target in TARGETS:
            cols = [target + "__" + prefix + label for prefix in ("", "raw_") for _, label in QUANTILES]
            require(set(cols) <= set(frame.columns), "Missing OOF quantile columns.")
            require(np.isfinite(frame[cols].to_numpy(float)).all(), "Nonfinite OOF prediction.")
        forecasts[method] = frame
    return features, targets, index, forecasts, specification["binding"]["internal_folds"], pinned


def diagnose(features, targets, index, forecasts):
    quantiles, mean, folds = constant_references(targets, index)
    forecasts = {**forecasts, "constant_quantiles": quantiles, "constant_mean": mean}
    summaries, fold_rows, group_rows, residual_rows = [], [], [], []
    for method, forecast in forecasts.items():
        for target in TARGETS:
            y, point = targets[target].to_numpy(float), forecast[target + "__median"].to_numpy(float)
            for weighting, weights in (("equal_group_weighted", group_weights(index.group_id)),
                                        ("scenario_weighted", np.ones(len(y)))):
                row = {"method": method, "target": target, "weighting": weighting, **metrics(y, point, weights)}
                if method != "constant_mean":
                    for q, label in QUANTILES:
                        error = y - forecast[target + "__" + label].to_numpy(float)
                        row[f"pinball_{q}"] = float(np.average(np.maximum(q*error, (q-1)*error), weights=weights))
                if method in ("lightgbm", "linear"):
                    raw = forecast[[target + "__raw_" + label for _, label in QUANTILES]].to_numpy(float)
                    row["crossing_rows"] = int((np.diff(raw, axis=1) < 0).any(axis=1).sum())
                    row["raw_median_mae"] = metrics(y, raw[:, 1], weights)["mae"]
                    row["postprocessing_changed_median_rows"] = int((np.abs(raw[:, 1]-point) > 1e-12).sum())
                summaries.append(row)
            for fold in folds:
                mask = index.group_id.isin(fold["validation_group_ids"]).to_numpy()
                fold_rows.append({"method": method, "target": target, "fold": fold["fold"],
                    "rows": int(mask.sum()), **metrics(y[mask], point[mask], group_weights(index.loc[mask, "group_id"]))})
            for group in sorted(index.group_id.unique()):
                mask = index.group_id.eq(group).to_numpy()
                group_rows.append({"method": method, "target": target, "group_id": group,
                    "rows": int(mask.sum()), **metrics(y[mask], point[mask], np.ones(mask.sum()))})
            for i, scenario in enumerate(index.scenario_id):
                residual_rows.append({"scenario_id": scenario, "group_id": index.group_id.iloc[i],
                    "method": method, "target": target, "actual": y[i], "predicted": point[i],
                    "residual_prediction_minus_actual": point[i]-y[i]})
    feature_rows = []
    for column in features:
        values = features[column].to_numpy(float)
        within = features.groupby(index.group_id.to_numpy())[column].nunique(dropna=False)
        feature_rows.append({"feature": column, "unique_values": int(features[column].nunique()),
            "std": float(np.std(values)), "min": float(values.min()), "max": float(values.max()),
            "groups_with_within_group_variation": int((within > 1).sum())})
    return {"metric_summary": pd.DataFrame(summaries), "fold_metrics": pd.DataFrame(fold_rows),
            "group_metrics": pd.DataFrame(group_rows), "residuals": pd.DataFrame(residual_rows),
            "feature_inventory": pd.DataFrame(feature_rows)}, folds


def report_text(tables):
    rows = tables["metric_summary"].query("weighting == 'equal_group_weighted'")
    lines = ["# N1 development diagnosis", "", "TRAINING GROUPED OOF ONLY — NOT NEW OUTER RESULTS.", "",
        "106 training scenarios / 29 nonempty groups. Existing LightGBM/linear OOF predictions are reused.",
        "Constant references use only each fold's training labels, with equal total weight per training group.",
        "The evaluation mean used by R2 is a diagnostic reference, not a deployable trained baseline.", "",
        "| Method | Target | MAE (pp) | RMSE (pp) | R2 | Bias (pp) | Target SD (pp) | Prediction SD (pp) |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows.to_dict("records"):
        r2 = "undefined" if pd.isna(row["r2"]) else f'{row["r2"]:.5f}'
        lines.append(f'| {row["method"]} | {row["target"]} | {100*row["mae"]:.5f} | {100*row["rmse"]:.5f} | {r2} | {100*row["bias_prediction_minus_actual"]:.5f} | {100*row["target_std"]:.5f} | {100*row["prediction_std"]:.5f} |')
    lines += ["", "## Interpretation boundaries", "",
        "- Negative R2 indicates worse squared error than the evaluation mean reference; it does not identify the cause.",
        "- Low prediction variation relative to target variation is a diagnostic signal, not proof of underfitting.",
        "- Compare pinball loss only at the same target, quantile and weighting. Constant_mean is a point-only baseline.",
        "- No parameters or feature sets are selected automatically by this diagnostic.",
        "- Original PP1 results and artifacts remain unchanged. Any revised final claim requires fresh evaluation.",
        "- No conformal, policy or outer tables or fitted model binaries are opened by this script.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--run-id", default="development-diagnosis-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    args = parser.parse_args()
    value = args.data_root or os.environ.get("C4_DATA_ROOT")
    require(bool(value), "Set C4_DATA_ROOT or pass --data-root.")
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}", args.run_id), "Invalid run ID.")
    root = Path(value).expanduser().resolve()
    c4_root = Path(__file__).resolve().parents[1]
    output = root / "c4_artifacts/forecasting_diagnostics/adult" / args.run_id
    require(not output.exists(), "Diagnostic output exists; preserve it.")
    print("VERIFYING ORIGINAL FORECAST_TRAIN SNAPSHOT AND SAVED OOF CHECKSUMS", flush=True)
    features, targets, index, forecasts, original_folds, pinned = load_inputs(root, c4_root)
    print("VERIFIED: 106 scenarios / 29 nonempty groups; no calibration/policy/outer reads", flush=True)
    tables, folds = diagnose(features, targets, index, forecasts)
    for original, reconstructed in zip(original_folds, folds):
        require(original["fold"] == reconstructed["fold"] and original["train_ids"] == reconstructed["train_ids"]
                and original["validation_ids"] == reconstructed["validation_ids"], "Original CV fold reconstruction differs.")
    require(len(original_folds) == len(folds) == 5, "Original fold count differs.")
    for path, expected in pinned.items():
        checked(Path(path), expected)
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    def write(name, contents):
        with (output / name).open("xb") as stream:
            stream.write(contents)
        files[name] = sha(contents)
    for name, table in tables.items():
        write(name + ".csv", table.to_csv(index=False, lineterminator="\n", na_rep="").encode("utf-8"))
    write("folds.json", (json.dumps(folds, indent=2, allow_nan=False) + "\n").encode())
    report = report_text(tables)
    write("N1_DEVELOPMENT_DIAGNOSIS.md", report.encode("utf-8"))
    import sklearn
    manifest = {"purpose": "training_only_n1_development_diagnosis", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": files, "inputs": pinned, "code_sha256": sha(Path(__file__).read_bytes()),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "constant_baselines_fitted": True, "n1_refitted": False, "calibration_policy_outer_accessed": False,
        "frozen_artifacts_changed": False, "model_selected": False, "production_release_authorized": False}
    contents = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with (output / "metadata.json").open("xb") as stream:
        stream.write(contents)
    with (output / "metadata.sha256").open("x", encoding="utf-8") as stream:
        stream.write(sha(contents) + "\n")
    print(report, flush=True)
    print("OUTPUT:", output)
    print("METADATA SHA256:", sha(contents))
    print("DIAGNOSIS COMPLETE. NO GIT COMMANDS EXECUTED. NO N1 REFIT OR OUTER EVALUATION.")


if __name__ == "__main__":
    main()
