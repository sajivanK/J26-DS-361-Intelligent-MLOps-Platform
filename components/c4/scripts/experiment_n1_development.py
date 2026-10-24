"""Isolated N1 revision study: nested grouped CV on forecast_train ONLY.

Plan, freeze the development protocol, then execute. No original model is changed,
no final model is fitted, and no calibration/policy/outer file is opened.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import diagnose_n1_development as d


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def environment():
    return {"python": platform.python_version(), **{
        name: importlib.metadata.version(name)
        for name in ("numpy", "pandas", "scikit-learn", "lightgbm", "scipy")}}


def compact_columns(columns):
    evidence = {"c1_predicted_harm", "c1_harm_width", "c2_slice_support",
                "c2_diagnosis_stability", "c3_fidelity_score", "c3_utility_score",
                "c3_target_coverage", "c3_augmentation_ratio"}
    return [c for c in columns if c in evidence or
            (c.startswith("c4_") and c.endswith(("candidate_minus_champion_accuracy",
                                                "champion_accuracy", "rows")))]


def candidates():
    """Fixed hypothesis grid, not a search over held-out PP1 outcomes."""
    result = [{"id": "constant", "kind": "constant", "features": "all", "parameters": {}}]
    trees = [
        ("original", dict(n_estimators=100, num_leaves=7, max_depth=3,
                          min_child_samples=10, reg_lambda=1.0)),
        ("conservative", dict(n_estimators=150, num_leaves=3, max_depth=2,
                              min_child_samples=15, reg_lambda=3.0)),
        ("smooth", dict(n_estimators=250, num_leaves=3, max_depth=2,
                        min_child_samples=20, reg_lambda=10.0)),
        ("flexible", dict(n_estimators=200, num_leaves=7, max_depth=3,
                          min_child_samples=5, reg_lambda=3.0)),
    ]
    for features in ("all", "compact"):
        for name, parameters in trees:
            result.append(dict(id=f"lgb_{name}_{features}", kind="lightgbm",
                               features=features, parameters=parameters))
        for alpha in (0.0001, 0.001, 0.005):
            result.append(dict(id=f"quantile_{alpha}_{features}", kind="linear_quantile",
                               features=features, parameters={"alpha": alpha}))
        for alpha in (1.0, 10.0, 100.0):
            result.append(dict(id=f"ridge_{alpha}_{features}", kind="ridge",
                               features=features, parameters={"alpha": alpha}))
    return result


def recipes(grid):
    return [(c["id"], shrink) for c in grid
            for shrink in ((0.0,) if c["kind"] == "constant" else (0.25, 0.5, 1.0))]


def ordered(raw):
    return np.clip(np.sort(np.asarray(raw, dtype=float), axis=1), -1, 1)


def fit_predict(candidate, x_train, y_train, groups, x_valid):
    """All feature filtering, scaling and residual references fit on train only."""
    w = d.group_weights(groups)
    w = w / w.mean()
    baseline = np.array([d.weighted_quantile(y_train, w, q) for q, _ in d.QUANTILES])
    if candidate["kind"] == "constant":
        return np.tile(baseline, (len(x_valid), 1))
    columns = list(x_train) if candidate["features"] == "all" else compact_columns(x_train)
    d.require(bool(columns), "Compact feature set is empty; inspect the planned feature list.")
    columns = [c for c in columns if x_train[c].nunique() > 1]
    if not columns:
        return np.tile(baseline, (len(x_valid), 1))
    a, b = x_train[columns].to_numpy(float), x_valid[columns].to_numpy(float)
    kind = candidate["kind"]
    params = candidate["parameters"]
    if kind in ("linear_quantile", "ridge"):
        scaler = StandardScaler().fit(a, sample_weight=w)
        a, b = scaler.transform(a), scaler.transform(b)
    if kind == "ridge":
        model = Ridge(**params).fit(a, y_train, sample_weight=w)
        center = model.predict(b)
        residuals = np.asarray(y_train) - model.predict(a)
        # Exploratory, UNCALIBRATED raw bands; these do not replace joint N2.
        offsets = [d.weighted_quantile(residuals, w, q) for q in (0.05, 0.95)]
        return np.column_stack((center + offsets[0], center, center + offsets[1]))
    predictions = []
    for q, _ in d.QUANTILES:
        if kind == "lightgbm":
            from lightgbm import LGBMRegressor
            model = LGBMRegressor(objective="quantile", alpha=q, learning_rate=0.05,
                                 random_state=42, n_jobs=1, deterministic=True,
                                 force_col_wise=True, verbosity=-1, **params)
        else:
            d.require(kind == "linear_quantile", "Unknown model family.")
            model = QuantileRegressor(quantile=q, solver="highs", **params)
        model.fit(a, y_train, sample_weight=w)
        predictions.append(model.predict(b))
    return np.column_stack(predictions)


def losses(y, pred, groups):
    w = d.group_weights(groups)
    metrics = d.metrics(y, pred[:, 1], w)
    pinballs = []
    for col, (q, _) in enumerate(d.QUANTILES):
        error = np.asarray(y) - pred[:, col]
        pinballs.append(float(np.average(np.maximum(q * error, (q - 1) * error), weights=w)))
        metrics[f"pinball_{q}"] = pinballs[-1]
    metrics["mean_pinball"] = float(np.mean(pinballs))
    return metrics


def split_records(index, n):
    records = []
    for train, valid in GroupKFold(n).split(index, groups=index.group_id):
        d.require(not set(index.iloc[train].group_id) & set(index.iloc[valid].group_id), "Group leakage.")
        records.append({"train_ids": index.iloc[train].scenario_id.tolist(),
                        "validation_ids": index.iloc[valid].scenario_id.tolist()})
    return records


def screen(x, y, index, grid, splits=3, progress=lambda _: None):
    """Inner grouped OOF screen. Returns scores and predictions for every recipe."""
    raw = {c["id"]: np.full((len(x), 3), np.nan) for c in grid}
    for fold, (train, valid) in enumerate(GroupKFold(splits).split(index, groups=index.group_id), 1):
        for c in grid:
            raw[c["id"]][valid] = fit_predict(c, x.iloc[train], y.iloc[train],
                                             index.iloc[train].group_id, x.iloc[valid])
        progress(f"  inner fold {fold}/{splits}: {len(grid)} base models complete")
    baseline = raw["constant"]
    ref = losses(y, ordered(baseline), index.group_id)
    predictions, rows = {}, []
    for name, shrink in recipes(grid):
        key = f"{name}@{shrink:g}"
        p = ordered(shrink * raw[name] + (1 - shrink) * baseline)
        d.require(np.isfinite(p).all(), "Incomplete OOF predictions.")
        m = losses(y, p, index.group_id)
        # Fixed equal importance to squared-error accuracy and quantile quality.
        score = 0.5 * m["rmse"] / max(ref["rmse"], 1e-15) + \
                0.5 * m["mean_pinball"] / max(ref["mean_pinball"], 1e-15)
        rows.append(dict(recipe=key, candidate=name, shrink=shrink, score=score, **m))
        predictions[key] = p
    table = pd.DataFrame(rows)
    # Complexity must improve BOTH objectives; a weak target can retain its baseline.
    eligible = table[(table.rmse <= ref["rmse"] + 1e-15) &
                     (table.mean_pinball <= ref["mean_pinball"] + 1e-15)]
    selected = eligible.sort_values(["score", "shrink", "recipe"], kind="stable").iloc[0].to_dict()
    return selected, table, predictions


def nested_study(x, targets, index, grid, outer_splits=5, inner_splits=3, progress=print):
    predictions = pd.DataFrame(index=x.index, dtype=float)
    selections, screens, fold_metrics = [], [], []
    for fold, (train, valid) in enumerate(GroupKFold(outer_splits).split(index, groups=index.group_id), 1):
        for target in d.TARGETS:
            progress(f"NESTED FOLD {fold}/{outer_splits}: {target}")
            idx = index.iloc[train].reset_index(drop=True)
            best, table, _ = screen(x.iloc[train], targets.iloc[train][target], idx,
                                     grid, inner_splits, progress)
            chosen = next(c for c in grid if c["id"] == best["candidate"])
            args = (x.iloc[train], targets.iloc[train][target], idx.group_id, x.iloc[valid])
            base = fit_predict(grid[0], *args)
            raw = fit_predict(chosen, *args)
            pred = ordered(best["shrink"] * raw + (1 - best["shrink"]) * base)
            for col, (_, label) in enumerate(d.QUANTILES):
                predictions.loc[x.index[valid], target + "__" + label] = pred[:, col]
            selections.append(dict(fold=fold, target=target, **best))
            table["fold"], table["target"] = fold, target
            screens.append(table)
            fold_metrics.append(dict(fold=fold, target=target,
                **losses(targets.iloc[valid][target], pred, index.iloc[valid].group_id)))
            progress(f"  selected {best['recipe']}; held-out development fold completed")
    d.require(not predictions.isna().any().any(), "Incomplete nested predictions.")
    return predictions, pd.DataFrame(selections), pd.concat(screens), pd.DataFrame(fold_metrics)


def make_protocol(x, index, pinned, grid):
    scripts = [Path(__file__).resolve(), Path(d.__file__).resolve()]
    return {"version": "n1-development-nested-v2", "scope": "forecast_train_only",
            "environment": environment(), "inputs": pinned,
            "scripts": {str(p): d.sha(p.read_bytes()) for p in scripts},
            "targets": list(d.TARGETS), "raw_quantiles": [q for q, _ in d.QUANTILES],
            "outer_development_folds": split_records(index, 5),
            "inner_development_folds": 3, "grid": grid, "recipes": recipes(grid),
            "compact_features": compact_columns(x),
            "selection": "Both RMSE and mean pinball must be <= constant quantiles; then minimize equal-weight normalized RMSE/pinball. Ties prefer more shrinkage.",
            "weights": "Equal total weight per nonempty group; normalized to mean 1 for fitting.",
            "postprocessing": "Blend raw forecasts with training-only constant quantiles; sort and clip to [-1,1].",
            "ridge_bands": "Training residual tails, uncalibrated; all families require separate N2 calibration.",
            "interpretation": "Nested OOF assesses selection procedure. Full-training screen is selection evidence, not fresh final performance.",
            "original_outer_already_seen": True, "fresh_final_evaluation_required": True,
            "original_artifacts_changed": False, "final_models_fitted": False}


def report(summary, choices):
    text = ["# N1 development experiment", "", "TRAINING NESTED GROUPED OOF ONLY. NOT NEW OUTER RESULTS.", "",
            "| Method | Target | MAE (pp) | RMSE (pp) | R2 | Mean pinball (pp) |",
            "|---|---|---:|---:|---:|---:|"]
    for row in summary.to_dict("records"):
        r2 = "undefined" if pd.isna(row["r2"]) else f"{row['r2']:.5f}"
        pb = "n/a" if pd.isna(row.get("mean_pinball")) else f"{100 * row['mean_pinball']:.5f}"
        text.append(f"| {row['method']} | {row['target']} | {100*row['mae']:.5f} | {100*row['rmse']:.5f} | {r2} | {pb} |")
    text += ["", "## Proposed recipes from full-training grouped selection", ""]
    text += [f"- {r['target']}: {r['recipe']}" for r in choices]
    text += ["", "These choices are provisional; their screening scores are not unbiased performance estimates.",
             "Nested folds may choose different recipes. No final model, N2 calibration, policy or UI was changed.",
             "The grid was designed after reviewing development diagnostics and prior PP1 results; this is an exploratory revision study.",
             "Existing PP1 outer data cannot establish an untouched final test for this revision.",
             "No improvement is guaranteed. A constant fallback is allowed. Raw intervals have no calibrated coverage claim.",
             "Do not select a revision on R2 alone: compare RMSE, MAE, pinball and fold consistency; then use fresh evaluation."]
    return "\n".join(text) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--run-id", default="n1-development-v2-01")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--freeze-spec", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    value = args.data_root or os.environ.get("C4_DATA_ROOT")
    d.require(bool(value), "Set C4_DATA_ROOT or --data-root.")
    d.require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", args.run_id) is not None, "Unsafe run ID.")
    root = Path(value)
    c4_root = Path(__file__).resolve().parents[1]
    print("VERIFYING FORECAST_TRAIN INPUTS ONLY", flush=True)
    x, targets, index, originals, folds, pinned = d.load_inputs(root, c4_root)
    grid = candidates()
    protocol = make_protocol(x, index, pinned, grid)
    # Loader verifies original fold binding. Also compare actual membership here.
    expected = [set(f["validation_ids"]) for f in protocol["outer_development_folds"]]
    supplied = [set(f["validation_ids"]) for f in folds]
    d.require(expected == supplied, "Original development fold membership differs.")
    contents = encoded(protocol)
    output = root / "c4_artifacts/forecasting_experiments/adult" / args.run_id
    print(f"106 scenarios / 29 groups; {len(grid)} base candidates / {len(recipes(grid))} recipes per target", flush=True)
    print("5 outer development folds x 3 inner folds; no TVAE, no calibration/policy/outer reads, no final model replacement", flush=True)
    print("COMPACT FEATURES:", ", ".join(protocol["compact_features"]), flush=True)
    d.require(bool(protocol["compact_features"]), "Empty compact feature set.")
    if args.freeze_spec:
        d.require(not output.exists(), "Run already exists; use a new run ID, do not overwrite.")
        output.mkdir(parents=True)
        (output / "specification.json").write_bytes(contents)
        (output / "specification.sha256").write_text(d.sha(contents) + "\n", encoding="utf-8")
        print("SPECIFICATION FROZEN:", output, flush=True)
        return
    if not args.execute:
        print("PLAN ONLY. Next --freeze-spec, then --execute with the same run ID.", flush=True)
        return
    d.checked(output / "specification.json", (output / "specification.sha256").read_text().strip())
    d.require((output / "specification.json").read_bytes() == contents, "Frozen experiment inputs/code/environment changed.")
    results = output / "results"
    d.require(not results.exists(), "Results/partial run already exist; preserve them and use a new run ID.")
    results.mkdir()
    log = (results / "progress.log").open("x", encoding="utf-8")
    def progress(message):
        line = datetime.now(timezone.utc).isoformat() + " " + message
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()
    try:
        prediction, selections, inner, fold_metrics = nested_study(x, targets, index, grid, progress=progress)
        prediction.to_csv(results / "nested_grouped_oof.csv", index_label="scenario_id")
        selections.to_csv(results / "nested_selections.csv", index=False)
        inner.to_csv(results / "inner_screen_scores.csv", index=False)
        fold_metrics.to_csv(results / "nested_fold_metrics.csv", index=False)
        q, mean, _ = d.constant_references(targets, index)
        forecasts = {**originals, "constant_quantiles": q, "constant_mean": mean, "nested_selected": prediction}
        rows, by_fold, by_group = [], [], []
        for name, p in forecasts.items():
            for target in d.TARGETS:
                if name == "constant_mean":
                    m = d.metrics(targets[target], p[target + "__median"], d.group_weights(index.group_id))
                else:
                    m = losses(targets[target], p[[target + "__" + label for _, label in d.QUANTILES]].to_numpy(), index.group_id)
                rows.append(dict(method=name, target=target, **m))
                for number, membership in enumerate(protocol["outer_development_folds"], 1):
                    mask = index.scenario_id.isin(membership["validation_ids"]).to_numpy()
                    by_fold.append(dict(method=name, target=target, fold=number,
                        **d.metrics(targets[target].to_numpy()[mask],
                                    p[target + "__median"].to_numpy()[mask],
                                    d.group_weights(index.loc[mask, "group_id"]))))
                for group in sorted(index.group_id.unique()):
                    mask = index.group_id.eq(group).to_numpy()
                    by_group.append(dict(method=name, target=target, group_id=group,
                        **d.metrics(targets[target].to_numpy()[mask],
                                    p[target + "__median"].to_numpy()[mask],
                                    np.ones(int(mask.sum())))))
        summary = pd.DataFrame(rows)
        summary.to_csv(results / "metric_summary.csv", index=False)
        pd.DataFrame(by_fold).to_csv(results / "all_method_fold_metrics.csv", index=False)
        pd.DataFrame(by_group).to_csv(results / "all_method_group_metrics.csv", index=False)
        choices, screens = [], []
        for target in d.TARGETS:
            progress(f"FULL-TRAINING RECIPE SELECTION: {target} (selection scores, not final evaluation)")
            best, table, _ = screen(x, targets[target], index, grid, 5, progress)
            choices.append(dict(target=target, **best))
            table["target"] = target
            screens.append(table)
        pd.concat(screens).to_csv(results / "full_training_screen.csv", index=False)
        (results / "provisional_recipes.json").write_bytes(encoded(choices))
        document = report(summary, choices)
        (results / "N1_DEVELOPMENT_EXPERIMENT.md").write_text(document, encoding="utf-8")
        for path, digest in {**pinned, **protocol["scripts"]}.items():
            d.checked(Path(path), digest)
        progress("COMPLETE. Original artifacts unchanged. NO GIT COMMANDS. No final model fitted.")
        log.close()
        hashes = {p.name: d.sha(p.read_bytes()) for p in sorted(results.iterdir()) if p.is_file()}
        metadata = encoded(dict(specification_sha256=d.sha(contents), files=hashes,
                               scope="forecast_train_only", final_models_fitted=False,
                               production_release_authorized=False))
        (results / "metadata.json").write_bytes(metadata)
        (results / "metadata.sha256").write_text(d.sha(metadata) + "\n", encoding="utf-8")
        print(document, flush=True)
        print("OUTPUT:", results, flush=True)
        print("METADATA SHA256:", d.sha(metadata), flush=True)
    finally:
        log.close()


if __name__ == "__main__":
    main()
