"""Freeze and fit an isolated DEVELOPMENT N1 hybrid; never register it for release.

Reuses the verified external-development labels. No candidate fitting, source-row
inference, calibration, policy fitting or outer evaluation is performed here.
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
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import diagnose_n1_development as d
import experiment_n1_development as e


RECIPE = {
    "global": "ridge_alpha_100_train_only_redundancy_filter_and_scaling",
    "slice": "lgb_original_all",
    "complement": "lgb_original_all",
    "quantiles": [0.05, 0.5, 0.95],
    "weighting": "equal_total_weight_per_scenario_group_normalized_to_mean_one",
    "postprocessing": "sort_three_predictions_then_clip_to_minus_one_plus_one",
    "ridge_intervals": "training_residual_quantiles_UNCALIBRATED",
}


def validate_x(x):
    d.require(isinstance(x, pd.DataFrame) and len(x) > 0 and len(x.columns) > 0,
              "Features must be a nonempty dataframe.")
    d.require(x.index.is_unique and x.columns.is_unique
              and all(isinstance(c, str) for c in x.columns), "Invalid feature identifiers.")
    d.require(all(pd.api.types.is_numeric_dtype(x[c]) for c in x), "Non-numeric features.")
    d.require(np.isfinite(x.to_numpy(float)).all(), "Nonfinite features.")


class HybridForecaster:
    """Nine raw/ordered outputs; raw bands carry no calibrated coverage claim."""

    def fit(self, x, targets, groups):
        validate_x(x)
        d.require(targets.index.equals(x.index) and list(targets) == list(d.TARGETS),
                  "Target order/identity differs.")
        y = targets.to_numpy(float)
        d.require(np.isfinite(y).all() and (np.abs(y) <= 1).all(), "Invalid target margins.")
        d.require(isinstance(groups, pd.Series) and groups.index.equals(x.index)
                  and not groups.isna().any() and groups.nunique() >= 2, "Invalid group alignment/support.")
        self.features = list(x)
        w = d.group_weights(groups)
        w /= w.mean()
        columns = [c for c in x if x[c].nunique() > 1]
        corr = x[columns].corr().abs().to_numpy()
        keep = []
        for j in range(len(columns)):
            if not any(corr[j, k] > 1 - 1e-10 for k in keep):
                keep.append(j)
        columns = [columns[k] for k in keep]
        d.require(bool(columns), "No varying global predictors.")
        scaler = StandardScaler().fit(x[columns], sample_weight=w)
        a = scaler.transform(x[columns])
        model = Ridge(alpha=100.0).fit(a, targets[d.TARGETS[0]], sample_weight=w)
        residual = targets[d.TARGETS[0]].to_numpy() - model.predict(a)
        self.ridge = dict(columns=columns, mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                          coefficients=model.coef_.tolist(), intercept=float(model.intercept_),
                          offsets=[d.weighted_quantile(residual, w, q) for q in (0.05, 0.95)])
        self.tree_columns = [c for c in x if x[c].nunique() > 1]
        self.constants, self.trees = {}, {}
        config = next(c for c in e.candidates() if c["id"] == "lgb_original_all")
        from lightgbm import LGBMRegressor
        for target in d.TARGETS[1:]:
            self.constants[target] = [d.weighted_quantile(targets[target], w, q) for q, _ in d.QUANTILES]
            self.trees[target] = []
            for q, _ in d.QUANTILES:
                model = LGBMRegressor(objective="quantile", alpha=q, learning_rate=0.05,
                    random_state=42, n_jobs=1, deterministic=True, force_col_wise=True,
                    verbosity=-1, **config["parameters"])
                model.fit(x[self.tree_columns].to_numpy(float), targets[target], sample_weight=w)
                self.trees[target].append(model.booster_)
        return self

    def predict(self, x, *, raw=False):
        validate_x(x)
        d.require(list(x) == self.features, "Prediction feature names/order differ.")
        r = self.ridge
        a = (x[r["columns"]].to_numpy(float) - np.array(r["mean"])) / np.array(r["scale"])
        center = a @ np.array(r["coefficients"]) + r["intercept"]
        values = {d.TARGETS[0]: np.column_stack((center+r["offsets"][0], center, center+r["offsets"][1]))}
        for target in d.TARGETS[1:]:
            values[target] = np.column_stack([m.predict(x[self.tree_columns].to_numpy(float), num_threads=1)
                                              for m in self.trees[target]])
        result = pd.DataFrame(index=x.index)
        for target, predictions in values.items():
            d.require(np.isfinite(predictions).all(), "Nonfinite model output.")
            predictions = predictions if raw else e.ordered(predictions)
            for j, (_, label) in enumerate(d.QUANTILES):
                result[target + "__" + label] = predictions[:, j]
        return result

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        files = {}
        for target, models in self.trees.items():
            for j, model in enumerate(models):
                name = f"{target}_{j}.txt"
                content = model.model_to_string().encode("utf-8")
                (directory / name).write_bytes(content)
                files[name] = d.sha(content)
        content = e.encoded(dict(schema="c4-n1-hybrid-development-v1", recipe=RECIPE,
            features=self.features, ridge=self.ridge, tree_columns=self.tree_columns,
            files=files, production_release_authorized=False, conformal_calibrated=False))
        (directory / "model.json").write_bytes(content)
        (directory / "model.sha256").write_text(d.sha(content)+"\n", encoding="utf-8")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        state, _ = d.checked_json(directory, "model")
        d.require(state["schema"] == "c4-n1-hybrid-development-v1" and state["recipe"] == RECIPE
                  and state["production_release_authorized"] is False
                  and state["conformal_calibrated"] is False, "Unsupported model state.")
        from lightgbm import Booster
        obj = cls()
        obj.features, obj.ridge, obj.tree_columns = state["features"], state["ridge"], state["tree_columns"]
        obj.trees = {}
        for target in d.TARGETS[1:]:
            obj.trees[target] = []
            for j in range(3):
                name = f"{target}_{j}.txt"
                content = d.checked(directory / name, state["files"][name])
                obj.trees[target].append(Booster(model_str=content.decode("utf-8")))
        return obj


def prepare(root, c4):
    x, _, index, _, _, pins = d.load_inputs(root, c4)
    recovery = root / "c4_artifacts/forecasting_experiments/adult/adult-external-development-v3-01/portable-resume-v1"
    meta, digest = d.checked_json(recovery)
    d.require(meta["calibration_policy_outer_cases_accessed"] is False
              and meta["production_release_authorized"] is False, "Source scope differs.")
    pins[str(recovery / "metadata.json")] = digest
    def csv(name):
        content = d.checked(recovery / name, meta["files"][name])
        pins[str(recovery / name)] = d.sha(content)
        return pd.read_csv(io.BytesIO(content), float_precision="round_trip").set_index("scenario_id")
    targets = csv("targets.csv")
    tree_oof, challenger_oof = csv("lightgbm_grouped_oof.csv"), csv("targetwise_challenger_grouped_oof.csv")
    d.validate_tables(x, targets, index)
    d.require(all(p.index.equals(x.index) for p in (tree_oof, challenger_oof)), "OOF row alignment differs.")
    selected = tree_oof.copy()
    for _, label in d.QUANTILES:
        name = d.TARGETS[0] + "__" + label
        selected[name] = challenger_oof[name]
    for target in d.TARGETS:
        p = selected[[target+"__"+label for _, label in d.QUANTILES]].to_numpy(float)
        d.require(np.isfinite(p).all() and (np.abs(p) <= 1).all()
                  and (np.diff(p, axis=1) >= 0).all(), "Invalid saved OOF bands.")
    return x, targets, index, selected, pins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--version", default="adult-hybrid-development-v1")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--freeze-spec", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    value = args.data_root or os.environ.get("C4_DATA_ROOT")
    d.require(bool(value), "Set C4_DATA_ROOT.")
    d.require(bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.version)), "Invalid version.")
    root, c4 = Path(value), Path(__file__).resolve().parents[1]
    x, targets, index, selected, pins = prepare(root, c4)
    for source in (Path(__file__), Path(d.__file__), Path(e.__file__)):
        pins[str(source.resolve())] = d.sha(source.read_bytes())
    specification = e.encoded(dict(version=args.version, recipe=RECIPE, inputs=pins,
        environment=e.environment(), rows=len(x), groups=int(index.group_id.nunique()),
        purpose="development_fit_after_model_selection_not_independent_validation",
        pending=["fresh_scenario_group_evaluation", "new_N2_calibration", "new_N4_policy_calibration"],
        production_release_authorized=False))
    directory = root / "c4_artifacts/forecasting_experiments/adult" / args.version
    if args.freeze_spec:
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "specification.json").write_bytes(specification)
        (directory / "specification.sha256").write_text(d.sha(specification)+"\n", encoding="utf-8")
        print("FROZEN:", directory)
        print("Next: repeat with --execute. No model fitted.")
        return
    if not args.execute:
        print("PLAN ONLY: verified", len(x), "development cases; use --freeze-spec then --execute.")
        return
    frozen, _ = d.checked_json(directory, "specification")
    d.require(e.encoded(frozen) == specification, "Frozen inputs/code/environment differ; do not overwrite.")
    output = directory / "fit-v1"
    d.require(not output.exists(), "Fit output already exists; preserve it.")
    model = HybridForecaster().fit(x, targets, index.set_index("scenario_id").group_id)
    before = model.predict(x)
    model.save(output)
    reloaded = HybridForecaster.load(output).predict(x)
    np.testing.assert_allclose(before.to_numpy(), reloaded.to_numpy(), rtol=0, atol=1e-12)
    selected.to_csv(output / "selected_development_oof.csv", index_label="scenario_id")
    rows = []
    for target in d.TARGETS:
        p = selected[[target+"__"+label for _, label in d.QUANTILES]].to_numpy()
        rows.append(dict(target=target, **e.losses(targets[target], p, index.group_id)))
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "selected_development_metrics.csv", index=False)
    for path, digest in pins.items():
        d.checked(Path(path), digest)
    files = {p.name: d.sha(p.read_bytes()) for p in output.iterdir() if p.is_file()}
    metadata = e.encoded(dict(specification_sha256=d.sha(specification), files=files,
        rows=len(x), groups=int(index.group_id.nunique()), training_predictions_used_as_validation=False,
        oof_note="Existing OOF components selected after inspecting development results; selection evidence only.",
        model_roundtrip_max_absolute_error=float(np.max(np.abs(before.to_numpy()-reloaded.to_numpy()))),
        calibration_policy_outer_cases_accessed=False, conformal_calibrated=False,
        production_release_authorized=False))
    (output / "metadata.json").write_bytes(metadata)
    (output / "metadata.sha256").write_text(d.sha(metadata)+"\n", encoding="utf-8")
    display = summary.copy()
    display["RMSE_pp"], display["MAE_pp"] = display.rmse*100, display.mae*100
    print(display[["target", "MAE_pp", "RMSE_pp", "r2"]].to_string(index=False))
    print("MODEL SAVE/RELOAD: PASS. DEVELOPMENT ONLY; no fresh validation or calibration performed.")
    print("OUTPUT:", output)
    print("METADATA SHA256:", d.sha(metadata))


if __name__ == "__main__":
    main()
