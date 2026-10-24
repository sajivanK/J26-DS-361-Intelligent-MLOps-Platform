"""Fit fixed same-cohort N1 baselines before fresh validation; no new outcomes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler

import fit_n1_hybrid_development as h
import freeze_n1_hybrid_validation as v

d, e = h.d, h.e
METHODS = {
    "same_external_training_lightgbm": "lightgbm",
    "same_external_training_linear": "linear",
    "same_external_training_constant_quantiles": "constant",
}


class FixedBaseline:
    """Fixed development recipe, saved as numeric JSON/native LightGBM text."""

    def __init__(self, kind):
        d.require(kind in METHODS.values(), "Unknown baseline family.")
        self.kind = kind

    def fit(self, x, targets, groups):
        h.validate_x(x)
        d.require(targets.index.equals(x.index) and list(targets) == list(d.TARGETS),
                  "Target order/identity differs.")
        y = targets.to_numpy(float)
        d.require(np.isfinite(y).all() and (np.abs(y) <= 1).all(), "Invalid targets.")
        d.require(isinstance(groups, pd.Series) and groups.index.equals(x.index)
                  and not groups.isna().any() and groups.nunique() >= 2, "Invalid group alignment/support.")
        self.features = list(x)
        self.columns = [c for c in x if x[c].nunique() > 1]
        weights = d.group_weights(groups)
        weights /= weights.mean()
        self.constants = {t: [d.weighted_quantile(targets[t], weights, q) for q, _ in d.QUANTILES]
                          for t in d.TARGETS}
        self.models, self.scaling = {}, None
        if self.kind == "constant" or not self.columns:
            return self
        a = x[self.columns].to_numpy(float)
        if self.kind == "linear":
            scaler = StandardScaler().fit(a, sample_weight=weights)
            self.scaling = dict(mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist())
            a = scaler.transform(a)
        config = next(c for c in e.candidates() if c["id"] == "lgb_original_all")
        for target in d.TARGETS:
            self.models[target] = []
            for q, _ in d.QUANTILES:
                if self.kind == "lightgbm":
                    from lightgbm import LGBMRegressor
                    m = LGBMRegressor(objective="quantile", alpha=q, learning_rate=.05,
                        random_state=42, n_jobs=1, deterministic=True, force_col_wise=True,
                        verbosity=-1, **config["parameters"])
                    m.fit(a, targets[target], sample_weight=weights)
                    self.models[target].append(m.booster_)
                else:
                    m = QuantileRegressor(quantile=q, alpha=.001, solver="highs")
                    m.fit(a, targets[target], sample_weight=weights)
                    self.models[target].append(dict(coefficients=m.coef_.tolist(), intercept=float(m.intercept_)))
        return self

    def predict(self, x, *, raw=False):
        h.validate_x(x)
        d.require(list(x) == self.features, "Prediction feature names/order differ.")
        a = x[self.columns].to_numpy(float)
        if self.scaling is not None:
            a = (a - np.array(self.scaling["mean"])) / np.array(self.scaling["scale"])
        output = pd.DataFrame(index=x.index)
        for target in d.TARGETS:
            if target not in self.models:
                p = np.tile(self.constants[target], (len(x), 1))
            elif self.kind == "lightgbm":
                p = np.column_stack([m.predict(a, num_threads=1) for m in self.models[target]])
            else:
                p = np.column_stack([a @ np.array(m["coefficients"]) + m["intercept"]
                                     for m in self.models[target]])
            d.require(np.isfinite(p).all(), "Nonfinite predictions.")
            p = p if raw else e.ordered(p)
            for j, (_, label) in enumerate(d.QUANTILES):
                output[target+"__"+label] = p[:, j]
        return output

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        tree_files = {}
        if self.kind == "lightgbm":
            for target, models in self.models.items():
                for j, model in enumerate(models):
                    name = f"{target}_{j}.txt"
                    content = model.model_to_string().encode("utf-8")
                    (directory / name).write_bytes(content)
                    tree_files[name] = d.sha(content)
        state = dict(schema="c4-n1-fixed-external-baseline-v1", kind=self.kind,
            features=self.features, columns=self.columns, constants=self.constants, scaling=self.scaling,
            linear_models=self.models if self.kind == "linear" else {},
            tree_files=tree_files, quantiles=[q for q, _ in d.QUANTILES],
            production_release_authorized=False, conformal_calibrated=False)
        content = e.encoded(state)
        (directory / "model.json").write_bytes(content)
        (directory / "model.sha256").write_text(d.sha(content)+"\n", encoding="utf-8")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        state, _ = d.checked_json(directory, "model")
        d.require(state["schema"] == "c4-n1-fixed-external-baseline-v1"
                  and state["quantiles"] == [.05, .5, .95]
                  and state["production_release_authorized"] is False
                  and state["conformal_calibrated"] is False, "Unsupported baseline state.")
        obj = cls(state["kind"])
        obj.features, obj.columns = state["features"], state["columns"]
        obj.constants, obj.scaling = state["constants"], state["scaling"]
        obj.models = state["linear_models"]
        if obj.kind == "lightgbm" and obj.columns:
            from lightgbm import Booster
            for target in d.TARGETS:
                obj.models[target] = []
                for j in range(3):
                    name = f"{target}_{j}.txt"
                    content = d.checked(directory / name, state["tree_files"][name])
                    obj.models[target].append(Booster(model_str=content.decode("utf-8")))
        return obj


def verify_manifest(protocol, manifest, original_protocol_bytes):
    # The saved validation protocol uses sorted JSON keys. Rebuilding from its
    # nested source changes injection-strength iteration order (low/medium/high
    # becomes high/low/medium). Preserve the original, checksum-verified source
    # order and compare the ENTIRE generated manifest, including scenario order.
    original = json.loads(original_protocol_bytes)
    d.require(original == protocol["factorial_source_protocol"], "Original factorial protocol differs.")
    d.require(v.create_design(original) == manifest, "New manifest differs from the frozen design.")


def prepare(root, c4):
    directory = root / "c4_artifacts/forecasting_experiments/adult" / v.VERSION
    protocol, protocol_hash = d.checked_json(directory, "validation_protocol")
    manifest, manifest_hash = d.checked_json(directory, "scenario_manifest")
    d.require(protocol["version"] == v.VERSION and protocol["experiments_executed"] == 0
              and protocol["execution_authorized"] is False
              and protocol["production_release_authorized"] is False, "Frozen validation scope differs.")
    d.require(protocol["comparison_methods"] == list(METHODS), "Comparison methods differ.")
    d.require(protocol["environment"] == e.environment(), "Frozen environment differs.")
    # Verify already frozen implementation and input bytes. New baseline code is
    # additionally bound in its execution claim before fitting begins.
    for path, digest in protocol["inputs"].items():
        d.checked(Path(path), digest)
    original_path = root / "c4_artifacts/benchmark_protocols/adult/adult-simulation-pp1-v2/protocol.json"
    d.require(str(original_path) in protocol["inputs"], "Original protocol input binding missing.")
    original_bytes = d.checked(original_path, protocol["inputs"][str(original_path)])
    verify_manifest(protocol, manifest, original_bytes)
    x, targets, index, _, pins = h.prepare(root, c4)
    hybrid = root / "c4_artifacts/forecasting_experiments/adult/adult-hybrid-development-v1/fit-v1"
    meta, digest = d.checked_json(hybrid)
    d.require(digest == protocol["fixed_hybrid_metadata_sha256"], "Primary hybrid binding differs.")
    for name, expected in meta["files"].items():
        d.checked(hybrid / name, expected)
    primary = h.HybridForecaster.load(hybrid)
    primary.predict(x.iloc[:10])
    pins.update(protocol["inputs"])
    pins[str(directory / "validation_protocol.json")] = protocol_hash
    pins[str(directory / "scenario_manifest.json")] = manifest_hash
    for path in (Path(__file__), Path(h.__file__), Path(v.__file__), Path(d.__file__), Path(e.__file__)):
        pins[str(path.resolve())] = d.sha(path.read_bytes())
    return directory, x, targets, index, pins, dict(directory=str(hybrid), metadata_sha256=digest)


def run_fit(directory, x, targets, index, pins, primary):
    output = directory / "baseline-fit-v1"
    d.require(not output.exists(), "Baseline output exists; preserve it. Do not overwrite.")
    groups = index.set_index("scenario_id").group_id
    d.require(groups.index.equals(x.index), "Group order differs.")
    for path, digest in pins.items():
        d.checked(Path(path), digest)
    output.mkdir(parents=True, exist_ok=False)
    claim = dict(purpose="same_external_development_training_fixed_baseline_fit",
        methods=METHODS, inputs=pins, primary_hybrid=primary, rows=len(x), groups=int(groups.nunique()),
        environment=e.environment(), new_validation_cases_accessed=False,
        fresh_validation_executor_implemented=False, production_release_authorized=False)
    claim_bytes = e.encoded(claim)
    (output / "claim.json").write_bytes(claim_bytes)
    (output / "claim.sha256").write_text(d.sha(claim_bytes)+"\n", encoding="utf-8")
    bindings = {"hybrid_ridge_lightgbm_lightgbm": primary}
    reload_errors = {}
    for name, kind in METHODS.items():
        print("FITTING FIXED BASELINE:", name, flush=True)
        model = FixedBaseline(kind).fit(x, targets, groups)
        before = model.predict(x)
        target_dir = output / name
        model.save(target_dir)
        after = FixedBaseline.load(target_dir).predict(x)
        np.testing.assert_allclose(before.to_numpy(), after.to_numpy(), rtol=0, atol=1e-12)
        reload_errors[name] = float(np.max(np.abs(before.to_numpy()-after.to_numpy())))
        _, digest = d.checked_json(target_dir, "model")
        bindings[name] = dict(directory=str(target_dir), model_sha256=digest)
        print("BASELINE SAVE/RELOAD PASS:", name, flush=True)
    for path, digest in pins.items():
        d.checked(Path(path), digest)
    files = {p.relative_to(output).as_posix(): d.sha(p.read_bytes())
             for p in output.rglob("*") if p.is_file()}
    metadata = e.encoded(dict(purpose="fixed_baselines_bound_before_fresh_validation", files=files,
        claim_sha256=d.sha(claim_bytes), models=bindings, model_reload_max_absolute_errors=reload_errors,
        fitting_inputs="same 106 external-development cases and original forecast_train features as hybrid",
        new_validation_cases_accessed=False, n2_calibrated=False, policy_calibrated=False,
        fresh_validation_executor_implemented=False, production_release_authorized=False))
    (output / "metadata.json").write_bytes(metadata)
    (output / "metadata.sha256").write_text(d.sha(metadata)+"\n", encoding="utf-8")
    print("ALL THREE BASELINES SAVED; HYBRID UNCHANGED. NO FRESH CASES EXECUTED.")
    print("OUTPUT:", output)
    print("METADATA SHA256:", d.sha(metadata))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    root = args.data_root or os.environ.get("C4_DATA_ROOT")
    d.require(bool(root), "Set C4_DATA_ROOT.")
    root, c4 = Path(root), Path(__file__).resolve().parents[1]
    directory, x, targets, index, pins, primary = prepare(root, c4)
    print("VERIFIED:", len(x), "external-development cases;", index.group_id.nunique(), "groups")
    if not args.execute:
        print("PLAN ONLY. --execute fits the three fixed baselines, not the 1924 validation cases.")
        return
    run_fit(directory, x, targets, index, pins, primary)


if __name__ == "__main__":
    main()
