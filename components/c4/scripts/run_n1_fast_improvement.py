"""Development-only larger-cohort N1 experiment; reuse saved candidate models.

No TVAE or candidate fitting. No original artifact changes, release, or Git.
The official external cohort is used for DEVELOPMENT labels in this experiment.
Grouped CV on these labels is not an untouched final test or a 95% guarantee.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import re
import sys
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import diagnose_n1_development as d
import experiment_n1_development as e

RAW_NAMES = "age workclass fnlwgt education education_num marital_status occupation relationship race sex capital_gain capital_loss hours_per_week native_country income".split()
FEATURES = "age workclass education_num marital_status occupation relationship race sex capital_gain capital_loss hours_per_week native_country".split()
CONTINUOUS = "age education_num capital_gain capital_loss hours_per_week".split()
CATEGORICAL = [c for c in FEATURES if c not in CONTINUOUS]
RAW_SHA = "5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d"
EXTERNAL_SHA = "a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05"
EXTERNAL_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
REFERENCE_SHA = "84cbc88cbfac5567a3954445aa8b154d58644d6eef3b43c24eec9fd547474ad3"
TEST_SHA = "6823ba1c88636d6c66d10b27bd16e983899628afc7567297eba0f57cae823d1a"


def raw_table(contents):
    records = []
    for row in csv.reader(io.StringIO(contents.decode("utf-8-sig"))):
        if not row or not any(c.strip() for c in row) or row[0].lstrip().startswith("|"):
            continue
        d.require(len(row) == 15, "Raw Adult record must have 15 fields.")
        fields = [c.strip() for c in row]
        if not records and fields == RAW_NAMES:
            continue
        records.append(fields)
    d.require(bool(records), "Empty raw Adult file.")
    frame = pd.DataFrame(records, columns=RAW_NAMES).replace("?", np.nan)
    labels = frame.income.str.rstrip(".")
    d.require(labels.isin(["<=50K", ">50K"]).all(), "Invalid raw income label.")
    frame["income"] = labels.eq(">50K").astype(int)
    for c in [*CONTINUOUS, "fnlwgt"]:
        try:
            frame[c] = pd.to_numeric(frame[c], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid/nonfinite raw numeric value: {c}") from exc
        d.require(np.isfinite(frame[c].to_numpy(float)).all(), "Missing/nonfinite raw numeric value.")
    return frame


def prepare_cohort(raw, external, reference, original_test):
    """Reconstruct original transform, verifying BOTH existing processed CSVs.

    Imputation/encoding/scaling fit on reference only. No external fitting.
    This is a compatibility reconstruction, not a new preprocessing convention.
    """
    ref, test = train_test_split(raw, test_size=0.2, stratify=raw.income, random_state=42)
    transforms = {}
    for c in CATEGORICAL:
        transforms[c] = {"mode": ref[c].mode().iloc[0]}
    encoders = {}
    ref_input = ref[FEATURES].copy()
    for c in CATEGORICAL:
        le = LabelEncoder().fit(ref_input[c].fillna(transforms[c]["mode"]).astype(str))
        encoders[c] = le
        ref_input[c] = le.transform(ref_input[c].fillna(transforms[c]["mode"]).astype(str))
        transforms[c]["classes"] = le.classes_.tolist()
    scaler = StandardScaler().fit(ref_input[CONTINUOUS])
    def transform(frame):
        x = frame[FEATURES].copy()
        unknown = {}
        for c, le in encoders.items():
            values = x[c].fillna(transforms[c]["mode"]).astype(str)
            known = values.isin(le.classes_)
            unknown[c] = int((~known).sum())
            # Original shared transform's fixed compatibility policy. Report it.
            x[c] = le.transform(values.where(known, le.classes_[0]))
        x[CONTINUOUS] = scaler.transform(x[CONTINUOUS])
        x["target"] = frame.income.to_numpy()
        d.require(np.isfinite(x.to_numpy(float)).all(), "Nonfinite processed cohort.")
        return x, unknown
    for expected, original in ((reference, ref), (original_test, test)):
        actual, _ = transform(original)
        d.require(list(expected) == [*FEATURES, "target"] and actual.shape == expected.shape,
                  "Original processed schema/shape differs.")
        d.require(np.allclose(actual.to_numpy(float), expected.to_numpy(float), rtol=0, atol=1e-12),
                  "Original preprocessing reconstruction failed; stop, do not refit to external data.")
    all_old, _ = transform(raw)
    fresh, unknown = transform(external)
    # Compare features only, including cross-label representations. All prior
    # repair/release/future source records are excluded, not just model reference.
    seen = set(pd.util.hash_pandas_object(all_old[FEATURES], index=False))
    overlap = pd.util.hash_pandas_object(fresh[FEATURES], index=False).isin(seen)
    fresh = fresh.loc[~overlap].copy()
    duplicates = fresh.duplicated(keep="first")
    duplicate_count = int(duplicates.sum())
    fresh = fresh.loc[~duplicates].copy()
    fresh.index = pd.Index(1_000_000 + fresh.index.to_numpy(), name="source_row_id")
    d.require(len(fresh) > 1624 and fresh.target.nunique() == 2, "Insufficient eligible external cohort.")
    return fresh, {"external_raw_rows": len(external), "old_feature_overlap_rows": int(overlap.sum()),
                   "external_exact_duplicate_extras": duplicate_count, "eligible_rows": len(fresh),
                   "unknown_category_rows_before_filtering": unknown,
                   "unknown_category_policy": "Original shared encoder compatibility: first fitted class; reported, never silently refitted.",
                   "feature_only_old_overlap_exclusion": True,
                   "shared_external_pool_between_training_cases": True}


def global_ridge(x, y, groups, valid):
    columns = [c for c in x if x[c].nunique() > 1]
    correlation = x[columns].corr().abs().to_numpy()
    keep = []
    for j in range(len(columns)):
        if not any(correlation[j, k] > 1 - 1e-10 for k in keep):
            keep.append(j)
    columns = [columns[k] for k in keep]
    d.require(bool(columns), "No varying global predictors.")
    w = d.group_weights(groups)
    w /= w.mean()
    scaler = StandardScaler().fit(x[columns], sample_weight=w)
    a, b = scaler.transform(x[columns]), scaler.transform(valid[columns])
    model = Ridge(alpha=100.0).fit(a, y, sample_weight=w)
    center = model.predict(b)
    residual = np.asarray(y) - model.predict(a)
    lo, hi = [d.weighted_quantile(residual, w, q) for q in (0.05, 0.95)]
    return np.column_stack((center + lo, center, center + hi))


def compare(x, targets, index, progress):
    config = next(c for c in e.candidates() if c["id"] == "lgb_original_all")
    linear = next(c for c in e.candidates() if c["id"] == "quantile_0.001_all")
    constant = e.candidates()[0]
    forecasts = {name: pd.DataFrame(index=x.index, dtype=float) for name in
                 ("lightgbm", "linear", "constant_quantiles", "targetwise_challenger")}
    for fold, (train, valid) in enumerate(GroupKFold(5).split(index, groups=index.group_id), 1):
        for target in d.TARGETS:
            args = (x.iloc[train], targets.iloc[train][target], index.iloc[train].group_id, x.iloc[valid])
            tree = e.fit_predict(config, *args)
            const = e.fit_predict(constant, *args)
            hybrid = (global_ridge(*args) if target == d.TARGETS[0] else
                      tree if target == d.TARGETS[1] else const)
            for method, raw in (("lightgbm", tree), ("linear", e.fit_predict(linear, *args)),
                                ("constant_quantiles", const), ("targetwise_challenger", hybrid)):
                pred = e.ordered(raw)
                for j, (_, label) in enumerate(d.QUANTILES):
                    forecasts[method].loc[x.index[valid], target + "__" + label] = pred[:, j]
        progress(f"N1 GROUPED DEVELOPMENT FOLD {fold}/5 complete")
    rows, folds = [], []
    split_list = list(GroupKFold(5).split(index, groups=index.group_id))
    for method, pred in forecasts.items():
        d.require(not pred.isna().any().any(), "Incomplete grouped forecasts.")
        for target in d.TARGETS:
            p = pred[[target + "__" + label for _, label in d.QUANTILES]].to_numpy()
            rows.append(dict(method=method, target=target, **e.losses(targets[target], p, index.group_id)))
            for fold, (_, valid) in enumerate(split_list, 1):
                folds.append(dict(method=method, target=target, fold=fold,
                    **e.losses(targets.iloc[valid][target], p[valid], index.iloc[valid].group_id)))
    return pd.DataFrame(rows), pd.DataFrame(folds), forecasts


def environment():
    return {"python": platform.python_version(), **{p: importlib.metadata.version(p) for p in
            ("numpy", "pandas", "scikit-learn", "lightgbm", "xgboost")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--run-id", default="adult-external-development-v3-01")
    parser.add_argument("--raw-adult", type=Path)
    parser.add_argument("--external-test", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--freeze-spec", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    value = args.data_root or os.environ.get("C4_DATA_ROOT")
    d.require(bool(value), "Set C4_DATA_ROOT.")
    root = Path(value)
    d.require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", args.run_id) is not None, "Unsafe run ID.")
    c4 = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(c4 / "src"))
    print("VERIFYING EXISTING FORECAST_TRAIN SNAPSHOT", flush=True)
    x, old_targets, index, _, _, pins = d.load_inputs(root, c4)
    output = root / "c4_artifacts/forecasting_experiments/adult" / args.run_id
    raw_path = args.raw_adult or root / "raw/adult.data"
    raw_bytes = d.checked(raw_path, RAW_SHA)
    reference_bytes = d.checked(root / "uci_adult_data/reference.csv", REFERENCE_SHA)
    test_bytes = d.checked(root / "uci_adult_data/test.csv", TEST_SHA)
    pins.update({str(raw_path): RAW_SHA, str(root / "uci_adult_data/reference.csv"): REFERENCE_SHA,
                 str(root / "uci_adult_data/test.csv"): TEST_SHA})
    scripts = [Path(__file__).resolve(), Path(d.__file__).resolve(), Path(e.__file__).resolve(),
               c4 / "scripts/_all_mechanism_transforms.py", c4 / "scripts/_future_research.py",
               c4 / "src/c4/models/candidate.py", c4 / "src/c4/models/champion.py"]
    for path in scripts:
        pins[str(path)] = d.sha(path.read_bytes())
    specification = {"version": "adult-external-development-v3", "inputs": pins,
                     "environment": environment(), "external_url": EXTERNAL_URL,
                     "dataset_credit": "Becker, B. and Kohavi, R. (1996). Adult. UCI Machine Learning Repository. DOI:10.24432/C5XW20. CC BY 4.0.",
                     "external_sha256": EXTERNAL_SHA, "external_file_used_for": "DEVELOPMENT ONLY",
                     "scope": "Original 106 forecast_train cases; fixed saved candidates; new external cohort.",
                     "fit_candidates": False, "fit_TVAE": False, "fit_final_N1_models": False,
                     "grouped_cv": "Original five GroupKFold splits; equal total weight per nonempty group.",
                     "N1_models": "Original fixed LightGBM and linear; constant quantiles; fixed targetwise challenger.",
                     "challenger": "Global Ridge(alpha=100) after train-only correlation pruning/scaling; original LightGBM slice; constant quantiles complement.",
                     "raw_ridge_bands": "Train-residual quantiles; uncalibrated, require new N2 before controller use.",
                     "cohort_policy": "Apply reconstructed original reference-fitted transform; verify old processed reference and test; remove feature overlap with ALL original raw Adult rows; remove exact external duplicate records.",
                     "future_drift": "Original frozen per-case mechanism, strength, true-slice rule and sealed-future RNG seed.",
                     "limitations": ["Shared external cohort does not create independent source datasets.",
                                     "Recipes informed by earlier development studies; CV is exploratory development evidence.",
                                     "New population changes labels; do not compare old/new R2 as an isolated model effect.",
                                     "No calibration/policy/outer case evaluation. No final-test or release claim."]}
    spec_bytes = e.encoded(specification)
    if args.freeze_spec:
        d.require(not output.exists(), "Run exists; preserve it and use a new run ID.")
        output.mkdir(parents=True)
        (output / "specification.json").write_bytes(spec_bytes)
        (output / "specification.sha256").write_text(d.sha(spec_bytes) + "\n", encoding="utf-8")
        print("FROZEN:", output, flush=True)
        print("Next --execute; source download only if no --external-test supplied.", flush=True)
        return
    if not args.execute:
        print("PLAN: 106 saved candidates; 0 TVAE fits; 0 candidate fits; new-cohort labels and grouped N1 CV.")
        print("Use --freeze-spec, then --execute.")
        return
    expected = (output / "specification.sha256").read_text().strip()
    d.require(d.checked(output / "specification.json", expected) == spec_bytes,
              "Frozen code/input/environment differs.")
    results = output / "results"
    d.require(not results.exists(), "Results/partial execution exist; preserve them and use a new run ID.")
    results.mkdir()
    with (results / "progress.log").open("x", encoding="utf-8") as log:
        def progress(message):
            line = datetime.now(timezone.utc).isoformat() + " " + message
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
        if args.external_test:
            external_bytes = d.checked(args.external_test, EXTERNAL_SHA)
        else:
            progress("DOWNLOADING OFFICIAL UCI adult.test (about 2 MB)")
            with urlopen(EXTERNAL_URL, timeout=60) as response:
                external_bytes = response.read(4_000_000)
            d.require(d.sha(external_bytes) == EXTERNAL_SHA, "Official external file checksum differs.")
        (results / "adult.test").write_bytes(external_bytes)
        clean, cohort_info = prepare_cohort(raw_table(raw_bytes), raw_table(external_bytes),
            pd.read_csv(io.BytesIO(reference_bytes)), pd.read_csv(io.BytesIO(test_bytes)))
        (results / "cohort_audit.json").write_bytes(e.encoded(cohort_info))
        clean.to_csv(results / "external_clean.csv", index_label="source_row_id")
        progress(f"EXTERNAL COHORT READY: {len(clean)} rows; original preprocessing reproduced")
        from _all_mechanism_transforms import transform_future_role
        from _future_research import verify_freeze_dependencies
        from c4.models import load_candidate, load_champion
        from c4.scenarios.config import ScenarioConfig
        from c4.scenarios.slices import SliceRule
        from c4.scenarios.adapter import _load_shared_functions
        from c4.scenarios.feature_metadata import load_adult_feature_metadata
        protocol_dir = root / "c4_artifacts/benchmark_protocols/adult" / d.PROTOCOL
        protocol_freeze, protocol_hash = d.checked_json(protocol_dir, "freeze")
        protocol_bytes = d.checked(protocol_dir / "protocol.json", protocol_freeze["protocol_sha256"])
        protocol = json.loads(protocol_bytes)
        rule = protocol["true_slice"]
        true_rule = SliceRule(name=rule["name"], feature=rule["feature"], operator=rule["operator"], value=rule["resolved_threshold"])
        shared, drift_hash = _load_shared_functions()
        feature_metadata = load_adult_feature_metadata()
        champion, champion_meta = load_champion(root / "c4_artifacts/champion/adult" / protocol["source_versions"]["champion"])
        d.require(champion_meta["reference_csv_sha256"] == REFERENCE_SHA
                  and champion_meta["model_sha256"] == protocol_freeze["champion_model_sha256"], "Champion binding differs.")
        benchmark = root / "c4_artifacts/benchmark_runs/adult" / d.PROTOCOL
        (results / "cases").mkdir()
        target_rows, records = [], []
        for number, row in enumerate(index.itertuples(index=False), 1):
            sid = row.scenario_id
            d.require(re.fullmatch(r"[A-Za-z0-9_-]+", sid) is not None, "Unsafe scenario ID.")
            run = benchmark / sid
            snapshot = json.loads(d.checked(run / "pre_release_freeze-v1/freeze.json", row.decision_freeze_sha256))
            scenario = snapshot["scenario"]
            d.require(scenario["scenario_id"] == sid and scenario["group_id"] == row.group_id
                      and scenario["split_role"] == "forecast_train"
                      and snapshot["protocol_freeze_sha256"] == protocol_hash, "Scenario scope/binding differs.")
            verify_freeze_dependencies(snapshot)
            old = json.loads(d.checked(run / "future_research-v1/outcomes.json", row.outcomes_sha256))
            d.require(old["scenario"]["scenario_id"] == sid and old["scenario"]["split_role"] == "forecast_train"
                      and old["decision_freeze_sha256"] == row.decision_freeze_sha256
                      and old["shared_drift_types_sha256"] == drift_hash
                      and old["feature_metadata_sha256"] == feature_metadata.source_sha256, "Original source binding differs.")
            candidate, candidate_meta = load_candidate(run / "scoped_candidate_validation-v1")
            d.require(candidate_meta["model_sha256"] == old["candidate_model_sha256"]
                      and champion_meta["model_sha256"] == old["champion_model_sha256"]
                      and candidate_meta["champion_model_sha256"] == champion_meta["model_sha256"], "Saved model binding differs.")
            config = ScenarioConfig(mechanism=scenario["mechanism"], magnitude=scenario["magnitude"],
                features=tuple(scenario["features"]), interaction_pairs=tuple(tuple(v) for v in scenario["interaction_pairs"]),
                seed=scenario["drift_seeds"]["sealed_future"])
            transformed, mask = transform_future_role(clean, config=config, true_rule=true_rule,
                                                     shared=shared, feature_types=feature_metadata.feature_types)
            truth = transformed.target.to_numpy()
            a = np.asarray(champion.predict(transformed[FEATURES]))
            b = np.asarray(candidate.predict(transformed[FEATURES]))
            d.require(a.shape == b.shape == truth.shape and np.isin(a, [0, 1]).all()
                      and np.isin(b, [0, 1]).all(), "Invalid native-model predictions.")
            paired = (b == truth).astype(float) - (a == truth).astype(float)
            masks = (np.ones(len(clean), bool), mask.to_numpy(), ~mask.to_numpy())
            d.require(all(pop.any() for pop in masks), "Empty external target population.")
            values = {target: float(paired[pop].mean()) for target, pop in zip(d.TARGETS, masks)}
            target_rows.append(dict(scenario_id=sid, **values))
            records.append(dict(scenario_id=sid, group_id=row.group_id,
                decision_freeze_sha256=row.decision_freeze_sha256, candidate_model_sha256=candidate_meta["model_sha256"],
                original_outcomes_sha256=row.outcomes_sha256, population_rows=[int(pop.sum()) for pop in masks]))
            (results / "cases" / (sid + ".json")).write_bytes(e.encoded(dict(
                **records[-1], primary_targets=values, external_file_sha256=EXTERNAL_SHA,
                scope="forecast_train_external_development_only")))
            progress(f"LARGER-COHORT TARGETS {number}/106 complete (saved candidate reused)")
        new_targets = pd.DataFrame(target_rows).set_index("scenario_id").loc[x.index, list(d.TARGETS)]
        d.validate_tables(x, new_targets, index)
        x.to_csv(results / "features.csv", index_label="scenario_id")
        new_targets.to_csv(results / "targets.csv", index_label="scenario_id")
        index.to_csv(results / "scenario_index_original_bindings.csv", index=False)
        (results / "case_bindings.json").write_bytes(e.encoded(records))
        old_summary, _, _ = compare(x, old_targets, index, progress)
        old_summary["cohort"] = "original_development"
        summary, fold_metrics, forecasts = compare(x, new_targets, index, progress)
        summary["cohort"] = "external_development"
        combined = pd.concat([old_summary, summary], ignore_index=True)
        combined.to_csv(results / "metric_summary.csv", index=False)
        fold_metrics.to_csv(results / "external_fold_metrics.csv", index=False)
        for method, forecast in forecasts.items():
            forecast.to_csv(results / (method + "_grouped_oof.csv"), index_label="scenario_id")
        for path, digest in pins.items():
            d.checked(Path(path), digest)
        d.require(_load_shared_functions()[1] == drift_hash, "Shared drift code changed during execution.")
        progress("COMPLETE. No TVAE/candidate fits. No final N1 model or N2/policy/controller replacement.")
    files = {p.relative_to(results).as_posix(): d.sha(p.read_bytes())
             for p in sorted(results.rglob("*")) if p.is_file()}
    metadata = e.encoded(dict(specification_sha256=d.sha(spec_bytes), files=files, cohort=cohort_info,
        interpretation="Exploratory grouped CV on development labels from a new fixed cohort; not final outer results.",
        training_cases=106, nonempty_groups=29, candidate_models_refitted=False,
        calibration_policy_outer_cases_accessed=False, production_release_authorized=False))
    (results / "metadata.json").write_bytes(metadata)
    (results / "metadata.sha256").write_text(d.sha(metadata) + "\n", encoding="utf-8")
    display = combined.copy()
    display["MAE_pp"], display["RMSE_pp"], display["Pinball_pp"] = display.mae*100, display.rmse*100, display.mean_pinball*100
    print(display[["cohort", "method", "target", "MAE_pp", "RMSE_pp", "r2", "Pinball_pp"]].to_string(index=False), flush=True)
    print("Compare models WITHIN the same cohort. Cohort changes alter targets and cannot isolate model improvement.")
    print("OUTPUT:", results)
    print("METADATA SHA256:", d.sha(metadata))


if __name__ == "__main__":
    main()
