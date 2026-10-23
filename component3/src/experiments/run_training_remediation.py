"""One fixed training comparison on a saved real upstream batch, without an evaluation search.

The old 20-epoch configuration receives quality/privacy screening only. The
predeclared improved configuration gets one independent utility evaluation if
it passes the unchanged gates. This is not a search for a favourable seed.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import uuid

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "webapp/backend"))
from app.settings import c3_data_context  # sets only the integrated artifact defaults
from c2.src.outputs import C2DiagnosisEngine
from component3.src import live_repair as live
from component3.src.generators.generator_selector import GeneratorSelector
from component3.src.validation.fidelity_validator import FidelityValidator
from component3.src.validation.privacy_validator import PrivacyValidator


def run(record_path, report_path, reuse_baseline=None):
    torch.set_num_threads(2)
    start = time.perf_counter()
    record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    pipeline = record["pipeline"]
    source = pipeline.get("component_sources", {}).get("c1")
    if source != "real":
        raise ValueError("This experiment requires a saved real C1 run.")
    context = {**c3_data_context(pipeline["context"]["model_id"]), **pipeline["context"]}
    # Preserve the original model, batch and labels, then declare the study config.
    context.update(generator_epochs=200, generator_discrete_numeric_features=[
        "age", "education_num", "capital_gain", "capital_loss", "hours_per_week"])
    context.pop("generator_batch_size", None)
    context["parent_pipeline_run_id"] = pipeline["run_id"]
    context["run_id"] = "training-study-" + uuid.uuid4().hex[:12]
    context["experiment_mode"] = "saved_real_c1_fresh_c2_training_remediation"
    current = pd.read_csv(context["current_dataset_uri"])
    features = context["feature_columns"]
    engine = C2DiagnosisEngine(log_path=None)
    engine.warmup([engine.a["test_df"].sample(1000, random_state=1000 + i) for i in range(2)])
    c1 = pipeline["outputs"]["c1"]
    diagnosis = engine.diagnose(current[features], c1, c1_trusted=True)["to_c3"]
    reference = pd.read_csv(context["reference_dataset_uri"])
    independent = reference.loc[~live._row_keys(reference, features).isin(set(live._row_keys(current, features)))]
    source_train, _ = live._split(independent, context["target_column"], 42)
    categories = context["categorical_features"]
    numeric = [f for f in features if f not in categories]
    comparison = {}
    if reuse_baseline:
        previous = json.loads(Path(reuse_baseline).read_text(encoding="utf-8"))
        previous_batch = previous["candidate"]["provenance"]["artifact_paths"]["current_dataset"]
        if (Path(previous_batch).resolve() != Path(context["current_dataset_uri"]).resolve()
                or previous["c2"] != diagnosis):
            raise ValueError("Baseline reuse requires the same exact batch and diagnosis.")
        previous_hashes = previous["candidate"]["provenance"]["artifact_sha256"]
        for key, path in (("current_dataset", context["current_dataset_uri"]),
                          ("reference_dataset", context["reference_dataset_uri"]), ("model", context["model_uri"])):
            if live.compute_file_sha256(path) != previous_hashes[key]:
                raise ValueError(f"Cannot reuse baseline after artifact content changed: {key}")
        comparison.update(baseline=previous["baseline"], baseline_reused_from=str(reuse_baseline))

    class ComparedGenerator:
        def fit_and_sample(self, **kwargs):
            if reuse_baseline:
                return GeneratorSelector().fit_and_sample(**kwargs)
            baseline_kwargs = {**kwargs, "categorical_columns": categories + [context["target_column"]],
                "numerical_columns": numeric, "generator_kwargs": {"epochs": 20, "batch_size": 500}}
            baseline, generation = GeneratorSelector().fit_and_sample(**baseline_kwargs)
            fidelity = FidelityValidator(max_mean_ks_statistic=.35).validate(
                baseline, kwargs["training_df"], numeric, categories + [context["target_column"]])
            privacy = PrivacyValidator(min_dcr_p05=.05, max_near_duplicate_ratio=0.).validate(
                baseline[features], pd.concat([source_train, kwargs["training_df"]], ignore_index=True)[features],
                continuous_columns=numeric, discrete_columns=categories)
            comparison.update(baseline={"generation": generation, "fidelity": fidelity.to_dict(),
                "privacy": privacy.to_dict(), "utility": "Not evaluated; baseline used for training diagnostics only."})
            return GeneratorSelector().fit_and_sample(**kwargs)

    payload = {"c1_evidence": c1, "c2_diagnosis": diagnosis, "data_context": context}
    package, log = live.execute_live_repair(payload, generator_selector=ComparedGenerator())
    result = live.response_for_package(package, log)
    report = {"protocol": __doc__, "parent_pipeline_run_id": pipeline["run_id"], "c2": diagnosis,
              **comparison, "candidate": result, "elapsed_seconds": round(time.perf_counter() - start, 3)}
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"report": str(destination), "status": result["execution_status"],
                      "reason": result["reason"], "generator": result["generator"],
                      "generated_rows": result["provenance"].get("generated_rows"),
                      "elapsed_seconds": report["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--report", type=Path, default=ROOT / "component3/results/training_remediation/controlled_study.json")
    parser.add_argument("--reuse-baseline", type=Path)
    arguments = parser.parse_args()
    run(arguments.record, arguments.report, arguments.reuse_baseline)
