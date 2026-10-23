"""Acceptance tests with new dataset IDs and caller-supplied model artifacts."""
from copy import deepcopy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from component3.src import live_repair as live
from component3.src.api import execute_contract_repair
from component3.src.contracts.external_contracts import (
    ContractValidationError, DependencyResolutionError, validate_c1_evidence, validate_c4_flat_contract,
)
from component3.src.contracts.repair_package import RepairPackageStatus


def observations(seed, count=600, offset=0.):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(count, 12)) + offset
    frame = pd.DataFrame(values, columns=[f"f{i}" for i in range(12)])
    frame["outcome"] = (values[:, 0] + values[:, 1] > 0).astype(int)
    return frame


class IndependentSampler:
    """Controlled generator double; all validation and model fitting remain real."""
    def __init__(self, mode="independent"):
        self.mode, self.seen = mode, None

    def fit_and_sample(self, **kwargs):
        self.seen = kwargs
        if self.mode == "copies":
            synthetic = kwargs["training_df"].head(kwargs["num_samples"]).copy()
        else:
            synthetic = observations(991, 2000)
            synthetic = synthetic[synthetic.f0 > -0.5].head(kwargs["num_samples"]).reset_index(drop=True)
        return synthetic, {"executed_generator": "tvae"}


@pytest.fixture
def supplied(tmp_path):
    reference, current = observations(1), observations(2)
    reference.to_csv(tmp_path / "ref.csv", index=False)
    current.to_csv(tmp_path / "current.csv", index=False)
    model = LogisticRegression().fit(reference.drop(columns="outcome"), 1 - reference.outcome)
    joblib.dump(model, tmp_path / "model.joblib")
    payload = {
        "c1_evidence": {"model_id": "brand-new-model", "predicted_harm": .052,
            "harm_interval": [-.01, .09], "harm_severity": "moderate", "drift_detected": True,
            "marginal_extent": "single", "interaction_suspected": .2, "affected_features": ["f0"]},
        "c2_diagnosis": {"root_cause_feature": "f0", "root_cause_category": "marginal",
            "affected_slice_rule": "f0 > -0.5", "affected_features_ranked": ["f0"],
            "slice_support": float((current.f0 > -.5).mean()), "diagnosis_stability": .9,
            "drift_impact_score": .052},
        "data_context": {"model_id": "brand-new-model", "model_uri": str(tmp_path / "model.joblib"),
            "dataset_id": "never-registered-dataset", "reference_dataset_uri": str(tmp_path / "ref.csv"),
            "current_dataset_uri": str(tmp_path / "current.csv"), "target_column": "outcome",
            "feature_columns": [f"f{i}" for i in range(12)], "categorical_features": [],
            "label_status": "AVAILABLE", "input_representation": "model_ready",
            "adaptation_budget_size": 1000,
            "repair_metric": "accuracy", "generator_epochs": 1},
    }
    return payload, tmp_path


@pytest.mark.parametrize("family", ["logistic", "forest", "pipeline"])
def test_new_models_datasets_preserve_family_and_actual_batch(supplied, family):
    payload, path = supplied
    reference = pd.read_csv(path / "ref.csv")
    model = {"logistic": LogisticRegression(), "forest": RandomForestClassifier(n_estimators=12, random_state=7),
             "pipeline": make_pipeline(StandardScaler(), LogisticRegression())}[family]
    model.fit(reference.drop(columns="outcome"), 1 - reference.outcome)
    joblib.dump(model, path / "model.joblib")
    if family == "pipeline":
        del payload["data_context"]["input_representation"]
    sampler = IndependentSampler()
    package, log = live.execute_live_repair(payload, path / "out", generator_selector=sampler)
    if family != "pipeline":
        assert package.package_status == RepairPackageStatus.APPROVED_FOR_REVIEW, package.rejection_reasons
    else:
        assert package.package_status in (RepairPackageStatus.APPROVED_FOR_REVIEW, RepairPackageStatus.REJECTED)
    assert package.dataset_id == "never-registered-dataset"
    assert package.model_id == "brand-new-model"
    if package.candidate_model_path:
        candidate = joblib.load(package.candidate_model_path)
        assert type(candidate) is type(model)
    current = pd.read_csv(path / "current.csv")
    observed = sampler.seen["training_df"]
    assert observed.f0.gt(-.5).all()
    assert set(pd.util.hash_pandas_object(observed, index=False)).issubset(set(pd.util.hash_pandas_object(current, index=False)))
    assert len(observed) < int((current.f0 > -.5).sum())  # held-out rows excluded
    assert package.utility_metrics["utility_gain"] > .01
    if package.package_status == RepairPackageStatus.APPROVED_FOR_REVIEW:
        assert package.utility_metrics["candidate_score"] >= package.utility_metrics["authentic_control_score"]
    else:
        assert package.rejection_reasons
    assert package.verify_artifact_integrity()[0]
    result = live.response_for_package(package, log)
    flat = {key: result[key] for key in live_contract_keys()}
    assert validate_c4_flat_contract(flat) == (True, [])
    if package.package_status == RepairPackageStatus.APPROVED_FOR_REVIEW:
        assert result["preview_rows"] and result["download_url"]
    else:
        assert not result["preview_rows"] and result["download_url"] is None


def live_contract_keys():
    from component3.src.contracts.external_contracts import C4_FLAT_CONTRACT_JSON_SCHEMA
    return C4_FLAT_CONTRACT_JSON_SCHEMA["required"]


@pytest.mark.parametrize("mutation,match", [
    (lambda p: p["c2_diagnosis"].update(affected_slice_rule="unknown > 2"), "Invalid C2 slice"),
    (lambda p: p["c2_diagnosis"].update(affected_slice_rule="__import__('os').system('echo unsafe')"), "Invalid C2 slice"),
    (lambda p: p["c2_diagnosis"].update(slice_support=.01), "disagrees"),
    (lambda p: p["c1_evidence"].update(model_id="other"), "model_id"),
    (lambda p: p["data_context"].update(feature_columns=["missing"]), "affected features"),
    (lambda p: p["data_context"].update(target_column="missing_label"), "missing columns"),
    (lambda p: p["data_context"].update(artifact_sha256={"model": "invalid"}), "hash mismatch"),
    (lambda p: p["data_context"].update(input_representation="raw"), "preprocessing_uri"),
    (lambda p: p["data_context"].update(task_type="regression"), "binary_classification"),
    (lambda p: p["data_context"].update(positive_label="missing"), "positive_label"),
])
def test_invalid_supplied_inputs_fail_before_generation(supplied, mutation, match):
    payload, path = supplied
    mutation(payload)
    sampler = IndependentSampler()
    with pytest.raises((ContractValidationError, DependencyResolutionError), match=match):
        live.execute_live_repair(payload, path / "out", generator_selector=sampler)
    assert sampler.seen is None


@pytest.mark.parametrize("case", ["no_harm", "labels_unavailable", "labels_unspecified", "unstable"])
def test_abstention_does_not_generate_or_claim_privacy(supplied, case):
    payload, path = supplied
    if case == "no_harm": payload["c1_evidence"].update(harm_severity="none", predicted_harm=-.01)
    elif case == "labels_unavailable": payload["data_context"]["label_status"] = "UNAVAILABLE"
    elif case == "labels_unspecified": del payload["data_context"]["label_status"]
    else: payload["c2_diagnosis"]["diagnosis_stability"] = .1
    sampler = IndependentSampler()
    package, log = live.execute_live_repair(payload, path / "out", generator_selector=sampler)
    assert sampler.seen is None
    result = live.response_for_package(package, log)
    assert result["generator"] == "NONE"
    assert result["privacy_status"] == "FAIL"
    assert result["repair_dataset_uri"] == "NONE"
    assert result["n_synthetic_rows"] == 0
    assert validate_c4_flat_contract({key: result[key] for key in live_contract_keys()})[0]


def test_copied_data_is_rejected_and_not_published(supplied):
    payload, path = supplied
    package, log = live.execute_live_repair(payload, path / "out", generator_selector=IndependentSampler("copies"))
    assert package.package_status == RepairPackageStatus.REJECTED
    assert package.privacy_risk_indicators["status"] == "FAIL"
    assert package.candidate_dataset_path is None
    assert live.response_for_package(package, log)["download_url"] is None


def test_privacy_exception_fails_closed(supplied, monkeypatch):
    payload, path = supplied
    def broken(*args, **kwargs): raise ValueError("distance calculation unavailable")
    monkeypatch.setattr(live.PrivacyValidator, "validate", broken)
    package, log = live.execute_live_repair(payload, path / "out", generator_selector=IndependentSampler())
    assert package.package_status == RepairPackageStatus.REJECTED
    assert "could not complete" in package.rejection_reasons[0]
    assert live.response_for_package(package, log)["privacy_status"] == "FAIL"


def test_current_evaluation_overlap_rejected(supplied):
    payload, path = supplied
    payload["data_context"]["evaluation_dataset_uri"] = payload["data_context"]["current_dataset_uri"]
    with pytest.raises(ContractValidationError, match="Overlapping"):
        live.execute_live_repair(payload, path / "out", generator_selector=IndependentSampler())


def test_current_v11_signed_harm_and_interaction():
    payload = {"predicted_harm": -.01, "harm_interval": [-.03, .01], "harm_severity": "none",
               "marginal_extent": "none", "interaction_suspected": .8, "affected_features": []}
    ok, errors, parsed = validate_c1_evidence(payload)
    assert ok, errors
    assert parsed.drift_type == "interaction"
    legacy = {"predicted_accuracy_drop": .5, "confidence_interval": [-.2, .9],
              "drift_type": "marginal_single", "drift_severity": "none", "affected_features": ["f0"]}
    ok, errors, parsed = validate_c1_evidence(legacy)
    assert ok, errors
    assert parsed.predicted_harm_proportion == .005
    assert parsed.confidence_interval_proportions == (-.002, .009)


def test_supplied_transformer_categorical_features_and_string_labels(supplied):
    payload, path = supplied
    frames = []
    for name in ("ref.csv", "current.csv"):
        frame = pd.read_csv(path / name)
        frame["group"] = np.where(frame.f2 > 0, "A", "B")
        frame.outcome = frame.outcome.map({0: "No", 1: "Yes"})
        frame.to_csv(path / name, index=False)
        frames.append(frame)
    features = payload["data_context"]["feature_columns"] + ["group"]
    prep = ColumnTransformer([("numeric", StandardScaler(), features[:-1]),
                              ("category", OneHotEncoder(handle_unknown="ignore"), ["group"])])
    reference = frames[0]
    model = LogisticRegression().fit(prep.fit_transform(reference[features]), reference.outcome.map({"No": 1, "Yes": 0}))
    joblib.dump(prep, path / "prep.joblib")
    joblib.dump(model, path / "model.joblib")
    payload["data_context"].update(feature_columns=features, categorical_features=["group"],
        preprocessing_uri=str(path / "prep.joblib"), target_mapping={"No": 0, "Yes": 1})
    class Sampler(IndependentSampler):
        def fit_and_sample(self, **kwargs):
            frame, info = super().fit_and_sample(**kwargs)
            frame["group"] = np.where(frame.f2 > 0, "A", "B")
            frame.outcome = frame.outcome.map({0: "No", 1: "Yes"})
            return frame, info
    package, _ = live.execute_live_repair(payload, path / "out", generator_selector=Sampler())
    assert package.package_status in (RepairPackageStatus.APPROVED_FOR_REVIEW, RepairPackageStatus.REJECTED)
    assert package.utility_metrics["utility_gain"] > 0.01, package.rejection_reasons
    assert package.downstream_handoff_metadata["artifact_sha256"]["preprocessor"]


def test_public_api_executes_new_dataset_without_registry(supplied, monkeypatch):
    from component3.src.api import validate_repair_request
    payload, path = supplied
    monkeypatch.setattr(live, "GeneratorSelector", IndependentSampler)
    assert validate_repair_request(payload)["valid"]
    contract, package, _ = execute_contract_repair(payload, output_directory=path / "out")
    assert package.package_status == RepairPackageStatus.APPROVED_FOR_REVIEW
    assert validate_c4_flat_contract(contract)[0]


def test_budget_and_duplicate_groups_are_respected(supplied):
    payload, path = supplied
    payload["data_context"]["adaptation_budget_size"] = 150
    sampler = IndependentSampler()
    package, _ = live.execute_live_repair(payload, path / "out", generator_selector=sampler)
    assert package.downstream_handoff_metadata["partition_rows"]["adaptation"] == 150
    frame = observations(88, 200)
    repeated = pd.concat([frame, frame.iloc[:50]], ignore_index=True)
    train, evaluation = live._split(repeated, "outcome", 42)
    assert not set(live._row_keys(train, payload["data_context"]["feature_columns"])) & set(live._row_keys(evaluation, payload["data_context"]["feature_columns"]))


def test_backtick_features_and_literal_boolean_words():
    from component3.src.contracts.safe_slice_evaluator import evaluate_slice_rule
    frame = pd.DataFrame({"Total Charges": [4, 20, 30], "category": ["AND", "AND", "OR"]})
    assert evaluate_slice_rule(frame, "`Total Charges` > 10 AND category == 'AND'").tolist() == [False, True, False]


def test_canonical_api_forbids_stale_candidates(supplied):
    payload, _ = supplied
    with pytest.raises(ContractValidationError, match="cached"):
        execute_contract_repair(payload, pre_generated_candidates_df=observations(33))


@pytest.mark.slow
@pytest.mark.parametrize("generator", ["TVAE", "CTGAN", "TabDDPM"])
def test_real_generator_training_new_dataset_and_model(supplied, generator):
    """Real one-epoch generator run; rejection is valid, fabricated approval is not."""
    payload, path = supplied
    payload["data_context"].update(generator=generator, max_synthetic_rows=20)
    package, log = live.execute_live_repair(payload, path / "out")
    assert package.generator_info["generator"] == generator, package.rejection_reasons
    assert package.package_status in (RepairPackageStatus.REJECTED, RepairPackageStatus.APPROVED_FOR_REVIEW)
    assert package.downstream_handoff_metadata["partition_rows"]["generator_training_slice"] >= 30
    assert Path(log.package_path).is_file()
    if package.package_status == RepairPackageStatus.REJECTED:
        assert package.rejection_reasons
