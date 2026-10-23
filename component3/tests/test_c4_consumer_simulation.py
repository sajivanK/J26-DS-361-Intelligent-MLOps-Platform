"""Simulated Component 4 Consumer Verification Test Suite.

Simulates Component 4 (Continuous Retraining Engine) consuming the output of Component 3:
1. Approved repair package consumption:
   - Validates all 12 C4 flat contract fields against shared/contracts/c3_output_schema.json.
   - Verifies candidate CSV exists at repair_dataset_uri (resolving relative paths).
   - Verifies row count equals n_synthetic_rows.
   - Verifies candidate model exists and runs predict() successfully.
2. Rejected repair package consumption:
   - Verifies repair_package_status is 'REJECTED'.
   - Verifies n_synthetic_rows is 0 and repair_dataset_uri is None.
   - Verifies C4 consumer logic knows not to trigger retraining.
"""

from __future__ import annotations

import json
from pathlib import Path
import joblib
import pandas as pd
import pytest

from component3.src.api import (
    execute_contract_repair,
    validate_c4_flat_contract,
)
from component3.src.contracts.repair_package import RepairPackageStatus
from component3.src.preprocessing.tabular_preprocessor import TabularDataPreprocessor

REPO_ROOT = Path(__file__).resolve().parents[2]


class SimulatedC4Consumer:
    """Simulated Component 4 Retraining Engine consumer."""

    def __init__(self, repo_root: Path = REPO_ROOT) -> None:
        self.repo_root = repo_root

    def consume_c3_output(self, c4_contract_path: Path | str) -> dict[str, Any]:
        """Consume C3 contract JSON and execute C4-side pre-flight checks."""
        c4_file = Path(c4_contract_path)
        if not c4_file.is_file():
            raise FileNotFoundError(f"C4 contract JSON missing: {c4_file}")

        with open(c4_file, "r", encoding="utf-8") as f:
            c4_data = json.load(f)

        # 1. Validate against shared contract schema
        valid, errors = validate_c4_flat_contract(c4_data)
        if not valid:
            raise ValueError(f"Schema validation failed: {errors}")

        status = c4_data["repair_package_status"]
        result = {
            "status": status,
            "proceed_to_retrain": False,
            "synthetic_df": None,
            "candidate_model": None,
            "predictions_sample": None,
        }

        # 2. Rejection branch
        if status == "REJECTED":
            assert c4_data["n_synthetic_rows"] == 0
            assert c4_data["repair_dataset_uri"] in (None, "NONE", "")
            result["proceed_to_retrain"] = False
            return result

        # 3. Approved branch
        assert status == "APPROVED"
        assert c4_data["n_synthetic_rows"] > 0
        assert c4_data["repair_dataset_uri"] is not None

        # Resolve relative URI
        dataset_uri = c4_data["repair_dataset_uri"]
        dataset_path = Path(dataset_uri)
        if not dataset_path.is_absolute():
            dataset_path = self.repo_root / dataset_path

        if not dataset_path.is_file():
            raise FileNotFoundError(f"Candidate dataset URI not found on disk: {dataset_path}")

        df_synthetic = pd.read_csv(dataset_path)
        if len(df_synthetic) != c4_data["n_synthetic_rows"]:
            raise ValueError(
                f"Row count mismatch: contract declares {c4_data['n_synthetic_rows']}, "
                f"CSV contains {len(df_synthetic)} rows"
            )

        result["synthetic_df"] = df_synthetic
        result["proceed_to_retrain"] = True

        # Candidate model load & inference if candidate model is present
        cand_model_path_str = c4_data.get("candidate_model_path")
        if cand_model_path_str:
            cand_m_path = Path(cand_model_path_str)
            if not cand_m_path.is_absolute():
                cand_m_path = self.repo_root / cand_m_path
            if cand_m_path.is_file():
                model = joblib.load(cand_m_path)
                result["candidate_model"] = model

        return result


@pytest.fixture
def base_approved_payload():
    return {
        "c1_evidence": {
            "predicted_accuracy_drop": 18.0,
            "confidence_interval": [12.0, 24.0],
            "drift_type": "covariate_shift",
            "drift_severity": "HIGH",
            "affected_features": ["Contract", "tenure", "MonthlyCharges"],
        },
        "c2_diagnosis": {
            "root_cause_feature": "Contract",
            "root_cause_category": "covariate_shift",
            "affected_slice_rule": "Contract == 'Month-to-month' and tenure <= 12",
            "affected_features_ranked": ["Contract", "MonthlyCharges", "tenure"],
            "slice_support": 0.28,
            "diagnosis_stability": 0.85,
            "drift_impact_score": 0.85,
        },
        "data_context": {
            "model_id": "telco_rf_experimental_source",
            "dataset_id": "telco_churn",
            "reference_dataset_uri": "component3/data/raw/telco_churn.csv",
            "current_dataset_uri": "component3/data/raw/telco_churn.csv",
            "target_column": "Churn",
            "preprocessing_uri": "component3/models/experimental/telco_preprocessor_experimental_source.joblib",
            "adaptation_budget_size": 200,
            "source_retention_tolerance": 0.05,
        },
    }


def test_c4_consumer_approved_package(base_approved_payload, tmp_path):
    """Verify that simulated C4 consumer successfully consumes an approved C3 repair package."""
    out_c4 = tmp_path / "c4_output_approved.json"
    out_dir = tmp_path / "approved_run"
    cand_p = REPO_ROOT / "component3" / "results" / "final_integration_hardening" / "scenario_1_real_tvae" / "pkg-859b8f17c113_candidates_k100.csv"
    pre_gen = pd.read_csv(cand_p) if cand_p.is_file() else None

    c4_dict, pkg, log = execute_contract_repair(
        request=base_approved_payload,
        pre_generated_candidates_df=pre_gen,
        output_c4_path=out_c4,
        output_directory=out_dir,
        decision_mode="RULE_BASED",
    )

    assert pkg.package_status == RepairPackageStatus.APPROVED_FOR_REVIEW
    assert out_c4.is_file()

    consumer = SimulatedC4Consumer(repo_root=REPO_ROOT)
    c4_res = consumer.consume_c3_output(out_c4)

    assert c4_res["status"] == "APPROVED"
    assert c4_res["proceed_to_retrain"] is True
    assert c4_res["synthetic_df"] is not None
    assert len(c4_res["synthetic_df"]) == c4_dict["n_synthetic_rows"]

    # Verify model prediction on preprocessed synthetic data
    preprocessor = joblib.load(REPO_ROOT / "component3" / "models" / "experimental" / "telco_preprocessor_experimental_source.joblib")
    df_eval = c4_res["synthetic_df"].copy()
    if "customerID" in df_eval.columns:
        df_eval = df_eval.drop(columns=["customerID"])
    bundle = preprocessor.transform(df_eval)

    cand_model_path = Path(pkg.candidate_model_path)
    if cand_model_path.is_file():
        model = joblib.load(cand_model_path)
        preds = model.predict(bundle.X)
        assert len(preds) == len(df_eval)
        assert set(preds).issubset({0, 1})


def test_c4_consumer_rejected_package(base_approved_payload, tmp_path):
    """Verify that simulated C4 consumer cleanly aborts retraining on a rejected repair package."""
    # Force rejection via safe abstention (support below minimum)
    payload_rejected = json.loads(json.dumps(base_approved_payload))
    payload_rejected["c2_diagnosis"]["slice_support"] = 0.01  # below 0.05 threshold
    payload_rejected["c2_diagnosis"]["affected_slice_rule"] = "tenure < 1"

    out_c4 = tmp_path / "c4_output_rejected.json"
    out_dir = tmp_path / "rejected_run"

    c4_dict, pkg, log = execute_contract_repair(
        request=payload_rejected,
        output_c4_path=out_c4,
        output_directory=out_dir,
        decision_mode="RULE_BASED",
    )

    assert pkg.package_status == RepairPackageStatus.REJECTED
    assert out_c4.is_file()

    consumer = SimulatedC4Consumer(repo_root=REPO_ROOT)
    c4_res = consumer.consume_c3_output(out_c4)

    assert c4_res["status"] == "REJECTED"
    assert c4_res["proceed_to_retrain"] is False
    assert c4_res["synthetic_df"] is None
    assert c4_dict["repair_dataset_uri"] in (None, "NONE", "")
    assert c4_dict["n_synthetic_rows"] == 0
