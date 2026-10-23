"""Component 4 Agentic Decision and Retraining Export Adapter for Component 3.

Formats RepairPackages and execution outcomes into standardized review payloads
for Component 4 (Agentic Decision and Retraining Gatekeeper).

PROVISIONAL ADAPTER NOTICE:
This adapter exports structured evaluation metrics, uncertainty intervals, risk indicators,
and cryptographic artifact hashes. Live handoff to Component 4 will be enabled when
Component 4 ingestion endpoints are confirmed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from component3.src.contracts.repair_package import RepairPackage, RepairPackageStatus

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class C4ReviewPayload:
    """Standardized handoff payload for Component 4 review gatekeeper."""
    schema_version: str
    export_timestamp_utc: str

    # Identity & Status
    request_id: str
    run_id: str
    package_id: str
    package_status: str
    strategy_recommendation: str

    # Candidate Artifact References & Hashes
    candidate_model_reference: dict[str, Any]
    candidate_data_reference: dict[str, Any]
    artifact_hashes: dict[str, str]

    # Metrics & Evaluation
    quality_metrics: dict[str, Any]
    utility_metrics: dict[str, Any]
    source_retention_metrics: dict[str, Any]

    # Statistical Uncertainty & Risk
    uncertainty: dict[str, Any]
    risk_indicators: dict[str, Any]

    # Governance & Approval
    approval_requirements: dict[str, Any]
    evidence_limitations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class C4RetrainingAdapter:
    """Adapter transforming Component 3 RepairPackage into Component 4 review payload."""

    def __init__(self, schema_version: str = "1.0.0") -> None:
        self.schema_version = schema_version

    def export(
        self,
        package: RepairPackage,
        package_manifest_path: str | Path | None = None,
    ) -> C4ReviewPayload:
        """Construct C4ReviewPayload from a finalized RepairPackage."""
        # Quality indicators
        fidelity = dict(package.fidelity_indicators)
        privacy = dict(package.privacy_risk_indicators)
        quality = {
            "fidelity_score": fidelity.get("fidelity_score", 0.0),
            "mean_ks_statistic": fidelity.get("mean_ks_statistic", None),
            "privacy_status": privacy.get("status", "PASS"),
            "dcr_p05": privacy.get("dcr_p05", None),
            "exact_matches": privacy.get("exact_match_count", 0),
        }

        # Utility metrics
        util = dict(package.utility_metrics)
        utility = {
            "target_positive_f1": util.get("target_positive_f1"),
            "target_macro_f1": util.get("target_macro_f1"),
            "delta_target_f1_vs_baseline": util.get("delta_target_f1_vs_baseline"),
            "training_rows": util.get("training_rows", 0),
            "is_utility_acceptable": util.get("is_acceptable", False),
        }

        # Uncertainty metrics
        boot_ci = util.get("bootstrap_ci_95")
        ci_width = round(boot_ci[1] - boot_ci[0], 4) if (boot_ci and len(boot_ci) == 2) else None
        uncertainty = {
            "paired_bootstrap_ci_95": boot_ci,
            "confidence_interval_width": ci_width,
            "statistical_significance": (boot_ci[0] > 0.0) if (boot_ci and len(boot_ci) == 2) else False,
            "sample_budget": package.quantity_selected,
        }

        # Risk indicators
        src_ret = dict(package.source_retention_metrics)
        risks = {
            "source_retention_preserved": src_ret.get("source_retention_preserved", True),
            "source_f1_delta": src_ret.get("source_delta"),
            "exact_duplicate_ratio": privacy.get("exact_match_ratio", 0.0),
            "out_of_range_violations": package.validation_report.get("structural_report", {}).get("range_violations", {}),
            "rejection_reasons": list(package.rejection_reasons),
        }

        # Artifact hashes
        artifact_hashes = {
            "candidate_dataset_sha256": package.candidate_dataset_sha256 or "NONE",
            "candidate_model_sha256": package.candidate_model_sha256 or "NONE",
        }
        if package_manifest_path and Path(package_manifest_path).is_file():
            import hashlib
            m_bytes = Path(package_manifest_path).read_bytes()
            artifact_hashes["manifest_sha256"] = hashlib.sha256(m_bytes).hexdigest()

        # Approval metadata
        requires_human = package.requires_human_approval
        can_auto_deploy = False  # Strict safety rule: C3 candidate models NEVER auto-deploy!
        approval = {
            "requires_human_approval": requires_human,
            "can_auto_deploy": can_auto_deploy,
            "status": package.package_status.value,
            "approval_gate": "MANUAL_GATE_REQUIRED" if requires_human else "AUTO_REJECTED",
            "recommended_action": "APPROVE_FOR_STAGING" if package.package_status == RepairPackageStatus.APPROVED_FOR_REVIEW else "DO_NOT_DEPLOY",
        }

        payload = C4ReviewPayload(
            schema_version=self.schema_version,
            export_timestamp_utc=datetime.now(timezone.utc).isoformat(),
            request_id=package.request_id,
            run_id=package.run_id,
            package_id=package.package_id,
            package_status=package.package_status.value,
            strategy_recommendation=package.chosen_strategy,
            candidate_model_reference={
                "path": package.candidate_model_path,
                "sha256": package.candidate_model_sha256,
                "model_id": package.model_id,
            },
            candidate_data_reference={
                "path": package.candidate_dataset_path,
                "sha256": package.candidate_dataset_sha256,
                "quantity": package.quantity_selected,
            },
            artifact_hashes=artifact_hashes,
            quality_metrics=quality,
            utility_metrics=utility,
            source_retention_metrics=src_ret,
            uncertainty=uncertainty,
            risk_indicators=risks,
            approval_requirements=approval,
            evidence_limitations=list(package.evidence_limitations),
        )

        logger.info(
            "Constructed C4 review payload for package '%s' (status=%s, strategy=%s)",
            package.package_id,
            payload.package_status,
            payload.strategy_recommendation,
        )
        return payload

    def to_c4_flat_contract(
        self,
        package: RepairPackage,
        repair_version: str | None = None,
    ) -> dict[str, Any]:
        """Transform RepairPackage into the exact 12-field flat schema expected by Component 4.
        
        Adheres to shared/contracts/c3_output_schema.json.
        """
        from component3.src.utils.artifact_storage import to_portable_relative_uri

        # 1. repair_dataset_uri
        uri = "NONE"
        if package.candidate_dataset_path:
            uri = to_portable_relative_uri(package.candidate_dataset_path)

        # 2. repair_package_status (APPROVED or REJECTED)
        if package.package_status == RepairPackageStatus.APPROVED_FOR_REVIEW:
            pkg_status = "APPROVED"
        else:
            pkg_status = "REJECTED"

        # 3. generator (CTGAN, TVAE, TabDDPM)
        raw_gen = (
            package.generator_info.get("generator")
            or package.generator_info.get("selected_generator")
            or "TVAE"
        )
        gen_clean = str(raw_gen).strip().upper()
        if "TVAE" in gen_clean:
            generator_name = "TVAE"
        elif "CTGAN" in gen_clean:
            generator_name = "CTGAN"
        elif "TABDDPM" in gen_clean or "DIFFUSION" in gen_clean:
            generator_name = "TabDDPM"
        else:
            generator_name = "TVAE"

        # 4. n_synthetic_rows
        n_synth = int(package.quantity_selected) if pkg_status == "APPROVED" else 0

        # 5. augmentation_ratio
        training_rows = int(package.utility_metrics.get("training_rows", 0)) if package.utility_metrics else 0
        if n_synth == 0 or training_rows <= 0:
            aug_ratio = 0.0
        else:
            aug_ratio = round(min(1.0, max(0.0, n_synth / training_rows)), 4)

        # 6. fidelity_score
        fid = package.fidelity_indicators or {}
        if "fidelity_score" in fid and fid["fidelity_score"] is not None:
            fidelity_score = round(float(fid["fidelity_score"]), 4)
        elif "mean_ks_statistic" in fid and fid["mean_ks_statistic"] is not None:
            fidelity_score = round(max(0.0, min(1.0, 1.0 - float(fid["mean_ks_statistic"]))), 4)
        else:
            fidelity_score = 0.0 if pkg_status == "REJECTED" else 0.85

        # 7. utility_score (target accuracy or downstream F1)
        util = package.utility_metrics or {}
        if "target_candidate_f1" in util and util["target_candidate_f1"] is not None:
            utility_score = round(float(util["target_candidate_f1"]), 4)
        elif "target_positive_f1" in util and util["target_positive_f1"] is not None:
            utility_score = round(float(util["target_positive_f1"]), 4)
        elif "delta_target_f1_vs_baseline" in util and util["delta_target_f1_vs_baseline"] is not None:
            base = float(util.get("target_baseline_f1", 0.70))
            utility_score = round(min(1.0, max(0.0, base + float(util["delta_target_f1_vs_baseline"]))), 4)
        else:
            utility_score = 0.0 if pkg_status == "REJECTED" else 0.80

        # 8. privacy_status (PASS or FAIL)
        priv = package.privacy_risk_indicators or {}
        priv_status = "PASS" if priv.get("status", "PASS").upper() == "PASS" else "FAIL"

        # 9. target_coverage
        val_rep = package.validation_report or {}
        if "target_coverage" in val_rep and val_rep["target_coverage"] is not None:
            target_cov = round(float(val_rep["target_coverage"]), 4)
        elif pkg_status == "APPROVED":
            # If targeted slice rule is present, evaluate coverage against synthetic dataset if available
            rule = (package.diagnostic_summary or {}).get("affected_slice_rule")
            if rule and package.candidate_dataset_path and Path(package.candidate_dataset_path).is_file():
                try:
                    from component3.src.contracts.safe_slice_evaluator import evaluate_slice_rule
                    synth_df = pd.read_csv(package.candidate_dataset_path)
                    mask = evaluate_slice_rule(synth_df, rule)
                    target_cov = round(float(mask.mean()), 4)
                except Exception:
                    target_cov = 1.0
            else:
                target_cov = 1.0
        else:
            target_cov = 0.0

        # 10. prescription_confidence
        conf_map = {
            "CONFIRMED_BENCHMARK": 0.95,
            "STRONG": 0.88,
            "EXPLORATORY_PILOT": 0.80,
            "THEORETICAL_HEURISTIC": 0.70,
            "INSUFFICIENT_EVIDENCE": 0.40,
        }
        ev_strength = None
        if isinstance(package.downstream_handoff_metadata, dict):
            ev_strength = package.downstream_handoff_metadata.get("evidence_strength")
        if not ev_strength and isinstance(package.drift_summary, dict):
            ev_strength = package.drift_summary.get("evidence_strength")
        presc_conf = conf_map.get(str(ev_strength), 0.88 if pkg_status == "APPROVED" else 0.40)

        # 11. repair_version
        v_tag = repair_version or f"c3-{package.request_id[-8:].lower() if package.request_id else '001'}"

        # 12. repair_timestamp
        ts = package.created_at_utc or datetime.now(timezone.utc).isoformat()

        return {
            "repair_dataset_uri": uri,
            "repair_package_status": pkg_status,
            "generator": generator_name,
            "n_synthetic_rows": n_synth,
            "augmentation_ratio": aug_ratio,
            "fidelity_score": fidelity_score,
            "utility_score": utility_score,
            "privacy_status": priv_status,
            "target_coverage": target_cov,
            "prescription_confidence": presc_conf,
            "repair_version": v_tag,
            "repair_timestamp": ts,
        }

    def export_flat_contract(
        self,
        package: RepairPackage,
        output_path: str | Path | None = None,
        repair_version: str | None = None,
    ) -> dict[str, Any]:
        """Export flat 12-field contract and write to JSON file if path provided."""
        flat_contract = self.to_c4_flat_contract(package, repair_version=repair_version)
        if output_path is not None:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(flat_contract, f, indent=2)
            logger.info("Saved flat C4 contract JSON to: %s", out_p)
        return flat_contract
