"""Repair acceptance and rejection policy engine for Component 3."""

from __future__ import annotations

from typing import Any

from component3.src.validation.validation_models import (
    AcceptanceDecision,
    RepairPackageDecision,
    ValidationStageResult,
    ValidationStatus,
)


class AcceptancePolicy:
    """Decision engine evaluating multi-stage validation outcomes for repair packages."""

    def __init__(
        self,
        mode: str = "production",
        strict_privacy_gate: bool = True,
        strict_fidelity_gate: bool = True,
    ) -> None:
        """Initialize AcceptancePolicy.

        Parameters
        ----------
        mode : str
            Operational mode: 'production', 'research', or 'exploratory'.
        strict_privacy_gate : bool
            Whether privacy failure triggers hard rejection.
        strict_fidelity_gate : bool
            Whether fidelity failure triggers rejection.
        """
        if mode not in ("production", "research", "exploratory"):
            raise ValueError(f"Invalid mode '{mode}'. Expected: 'production', 'research', 'exploratory'")
        self.mode = mode
        self.strict_privacy_gate = strict_privacy_gate
        self.strict_fidelity_gate = strict_fidelity_gate

    def decide(
        self,
        structural: ValidationStageResult,
        fidelity: ValidationStageResult | None = None,
        privacy: ValidationStageResult | None = None,
        utility: ValidationStageResult | None = None,
    ) -> RepairPackageDecision:
        """Apply acceptance policy to validation stage results."""
        return evaluate_repair_acceptance(
            structural=structural,
            fidelity=fidelity,
            privacy=privacy,
            utility=utility,
            mode=self.mode,
            strict_privacy_gate=self.strict_privacy_gate,
            strict_fidelity_gate=self.strict_fidelity_gate,
        )


def evaluate_repair_acceptance(
    structural: ValidationStageResult,
    fidelity: ValidationStageResult | None = None,
    privacy: ValidationStageResult | None = None,
    utility: ValidationStageResult | None = None,
    mode: str = "production",
    strict_privacy_gate: bool = True,
    strict_fidelity_gate: bool = True,
) -> RepairPackageDecision:
    """Evaluate multi-stage validation results and determine final acceptance decision.

    Parameters
    ----------
    structural : ValidationStageResult
        Structural validation result (mandatory).
    fidelity : ValidationStageResult | None
        Fidelity validation result.
    privacy : ValidationStageResult | None
        Privacy validation result.
    utility : ValidationStageResult | None
        Offline utility evaluation result.
    mode : str
        'production', 'research', or 'exploratory'.
    strict_privacy_gate : bool
        Whether privacy failure enforces hard REJECT.
    strict_fidelity_gate : bool
        Whether fidelity failure enforces hard REJECT.

    Returns
    -------
    RepairPackageDecision
        Decision outcome with ACCEPT, REVIEW, or REJECT, package status, and reasons.
    """
    if not isinstance(structural, ValidationStageResult):
        return RepairPackageDecision(
            decision=AcceptanceDecision.REJECT,
            stage_results={},
            reasons=("Structural validation result is missing or malformed",),
            supervised_retraining_approved=False,
            package_status="REJECTED",
        )

    for name, stage in [("fidelity", fidelity), ("privacy", privacy), ("utility", utility)]:
        if stage is not None and not isinstance(stage, ValidationStageResult):
            return RepairPackageDecision(
                decision=AcceptanceDecision.REJECT,
                stage_results={"structural": structural},
                reasons=(f"{name.capitalize()} validation result is malformed",),
                supervised_retraining_approved=False,
                package_status="REJECTED",
            )

    stage_results: dict[str, ValidationStageResult] = {"structural": structural}
    if fidelity is not None:
        stage_results["fidelity"] = fidelity
    if privacy is not None:
        stage_results["privacy"] = privacy
    if utility is not None:
        stage_results["utility"] = utility

    reasons: list[str] = []

    # 1. Structural Gate (Hard Gate)
    if structural.status == ValidationStatus.FAIL:
        reasons.append(f"Structural validation failed: {list(structural.reasons)}")
        return RepairPackageDecision(
            decision=AcceptanceDecision.REJECT,
            stage_results=stage_results,
            reasons=tuple(reasons),
            supervised_retraining_approved=False,
            package_status="REJECTED",
        )

    # 2. Privacy Gate (Hard Gate per C4 contract)
    if privacy is not None:
        if privacy.status == ValidationStatus.FAIL and strict_privacy_gate:
            reasons.append(f"Privacy validation failed (C4 hard gate): {list(privacy.reasons)}")
            return RepairPackageDecision(
                decision=AcceptanceDecision.REJECT,
                stage_results=stage_results,
                reasons=tuple(reasons),
                supervised_retraining_approved=False,
                package_status="REJECTED",
            )
    else:
        reasons.append("Privacy validation has not been performed")

    # 3. Fidelity Gate
    if fidelity is not None:
        if fidelity.status == ValidationStatus.FAIL:
            if strict_fidelity_gate and mode == "production":
                reasons.append(f"Fidelity validation failed: {list(fidelity.reasons)}")
                return RepairPackageDecision(
                    decision=AcceptanceDecision.REJECT,
                    stage_results=stage_results,
                    reasons=tuple(reasons),
                    supervised_retraining_approved=False,
                    package_status="REJECTED",
                )
            else:
                reasons.append(f"Fidelity validation indicated divergence: {list(fidelity.reasons)}")
    else:
        reasons.append("Fidelity validation has not been performed")

    # 4. Utility Gate
    supervised_approved = False
    if utility is not None:
        if utility.status == ValidationStatus.FAIL:
            reasons.append(f"Utility evaluation failed (negative transfer or degradation): {list(utility.reasons)}")
            if mode == "production":
                return RepairPackageDecision(
                    decision=AcceptanceDecision.REJECT,
                    stage_results=stage_results,
                    reasons=tuple(reasons),
                    supervised_retraining_approved=False,
                    package_status="REJECTED",
                )
        elif utility.status == ValidationStatus.PASS:
            supervised_approved = True
        elif utility.status == ValidationStatus.NOT_EVALUATED:
            reasons.append("Utility evaluation was NOT_EVALUATED: supervised retraining is not authorized")
    else:
        reasons.append("Utility evaluation has not been performed")

    # 5. Final Decision Synthesis
    # Check if any stage is in REVIEW or NOT_EVALUATED
    has_unevaluated = any(res.status in (ValidationStatus.NOT_EVALUATED, ValidationStatus.REVIEW) for res in stage_results.values())
    missing_stages = fidelity is None or privacy is None or utility is None

    if has_unevaluated or missing_stages or reasons:
        # In exploratory or research mode, allow REVIEW if structural and privacy passed
        decision = AcceptanceDecision.REVIEW
        package_status = "REJECTED"  # C4 only accepts APPROVED
    else:
        # All mandatory stages PASSED cleanly
        decision = AcceptanceDecision.ACCEPT
        package_status = "APPROVED"

    return RepairPackageDecision(
        decision=decision,
        stage_results=stage_results,
        reasons=tuple(reasons),
        supervised_retraining_approved=supervised_approved,
        package_status=package_status,
    )
