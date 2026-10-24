from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictContractModel(BaseModel):
    """Base class that rejects unknown contract fields."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class DetectorScores(StrictContractModel):
    ks_mean: float = Field(ge=0.0)
    ks_max: float = Field(ge=0.0)
    chi2_mean: float = Field(ge=0.0)
    chi2_max: float = Field(ge=0.0)
    psi_mean: float = Field(ge=0.0)
    psi_max: float = Field(ge=0.0)
    mmd: float = Field(ge=0.0)
    lsdd: float = Field(ge=0.0)
    clf_auc: float = Field(ge=0.0, le=1.0)


class PredictionStats(StrictContractModel):
    conf_mean: float = Field(ge=0.0, le=1.0)
    conf_std: float = Field(ge=0.0)
    conf_pct_lt_06: float = Field(ge=0.0, le=1.0)
    mean_pos_prob: float = Field(ge=0.0, le=1.0)
    pos_prob_std: float = Field(ge=0.0)
    pct_pos_prob_lt_06: float = Field(ge=0.0, le=1.0)


class C1Evidence(StrictContractModel):
    evidence_version: str = Field(min_length=1)
    engine_version: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    reference_dataset_uri: str = Field(min_length=1)
    current_dataset_uri: str = Field(min_length=1)

    batch_size: int = Field(ge=1)
    drift_detected: bool

    predicted_harm: float = Field(ge=-1.0, le=1.0)
    harm_interval: tuple[float, float]
    harm_metric: Literal["accuracy_drop", "auc_drop"]

    harm_severity: Literal[
        "none",
        "low",
        "moderate",
        "high",
        "critical",
    ]

    marginal_extent: Literal["none", "single", "multi"]
    interaction_suspected: float = Field(ge=0.0, le=1.0)

    affected_features: list[str]
    detector_scores: DetectorScores
    detectors_fired: int = Field(ge=0)
    detectors_fired_list: list[str]
    prediction_stats: PredictionStats
    warnings: list[str]
    runtime_sec: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_c1_evidence(self) -> Self:
        lower, upper = self.harm_interval

        if lower > upper:
            raise ValueError(
                "harm_interval lower bound cannot exceed upper bound"
            )

        if not lower <= self.predicted_harm <= upper:
            raise ValueError(
                "predicted_harm must fall inside harm_interval"
            )

        if self.detectors_fired != len(self.detectors_fired_list):
            raise ValueError(
                "detectors_fired must equal the number of entries "
                "in detectors_fired_list"
            )

        return self


class C2Diagnosis(StrictContractModel):
    root_cause_category: str = Field(min_length=1)
    root_cause_feature: str = Field(min_length=1)
    affected_slice_rule: str = Field(min_length=1)
    slice_support: float = Field(ge=0.0, le=1.0)
    diagnosis_stability: float = Field(ge=0.0, le=1.0)
    affected_features_ranked: list[str] = Field(min_length=1)
    diagnosis_version: str = Field(min_length=1)


class C3RepairPackage(StrictContractModel):
    repair_dataset_uri: str = Field(min_length=1)
    repair_package_status: Literal["APPROVED", "REJECTED"]
    generator: Literal["CTGAN", "TVAE", "TabDDPM", "NONE"]
    n_synthetic_rows: int = Field(ge=0)
    augmentation_ratio: float = Field(ge=0.0, le=1.0)
    fidelity_score: float = Field(ge=0.0, le=1.0)

    # The guide calculates utility as candidate accuracy minus champion
    # accuracy, so this margin may be negative.
    utility_score: float = Field(ge=-1.0, le=1.0)

    privacy_status: Literal["PASS", "FAIL"]
    target_coverage: float = Field(ge=0.0, le=1.0)
    prescription_confidence: float = Field(ge=0.0, le=1.0)
    repair_version: str = Field(min_length=1)


class UpstreamInput(StrictContractModel):
    c1_evidence: C1Evidence
    c2_diagnosis: C2Diagnosis
    c3_repair_package: C3RepairPackage