"""Model training and artifact management for C4."""

from c4.models.champion import (
    ChampionConfig,
    ChampionArtifact,
    load_champion,
    train_and_save_champion,
    train_champion,
)
from c4.models.candidate import (
    CandidateArtifact,
    CandidateTrainingResult,
    load_candidate,
    train_and_save_candidate,
    train_candidate,
)

from c4.models.evaluation import evaluate_pre_release

__all__ = [
    "ChampionConfig",
    "ChampionArtifact",
    "load_champion",
    "train_and_save_champion",
    "train_champion",
    "CandidateTrainingResult",
    "train_candidate",
    "CandidateArtifact",
    "load_candidate",
    "train_and_save_candidate",
    "evaluate_pre_release",
]