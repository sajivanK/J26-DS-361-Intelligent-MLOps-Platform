"""Model training and artifact management for C4."""

from c4.models.champion import (
    ChampionConfig,
    ChampionArtifact,
    load_champion,
    train_and_save_champion,
    train_champion,
)

__all__ = [
    "ChampionConfig",
    "ChampionArtifact",
    "load_champion",
    "train_and_save_champion",
    "train_champion",
]