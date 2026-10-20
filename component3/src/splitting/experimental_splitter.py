"""Experimental Four-Way Partition Builder and Leakage Prevention Framework.

Designed for post-drift adaptation research within the permitted development scope.
Strictly adheres to:
1. Row conservation: Exactly partitions the canonical training dataset (N=4,931 for Telco).
2. Zero data leakage: Pairwise index disjointness and group-aware duplicate protection.
3. Nested adaptation budgeting: B_50 subset of B_100 subset of B_200.
4. Formal label provenance tracking: Decoupling observational ground truth from simulated concept truth.
5. Inviolability of canonical validation (N=1,056) and final test (N=1,056) partitions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from component3.src.datasets.dataset_registry import DatasetMetadata, get_dataset_registry
from component3.src.splitting.splitter import (
    compute_dataset_fingerprint,
    extract_feature_groups,
)


@dataclass
class LabelProvenanceMetadata:
    """Explicit governance contract capturing label provenance across source and target domains."""
    scenario_id: str
    dataset_id: str
    source_label_provenance: str
    target_label_provenance: str
    is_target_observational: bool
    is_adaptation_eligible: bool
    is_evaluation_eligible: bool
    transformation_rule_version: str
    seed: int
    split_fingerprint: str
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def validate(self) -> None:
        """Enforce scientific validity constraints on label provenance."""
        # Simulated concept labels can never be classified as observational ground truth
        if self.target_label_provenance == "synthetic_concept_truth" and self.is_target_observational:
            raise ValueError(
                "Scientific Defect: target_label_provenance is 'synthetic_concept_truth' "
                "but is_target_observational is set to True. Simulated labels are not observational."
            )
        # Stress test uncertain labels are never eligible for supervised adaptation
        if self.target_label_provenance == "stress_test_uncertain" and self.is_adaptation_eligible:
            raise ValueError(
                "Scientific Defect: 'stress_test_uncertain' labels cannot be marked adaptation eligible. "
                "Feature shifts with unverified labels force models to memorize noise."
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata to dictionary."""
        return asdict(self)


@dataclass
class ExperimentalFourWaySplits:
    """Container holding the four disjoint experimental partitions and nested adaptation budgets."""
    source_train: pd.DataFrame
    source_eval: pd.DataFrame
    target_adapt_pool: pd.DataFrame
    target_eval: pd.DataFrame
    nested_adapt_budgets: dict[int, pd.DataFrame] = field(default_factory=dict)


@dataclass
class ExperimentalSplitManifest:
    """Audit manifest capturing exact partition sizes, distributions, and leakage verification."""
    dataset_id: str
    seed: int
    source_dataset_fingerprint: str
    canonical_train_size: int
    realized_counts: dict[str, int]
    class_distributions: dict[str, dict[str, float]]
    slice_supports: dict[str, dict[str, int]]
    duplicate_group_audit: dict[str, Any]
    nested_budget_counts: dict[str, dict[str, Any]]
    indices: dict[str, list[int]]
    provenance_contract: dict[str, Any]
    leakage_verified: bool = False
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize manifest to dictionary."""
        return asdict(self)

    def save(self, filepath: Path | str) -> None:
        """Write audit manifest to JSON file."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Path | str) -> ExperimentalSplitManifest:
        """Load audit manifest from JSON file."""
        p = Path(filepath)
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


def four_way_stratified_group_partition(
    group_ids: np.ndarray,
    n_groups: int,
    y: np.ndarray | None,
    ratios: tuple[float, float, float, float] = (0.70, 0.10, 0.10, 0.10),
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Greedy deficit-balancing group partition into 4 mutually exclusive sets.

    Guarantees:
    1. Every duplicate-feature group is assigned exclusively to ONE partition (0% intra-group overlap).
    2. Realized row counts closely match ratios (70%, 10%, 10%, 10%).
    3. Target class balance is preserved across partitions.
    """
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios)}")

    n_samples = len(group_ids)
    r_vec = np.array(ratios, dtype=np.float64)

    if y is not None:
        classes, y_inv = np.unique(y, return_inverse=True)
        n_classes = len(classes)
    else:
        classes = np.array([0])
        y_inv = np.zeros(n_samples, dtype=np.int64)
        n_classes = 1

    # Count group class frequencies
    group_class_counts = np.zeros((n_groups, n_classes), dtype=np.int64)
    for g_id, c_idx in zip(group_ids, y_inv):
        group_class_counts[g_id, c_idx] += 1
    group_sizes = group_class_counts.sum(axis=1)

    total_class_counts = group_class_counts.sum(axis=0)
    target_counts = r_vec[:, np.newaxis] * total_class_counts[np.newaxis, :]
    target_sizes = r_vec * n_samples

    inv_target_counts = 1.0 / np.maximum(target_counts, 1e-9)
    inv_target_sizes = 1.0 / np.maximum(target_sizes, 1e-9)

    # Deterministic sorting: largest groups first to minimize quantization deficit
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_groups)
    sorted_order = np.argsort(-group_sizes[perm], kind="stable")
    ordered_groups = perm[sorted_order]

    alloc_class_counts = np.zeros((4, n_classes), dtype=np.float64)
    alloc_sizes = np.zeros(4, dtype=np.float64)
    group_partition = np.empty(n_groups, dtype=np.int8)

    for g in ordered_groups:
        g_counts = group_class_counts[g]
        g_size = group_sizes[g]

        best_p = 0
        best_score = float("inf")

        for p in range(4):
            alloc_class_counts[p] += g_counts
            alloc_sizes[p] += g_size

            r_classes = alloc_class_counts * inv_target_counts
            r_sizes = alloc_sizes * inv_target_sizes

            # Minimizes variance of relative completion across partitions
            score = np.std(r_classes, axis=0).mean() + np.std(r_sizes)

            alloc_class_counts[p] -= g_counts
            alloc_sizes[p] -= g_size

            if score < best_score:
                best_score = score
                best_p = p

        group_partition[g] = best_p
        alloc_class_counts[best_p] += g_counts
        alloc_sizes[best_p] += g_size

    row_partitions = group_partition[group_ids]
    p0 = np.where(row_partitions == 0)[0]
    p1 = np.where(row_partitions == 1)[0]
    p2 = np.where(row_partitions == 2)[0]
    p3 = np.where(row_partitions == 3)[0]

    return p0, p1, p2, p3


class ExperimentalPartitionBuilder:
    """Deterministic, group-aware experimental partition builder for post-drift adaptation."""

    @staticmethod
    def build_splits(
        canonical_train_df: pd.DataFrame,
        metadata: DatasetMetadata | None = None,
        feature_columns: list[str] | None = None,
        target_column: str | None = None,
        ratios: tuple[float, float, float, float] = (0.70, 0.10, 0.10, 0.10),
        adaptation_budgets: tuple[int, ...] = (50, 100, 200),
        seed: int = 42,
        scenario_id: str = "TELCO_CONCEPT",
        target_label_provenance: str = "synthetic_concept_truth",
        is_target_observational: bool = False,
    ) -> tuple[ExperimentalFourWaySplits, ExperimentalSplitManifest]:
        """Partition canonical training dataset into 4 disjoint, group-aware sets with nested budgets."""
        df_indexed = canonical_train_df.reset_index(drop=True)
        n_rows = len(df_indexed)

        if n_rows == 0:
            raise ValueError("Cannot partition an empty DataFrame.")

        dataset_id = metadata.dataset_id if metadata else "telco_churn"
        target_col = target_column or (metadata.target_column if metadata else "Churn")

        if target_col not in df_indexed.columns:
            raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

        # Determine active feature columns
        if feature_columns is not None:
            active_features = list(feature_columns)
        elif metadata is not None:
            active_features = [c for c in metadata.feature_columns if c not in metadata.drop_columns]
        else:
            active_features = [c for c in df_indexed.columns if c != target_col]

        missing_reps = metadata.missing_representations if metadata else ["?", "", "NA", "null", "NaN", " "]

        # Extract duplicate feature groups
        group_ids, n_groups, n_dup_groups, rows_in_dups = extract_feature_groups(
            df_indexed,
            feature_columns=active_features,
            missing_representations=missing_reps,
        )

        y = df_indexed[target_col].to_numpy()

        # Run 4-way group-aware partition
        idx_p0, idx_p1, idx_p2, idx_p3 = four_way_stratified_group_partition(
            group_ids=group_ids,
            n_groups=n_groups,
            y=y,
            ratios=ratios,
            seed=seed,
        )

        source_train_df = df_indexed.iloc[idx_p0].copy().reset_index(drop=True)
        source_eval_df = df_indexed.iloc[idx_p1].copy().reset_index(drop=True)
        target_adapt_pool_df = df_indexed.iloc[idx_p2].copy().reset_index(drop=True)
        target_eval_df = df_indexed.iloc[idx_p3].copy().reset_index(drop=True)

        # Build nested adaptation budgets deterministically
        nested_budgets: dict[int, pd.DataFrame] = {}
        nested_budget_counts: dict[str, dict[str, Any]] = {}

        # Stratified sampling within adaptation pool to form nested subsets: B_50 subset B_100 subset B_200
        adapt_y = target_adapt_pool_df[target_col]
        pos_keys = {k for k, v in metadata.target_mapping.items() if v == 1} | {1, "1"} if metadata and metadata.target_mapping else {"Yes", 1, "1"}
        if metadata and metadata.dataset_id == "telco_churn":
            # Preserve exact frozen historical membership hashes for Batch 5G/5H/5M audits
            is_pos = (target_adapt_pool_df[target_col] == 1)
        else:
            is_pos = adapt_y.isin(pos_keys) | (adapt_y == 1)

        pos_indices = target_adapt_pool_df[is_pos].index.to_numpy()
        neg_indices = target_adapt_pool_df[~is_pos].index.to_numpy()

        rng_sub = np.random.default_rng(seed)
        perm_pos = rng_sub.permutation(pos_indices)
        perm_neg = rng_sub.permutation(neg_indices)

        pos_ratio = len(pos_indices) / max(len(target_adapt_pool_df), 1)

        sorted_budgets = sorted(adaptation_budgets)
        # Verify adaptation pool has enough records for the max budget
        max_budget = sorted_budgets[-1] if sorted_budgets else 0
        if max_budget > len(target_adapt_pool_df):
            raise ValueError(
                f"Requested maximum adaptation budget {max_budget} exceeds available adaptation pool size {len(target_adapt_pool_df)}"
            )

        budget_indices_map: dict[int, np.ndarray] = {}
        for b_size in sorted_budgets:
            n_pos = int(round(b_size * pos_ratio))
            n_neg = b_size - n_pos

            sub_pos = perm_pos[:n_pos]
            sub_neg = perm_neg[:n_neg]
            b_idx = np.sort(np.concatenate([sub_pos, sub_neg]))
            budget_indices_map[b_size] = b_idx

            b_df = target_adapt_pool_df.iloc[b_idx].copy().reset_index(drop=True)
            nested_budgets[b_size] = b_df

            # Log budget composition
            pos_c = int(b_df[target_col].isin(pos_keys).sum() if hasattr(b_df[target_col], "isin") else (b_df[target_col] == 1).sum())
            neg_c = len(b_df) - pos_c
            nested_budget_counts[f"K_{b_size}"] = {
                "budget_size": b_size,
                "positive_count": pos_c,
                "negative_count": neg_c,
                "positive_rate": round(pos_c / max(b_size, 1), 4),
            }

        # Verify nesting: B_50 subset B_100 subset B_200
        for i in range(len(sorted_budgets) - 1):
            smaller_b = sorted_budgets[i]
            larger_b = sorted_budgets[i + 1]
            smaller_set = set(budget_indices_map[smaller_b])
            larger_set = set(budget_indices_map[larger_b])
            if not smaller_set.issubset(larger_set):
                raise AssertionError(f"Nesting defect: Budget K={smaller_b} is not a strict subset of K={larger_b}")

        splits = ExperimentalFourWaySplits(
            source_train=source_train_df,
            source_eval=source_eval_df,
            target_adapt_pool=target_adapt_pool_df,
            target_eval=target_eval_df,
            nested_adapt_budgets=nested_budgets,
        )

        # Audit class distributions
        class_distributions: dict[str, dict[str, float]] = {}
        realized_counts: dict[str, int] = {
            "source_train": len(source_train_df),
            "source_eval": len(source_eval_df),
            "target_adapt_pool": len(target_adapt_pool_df),
            "target_eval": len(target_eval_df),
        }

        for name, part_df in [
            ("source_train", source_train_df),
            ("source_eval", source_eval_df),
            ("target_adapt_pool", target_adapt_pool_df),
            ("target_eval", target_eval_df),
        ]:
            val_counts = part_df[target_col].value_counts(normalize=True).to_dict()
            class_distributions[name] = {str(k): round(float(v), 4) for k, v in val_counts.items()}

        # Audit slice supports
        slice_supports: dict[str, dict[str, int]] = {}
        for name, part_df in [
            ("source_train", source_train_df),
            ("source_eval", source_eval_df),
            ("target_adapt_pool", target_adapt_pool_df),
            ("target_eval", target_eval_df),
        ]:
            m2m_c = int((part_df["Contract"] == "Month-to-month").sum()) if "Contract" in part_df.columns else 0
            ten29_c = (
                int((pd.to_numeric(part_df["tenure"], errors="coerce") > 29.0).sum())
                if "tenure" in part_df.columns
                else 0
            )
            slice_supports[name] = {
                "month_to_month_count": m2m_c,
                "tenure_gt_29_count": ten29_c,
            }

        # Build provenance contract
        fingerprint = compute_dataset_fingerprint(df_indexed)
        provenance = LabelProvenanceMetadata(
            scenario_id=scenario_id,
            dataset_id=dataset_id,
            source_label_provenance="verified_observational_ground_truth",
            target_label_provenance=target_label_provenance,
            is_target_observational=is_target_observational,
            is_adaptation_eligible=True,
            is_evaluation_eligible=True,
            transformation_rule_version="1.0.0",
            seed=seed,
            split_fingerprint=fingerprint,
        )
        provenance.validate()

        duplicate_audit = {
            "total_groups": n_groups,
            "duplicate_groups": n_dup_groups,
            "rows_in_duplicate_groups": rows_in_dups,
            "cross_partition_overlap_groups": 0,
        }

        manifest = ExperimentalSplitManifest(
            dataset_id=dataset_id,
            seed=seed,
            source_dataset_fingerprint=fingerprint,
            canonical_train_size=n_rows,
            realized_counts=realized_counts,
            class_distributions=class_distributions,
            slice_supports=slice_supports,
            duplicate_group_audit=duplicate_audit,
            nested_budget_counts=nested_budget_counts,
            indices={
                "source_train": [int(i) for i in idx_p0],
                "source_eval": [int(i) for i in idx_p1],
                "target_adapt_pool": [int(i) for i in idx_p2],
                "target_eval": [int(i) for i in idx_p3],
            },
            provenance_contract=provenance.to_dict(),
            leakage_verified=False,
        )

        # Run automated leakage assertions
        assert_no_experimental_leakage(manifest, splits, active_features, missing_reps)
        manifest.leakage_verified = True

        return splits, manifest


def assert_no_experimental_leakage(
    manifest: ExperimentalSplitManifest,
    splits: ExperimentalFourWaySplits,
    feature_columns: list[str],
    missing_representations: list[Any] | None = None,
) -> dict[str, Any]:
    """Verify that indices and identical feature groups across the 4 partitions are strictly disjoint."""
    # 1. Index-level disjointness and row conservation
    idx_p0 = set(manifest.indices["source_train"])
    idx_p1 = set(manifest.indices["source_eval"])
    idx_p2 = set(manifest.indices["target_adapt_pool"])
    idx_p3 = set(manifest.indices["target_eval"])

    # Pairwise index intersection check
    pairs = [
        ("source_train", "source_eval", idx_p0 & idx_p1),
        ("source_train", "target_adapt_pool", idx_p0 & idx_p2),
        ("source_train", "target_eval", idx_p0 & idx_p3),
        ("source_eval", "target_adapt_pool", idx_p1 & idx_p2),
        ("source_eval", "target_eval", idx_p1 & idx_p3),
        ("target_adapt_pool", "target_eval", idx_p2 & idx_p3),
    ]

    for name_a, name_b, intersection in pairs:
        if len(intersection) > 0:
            raise AssertionError(
                f"Index leakage detected between {name_a} and {name_b}: {len(intersection)} overlapping indices!"
            )

    total_realized = len(idx_p0) + len(idx_p1) + len(idx_p2) + len(idx_p3)
    if total_realized != manifest.canonical_train_size:
        raise AssertionError(
            f"Row conservation violation: total rows = {manifest.canonical_train_size}, "
            f"but partitioned rows = {total_realized}."
        )

    # 2. Duplicate feature-vector disjointness check
    missing_tokens = list(missing_representations or ["?", "", "NA", "null", "NaN", " "])

    def _get_feature_tuples(df_part: pd.DataFrame) -> set[tuple]:
        df_clean = df_part[feature_columns].copy()
        for col in feature_columns:
            if df_clean[col].dtype == object or pd.api.types.is_string_dtype(df_clean[col]):
                s = df_clean[col].astype(str).str.strip()
                for tok in missing_tokens:
                    s = s.replace(str(tok).strip(), "__MISSING__")
                df_clean[col] = s.replace(["nan", "None", "<NA>"], "__MISSING__")
            else:
                df_clean[col] = df_clean[col].fillna("__MISSING__")
        return set(df_clean.itertuples(index=False, name=None))

    t_p0 = _get_feature_tuples(splits.source_train)
    t_p1 = _get_feature_tuples(splits.source_eval)
    t_p2 = _get_feature_tuples(splits.target_adapt_pool)
    t_p3 = _get_feature_tuples(splits.target_eval)

    group_pairs = [
        ("source_train", "source_eval", t_p0 & t_p1),
        ("source_train", "target_adapt_pool", t_p0 & t_p2),
        ("source_train", "target_eval", t_p0 & t_p3),
        ("source_eval", "target_adapt_pool", t_p1 & t_p2),
        ("source_eval", "target_eval", t_p1 & t_p3),
        ("target_adapt_pool", "target_eval", t_p2 & t_p3),
    ]

    for name_a, name_b, overlap in group_pairs:
        if len(overlap) > 0:
            raise AssertionError(
                f"Duplicate feature signature leakage detected between {name_a} and {name_b}: "
                f"{len(overlap)} identical feature vectors cross partition boundaries!"
            )

    total_idx_overlap = sum(len(intersection) for _, _, intersection in pairs)
    total_group_overlap = sum(len(overlap) for _, _, overlap in group_pairs)

    return {
        "status": "PASS",
        "total_rows_partitioned": total_realized,
        "cross_partition_index_overlaps": total_idx_overlap,
        "cross_partition_feature_overlaps": total_group_overlap,
    }


def audit_experimental_partitions(
    canonical_train_df: pd.DataFrame,
    manifest: ExperimentalSplitManifest,
    feature_columns: list[str],
    missing_representations: list[Any] | None = None,
    save_path: Path | str | None = None,
) -> dict[str, Any]:
    """Execute dynamic, non-hardcoded leakage audit calculating all 6 pairwise index and group intersections.

    Reconstructs partition membership, computes exact set intersections for row indices and feature groups,
    verifies row conservation and fingerprint, and raises AssertionError if any prohibited overlap is nonzero.
    """
    df_clean = canonical_train_df.reset_index(drop=True)

    # 1. Dataset fingerprint verification
    current_fp = compute_dataset_fingerprint(df_clean)
    if current_fp != manifest.source_dataset_fingerprint:
        raise ValueError(
            f"Dataset fingerprint mismatch! Expected {manifest.source_dataset_fingerprint}, "
            f"got {current_fp}."
        )

    # 2. Extract feature-signature groups across entire canonical training population
    missing_tokens = list(missing_representations or ["?", "", "NA", "null", "NaN", " "])
    group_ids, n_groups, n_dup_groups, rows_in_dups = extract_feature_groups(
        df_clean,
        feature_columns=feature_columns,
        missing_representations=missing_tokens,
    )

    indices = manifest.indices
    part_names = ["source_train", "source_eval", "target_adapt_pool", "target_eval"]

    # 3. Calculate all 6 pairwise index and group overlaps dynamically
    idx_sets = {name: set(indices[name]) for name in part_names}
    group_sets = {name: set(group_ids[indices[name]]) for name in part_names}

    pairwise_idx: dict[str, int] = {}
    pairwise_group: dict[str, int] = {}

    for i in range(len(part_names)):
        for j in range(i + 1, len(part_names)):
            p_a, p_b = part_names[i], part_names[j]
            pair_key = f"{p_a}_vs_{p_b}"

            idx_overlap = len(idx_sets[p_a].intersection(idx_sets[p_b]))
            group_overlap = len(group_sets[p_a].intersection(group_sets[p_b]))

            pairwise_idx[pair_key] = idx_overlap
            pairwise_group[pair_key] = group_overlap

    total_idx_overlap = sum(pairwise_idx.values())
    total_group_overlap = sum(pairwise_group.values())

    # 4. Fail if any prohibited overlap is nonzero
    if total_idx_overlap > 0:
        raise AssertionError(
            f"Pairwise index leakage detected across experimental partitions: {pairwise_idx}"
        )
    if total_group_overlap > 0:
        raise AssertionError(
            f"Pairwise feature-group leakage detected across experimental partitions: {pairwise_group}"
        )

    # 5. Verify row conservation
    total_rows_partitioned = sum(len(indices[name]) for name in part_names)
    row_conservation_verified = total_rows_partitioned == len(df_clean) == manifest.canonical_train_size
    if not row_conservation_verified:
        raise AssertionError(
            f"Row conservation violation: canonical rows={len(df_clean)}, "
            f"manifest canonical_train_size={manifest.canonical_train_size}, "
            f"partitioned rows sum={total_rows_partitioned}"
        )

    audit_result = {
        "dataset_id": manifest.dataset_id,
        "source_dataset_fingerprint": manifest.source_dataset_fingerprint,
        "canonical_train_size": manifest.canonical_train_size,
        "total_unique_feature_groups": int(n_groups),
        "duplicate_feature_groups": int(n_dup_groups),
        "rows_in_duplicate_groups": int(rows_in_dups),
        "pairwise_cross_partition_index_overlaps": pairwise_idx,
        "pairwise_cross_partition_group_overlaps": pairwise_group,
        "total_cross_partition_index_overlaps": total_idx_overlap,
        "total_cross_partition_group_overlaps": total_group_overlap,
        "row_conservation_verified": row_conservation_verified,
        "fingerprint_verified": True,
        "leakage_audit_status": "PASS",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(audit_result, f, indent=2)

    return audit_result

