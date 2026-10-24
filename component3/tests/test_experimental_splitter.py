"""Automated Unit Tests for Experimental Four-Way Splitter and Leakage Prevention.

Validates:
1. Deterministic reproducibility across independent runs.
2. Exact row conservation (sum of partitions == canonical train rows).
3. Pairwise index disjointness across all 4 partitions.
4. Duplicate-feature group isolation (0 cross-partition duplicate leakage).
5. Strict nesting of adaptation budgets (B_50 subset of B_100 subset of B_200).
6. Class balance and drift-affected slice support reporting.
7. Label provenance validation and rejection of invalid / contradictory tags.
8. Dataset fingerprinting sensitivity.
9. Zero dependency on validation or final test partitions.
10. Inviolability of canonical split manifests (no mutation).
11. Negative control tests verifying that artificial index and group leakage triggers hard failure.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from component3.src.datasets.dataset_registry import get_dataset_registry
from component3.src.splitting.experimental_splitter import (
    ExperimentalFourWaySplits,
    ExperimentalPartitionBuilder,
    ExperimentalSplitManifest,
    LabelProvenanceMetadata,
    assert_no_experimental_leakage,
    audit_experimental_partitions,
    four_way_stratified_group_partition,
)


@pytest.fixture(scope="module")
def telco_canonical_train_data():
    """Load authentic canonical Telco training partition without touching test data."""
    reg = get_dataset_registry()
    df = reg.load_dataset("telco_churn")
    meta = reg.get("telco_churn")

    manifest_path = Path("component3/results/split_manifests/telco_churn_split_manifest.json")
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_indices = manifest["train_indices"]
    train_df = df.iloc[train_indices].copy().reset_index(drop=True)
    return train_df, meta, manifest_path


def test_experimental_splitter_deterministic_reproducibility(telco_canonical_train_data):
    """Verify that identical random seed produces bitwise-identical partitions and manifests."""
    train_df, meta, _ = telco_canonical_train_data

    splits1, manifest1 = ExperimentalPartitionBuilder.build_splits(
        canonical_train_df=train_df,
        metadata=meta,
        seed=42,
    )
    splits2, manifest2 = ExperimentalPartitionBuilder.build_splits(
        canonical_train_df=train_df,
        metadata=meta,
        seed=42,
    )

    assert manifest1.realized_counts == manifest2.realized_counts
    assert manifest1.indices == manifest2.indices
    pd.testing.assert_frame_equal(splits1.source_train, splits2.source_train)
    pd.testing.assert_frame_equal(splits1.target_eval, splits2.target_eval)


def test_experimental_splitter_row_conservation(telco_canonical_train_data):
    """Verify that all canonical training rows are strictly accounted for with zero loss."""
    train_df, meta, _ = telco_canonical_train_data
    splits, manifest = ExperimentalPartitionBuilder.build_splits(
        canonical_train_df=train_df,
        metadata=meta,
        seed=42,
    )

    total_realized = (
        len(splits.source_train)
        + len(splits.source_eval)
        + len(splits.target_adapt_pool)
        + len(splits.target_eval)
    )
    assert total_realized == len(train_df) == 4931
    assert total_realized == manifest.canonical_train_size
    assert sum(manifest.realized_counts.values()) == 4931


def test_experimental_splitter_pairwise_disjointness(telco_canonical_train_data):
    """Verify that no two partitions share common row indices."""
    train_df, meta, _ = telco_canonical_train_data
    _, manifest = ExperimentalPartitionBuilder.build_splits(
        canonical_train_df=train_df,
        metadata=meta,
        seed=42,
    )

    idx = {k: set(v) for k, v in manifest.indices.items()}
    names = list(idx.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = idx[names[i]].intersection(idx[names[j]])
            assert len(inter) == 0, f"Leakage detected between {names[i]} and {names[j]}: {len(inter)} shared rows"


def test_experimental_splitter_group_separation(telco_canonical_train_data):
    """Verify that duplicate-feature groups never cross partition boundaries."""
    train_df, meta, _ = telco_canonical_train_data
    splits, manifest = ExperimentalPartitionBuilder.build_splits(
        canonical_train_df=train_df,
        metadata=meta,
        seed=42,
    )

    assert manifest.leakage_verified is True
    assert manifest.duplicate_group_audit["duplicate_groups"] == 23
    assert manifest.duplicate_group_audit["rows_in_duplicate_groups"] == 51

    # Verify directly via assert_no_experimental_leakage
    feature_cols = [c for c in meta.feature_columns if c not in meta.drop_columns]
    audit_res = assert_no_experimental_leakage(manifest, splits, feature_cols)
    assert audit_res["status"] == "PASS"
    assert audit_res["cross_partition_feature_overlaps"] == 0


def test_experimental_splitter_nested_adaptation_budgets(telco_canonical_train_data):
    """Verify that adaptation budgets are strictly nested: B_50 subset of B_100 subset of B_200."""
    train_df, meta, _ = telco_canonical_train_data
    splits, manifest = ExperimentalPartitionBuilder.build_splits(
        canonical_train_df=train_df,
        metadata=meta,
        adaptation_budgets=(50, 100, 200),
        seed=42,
    )

    b50 = splits.nested_adapt_budgets[50]
    b100 = splits.nested_adapt_budgets[100]
    b200 = splits.nested_adapt_budgets[200]

    assert len(b50) == 50
    assert len(b100) == 100
    assert len(b200) == 200

    # Test set nesting using customerID (unique identifier)
    id_b50 = set(b50["customerID"])
    id_b100 = set(b100["customerID"])
    id_b200 = set(b200["customerID"])
    id_pool = set(splits.target_adapt_pool["customerID"])

    assert id_b50.issubset(id_b100), "B_50 is not a subset of B_100"
    assert id_b100.issubset(id_b200), "B_100 is not a subset of B_200"
    assert id_b200.issubset(id_pool), "B_200 is not a subset of adaptation pool"


def test_experimental_splitter_class_and_slice_support(telco_canonical_train_data):
    """Verify that each partition maintains balanced class and slice support."""
    train_df, meta, _ = telco_canonical_train_data
    splits, manifest = ExperimentalPartitionBuilder.build_splits(
        canonical_train_df=train_df,
        metadata=meta,
        seed=42,
    )

    for part_name, counts in manifest.realized_counts.items():
        assert counts > 0
        # Positive class rate should be within [0.24, 0.29]
        churn_rate = manifest.class_distributions[part_name].get("Yes", 0.0)
        assert 0.24 <= churn_rate <= 0.29, f"Unbalanced churn rate in {part_name}: {churn_rate}"

        # Slice supports must be sufficient for evaluation
        m2m = manifest.slice_supports[part_name]["month_to_month_count"]
        ten29 = manifest.slice_supports[part_name]["tenure_gt_29_count"]
        assert m2m >= 200, f"Insufficient month-to-month support in {part_name}: {m2m}"
        assert ten29 >= 200, f"Insufficient tenure>29 support in {part_name}: {ten29}"


def test_experimental_splitter_invalid_provenance_rejection():
    """Verify that contradictory or unverified label provenance tags are rejected."""
    # 1. Reject simulated concept truth claiming to be observational
    prov_invalid1 = LabelProvenanceMetadata(
        scenario_id="TELCO_CONCEPT",
        dataset_id="telco_churn",
        source_label_provenance="verified_observational_ground_truth",
        target_label_provenance="synthetic_concept_truth",
        is_target_observational=True,  # Contradiction!
        is_adaptation_eligible=True,
        is_evaluation_eligible=True,
        transformation_rule_version="1.0.0",
        seed=42,
        split_fingerprint="abc",
    )
    with pytest.raises(ValueError, match="Simulated labels are not observational"):
        prov_invalid1.validate()

    # 2. Reject stress_test_uncertain claiming adaptation eligibility
    prov_invalid2 = LabelProvenanceMetadata(
        scenario_id="TELCO_NUM",
        dataset_id="telco_churn",
        source_label_provenance="verified_observational_ground_truth",
        target_label_provenance="stress_test_uncertain",
        is_target_observational=False,
        is_adaptation_eligible=True,  # Contradiction!
        is_evaluation_eligible=True,
        transformation_rule_version="1.0.0",
        seed=42,
        split_fingerprint="abc",
    )
    with pytest.raises(ValueError, match="stress_test_uncertain.*cannot be marked adaptation eligible"):
        prov_invalid2.validate()


def test_experimental_splitter_no_canonical_manifest_mutation(telco_canonical_train_data):
    """Verify that running the experimental partitioner does not modify canonical split manifests."""
    train_df, meta, manifest_path = telco_canonical_train_data

    with manifest_path.open("r", encoding="utf-8") as f:
        before_manifest = json.load(f)

    _ = ExperimentalPartitionBuilder.build_splits(
        canonical_train_df=train_df,
        metadata=meta,
        seed=42,
    )

    with manifest_path.open("r", encoding="utf-8") as f:
        after_manifest = json.load(f)

    assert before_manifest == after_manifest, "Canonical split manifest was unexpectedly mutated!"


def test_negative_control_leakage_detection():
    """Verify that artificial index and group leakage triggers immediate AssertionError."""
    # Mock DataFrame with duplicate feature rows
    mock_df = pd.DataFrame({
        "feat_a": [1, 1, 2, 3, 4, 5, 6, 7],
        "feat_b": ["x", "x", "y", "z", "w", "v", "u", "t"],
        "target": [0, 1, 0, 1, 0, 1, 0, 1],
    })

    # Case 1: Artificial index leakage
    mock_manifest = ExperimentalSplitManifest(
        dataset_id="mock",
        seed=42,
        source_dataset_fingerprint="123",
        canonical_train_size=8,
        realized_counts={"source_train": 3, "source_eval": 2, "target_adapt_pool": 2, "target_eval": 2},
        class_distributions={},
        slice_supports={},
        duplicate_group_audit={},
        nested_budget_counts={},
        indices={
            "source_train": [0, 1, 2],
            "source_eval": [2, 3],  # Index 2 leaked into source_train!
            "target_adapt_pool": [4, 5],
            "target_eval": [6, 7],
        },
        provenance_contract={},
    )
    mock_splits = ExperimentalFourWaySplits(
        source_train=mock_df.iloc[[0, 1, 2]],
        source_eval=mock_df.iloc[[2, 3]],
        target_adapt_pool=mock_df.iloc[[4, 5]],
        target_eval=mock_df.iloc[[6, 7]],
    )

    with pytest.raises(AssertionError, match="Index leakage detected between source_train and source_eval"):
        assert_no_experimental_leakage(mock_manifest, mock_splits, ["feat_a", "feat_b"])

    # Case 2: Artificial duplicate feature group leakage (indices disjoint, but rows 0 and 1 are identical features)
    mock_manifest_clean_idx = ExperimentalSplitManifest(
        dataset_id="mock",
        seed=42,
        source_dataset_fingerprint="123",
        canonical_train_size=8,
        realized_counts={"source_train": 2, "source_eval": 2, "target_adapt_pool": 2, "target_eval": 2},
        class_distributions={},
        slice_supports={},
        duplicate_group_audit={},
        nested_budget_counts={},
        indices={
            "source_train": [0, 2],  # row 0 has (1, 'x')
            "source_eval": [1, 3],   # row 1 ALSO has (1, 'x')!
            "target_adapt_pool": [4, 5],
            "target_eval": [6, 7],
        },
        provenance_contract={},
    )
    mock_splits_group_leak = ExperimentalFourWaySplits(
        source_train=mock_df.iloc[[0, 2]],
        source_eval=mock_df.iloc[[1, 3]],
        target_adapt_pool=mock_df.iloc[[4, 5]],
        target_eval=mock_df.iloc[[6, 7]],
    )

    with pytest.raises(AssertionError, match="Duplicate feature signature leakage detected"):
        assert_no_experimental_leakage(mock_manifest_clean_idx, mock_splits_group_leak, ["feat_a", "feat_b"])


def test_audit_experimental_partitions_telco_pass(telco_canonical_train_data):
    """Verify that audit_experimental_partitions calculates all 6 pairwise overlaps dynamically and passes on Telco."""
    train_df, meta, _ = telco_canonical_train_data
    _, manifest = ExperimentalPartitionBuilder.build_splits(
        canonical_train_df=train_df,
        metadata=meta,
        seed=42,
    )

    feature_cols = [c for c in meta.feature_columns if c not in meta.drop_columns]
    audit_res = audit_experimental_partitions(
        canonical_train_df=train_df,
        manifest=manifest,
        feature_columns=feature_cols,
    )

    # 1. Assert status and checks
    assert audit_res["leakage_audit_status"] == "PASS"
    assert audit_res["fingerprint_verified"] is True
    assert audit_res["row_conservation_verified"] is True
    assert audit_res["canonical_train_size"] == 4931

    # 2. Assert all 6 pairwise comparisons exist and are strictly zero
    expected_pairs = [
        "source_train_vs_source_eval",
        "source_train_vs_target_adapt_pool",
        "source_train_vs_target_eval",
        "source_eval_vs_target_adapt_pool",
        "source_eval_vs_target_eval",
        "target_adapt_pool_vs_target_eval",
    ]
    assert set(audit_res["pairwise_cross_partition_index_overlaps"].keys()) == set(expected_pairs)
    assert set(audit_res["pairwise_cross_partition_group_overlaps"].keys()) == set(expected_pairs)

    for pair in expected_pairs:
        assert audit_res["pairwise_cross_partition_index_overlaps"][pair] == 0
        assert audit_res["pairwise_cross_partition_group_overlaps"][pair] == 0

    assert audit_res["total_cross_partition_index_overlaps"] == 0
    assert audit_res["total_cross_partition_group_overlaps"] == 0


def test_audit_experimental_partitions_negative_controls():
    """Verify that audit_experimental_partitions fails on fingerprint mismatch, index overlap, group overlap, or row loss."""
    # Base mock dataset with 8 rows where rows 0 and 1 share identical features
    mock_df = pd.DataFrame({
        "feat_a": [1, 1, 2, 3, 4, 5, 6, 7],
        "feat_b": ["x", "x", "y", "z", "w", "v", "u", "t"],
        "target": [0, 1, 0, 1, 0, 1, 0, 1],
    })

    from component3.src.splitting.experimental_splitter import compute_dataset_fingerprint
    correct_fp = compute_dataset_fingerprint(mock_df)

    # Base valid manifest
    base_manifest = ExperimentalSplitManifest(
        dataset_id="mock",
        seed=42,
        source_dataset_fingerprint=correct_fp,
        canonical_train_size=8,
        realized_counts={"source_train": 2, "source_eval": 2, "target_adapt_pool": 2, "target_eval": 2},
        class_distributions={},
        slice_supports={},
        duplicate_group_audit={},
        nested_budget_counts={},
        indices={
            "source_train": [0, 1],  # Duplicate group {0, 1} isolated together in source_train
            "source_eval": [2, 3],
            "target_adapt_pool": [4, 5],
            "target_eval": [6, 7],
        },
        provenance_contract={},
    )

    # 1. Clean run passes
    clean_audit = audit_experimental_partitions(mock_df, base_manifest, ["feat_a", "feat_b"])
    assert clean_audit["leakage_audit_status"] == "PASS"

    # Negative Control 1: Fingerprint mismatch
    bad_fp_manifest = ExperimentalSplitManifest(
        dataset_id="mock",
        seed=42,
        source_dataset_fingerprint="corrupted_fp_12345",
        canonical_train_size=8,
        realized_counts=base_manifest.realized_counts,
        class_distributions={},
        slice_supports={},
        duplicate_group_audit={},
        nested_budget_counts={},
        indices=base_manifest.indices,
        provenance_contract={},
    )
    with pytest.raises(ValueError, match="Dataset fingerprint mismatch"):
        audit_experimental_partitions(mock_df, bad_fp_manifest, ["feat_a", "feat_b"])

    # Negative Control 2: Pairwise index overlap (row 2 in both source_train and source_eval)
    bad_idx_manifest = ExperimentalSplitManifest(
        dataset_id="mock",
        seed=42,
        source_dataset_fingerprint=correct_fp,
        canonical_train_size=8,
        realized_counts=base_manifest.realized_counts,
        class_distributions={},
        slice_supports={},
        duplicate_group_audit={},
        nested_budget_counts={},
        indices={
            "source_train": [0, 1, 2],
            "source_eval": [2, 3],  # Overlap at index 2!
            "target_adapt_pool": [4, 5],
            "target_eval": [6, 7],
        },
        provenance_contract={},
    )
    with pytest.raises(AssertionError, match="Pairwise index leakage detected"):
        audit_experimental_partitions(mock_df, bad_idx_manifest, ["feat_a", "feat_b"])

    # Negative Control 3: Duplicate feature-group leakage across partitions
    # Row 0 is in source_train, row 1 is in source_eval, but both have feat_a=1, feat_b='x'
    bad_grp_manifest = ExperimentalSplitManifest(
        dataset_id="mock",
        seed=42,
        source_dataset_fingerprint=correct_fp,
        canonical_train_size=8,
        realized_counts=base_manifest.realized_counts,
        class_distributions={},
        slice_supports={},
        duplicate_group_audit={},
        nested_budget_counts={},
        indices={
            "source_train": [0, 2],  # Row 0 has (1, 'x')
            "source_eval": [1, 3],   # Row 1 ALSO has (1, 'x')!
            "target_adapt_pool": [4, 5],
            "target_eval": [6, 7],
        },
        provenance_contract={},
    )
    with pytest.raises(AssertionError, match="Pairwise feature-group leakage detected"):
        audit_experimental_partitions(mock_df, bad_grp_manifest, ["feat_a", "feat_b"])

    # Negative Control 4: Row conservation failure (missing index 7)
    bad_rows_manifest = ExperimentalSplitManifest(
        dataset_id="mock",
        seed=42,
        source_dataset_fingerprint=correct_fp,
        canonical_train_size=8,
        realized_counts=base_manifest.realized_counts,
        class_distributions={},
        slice_supports={},
        duplicate_group_audit={},
        nested_budget_counts={},
        indices={
            "source_train": [0, 1],
            "source_eval": [2, 3],
            "target_adapt_pool": [4, 5],
            "target_eval": [6],  # Row 7 is missing!
        },
        provenance_contract={},
    )
    with pytest.raises(AssertionError, match="Row conservation violation"):
        audit_experimental_partitions(mock_df, bad_rows_manifest, ["feat_a", "feat_b"])

