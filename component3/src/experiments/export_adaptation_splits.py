"""Export verified four-way experimental adaptation splits for multi-dataset benchmarks.

Generates disjoint, duplicate-safe partitions for UCI Adult Full and Bank Marketing
from canonical train sets:
1. source_train (70%)
2. source_eval (10%)
3. target_adapt_pool (10%) -> with nested budgets B=50, 100, 200
4. target_eval (10%)

Preserves exact row conservation, pairwise disjointness, and class balance.
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from component3.src.datasets.dataset_registry import get_dataset_registry
from component3.src.splitting.experimental_splitter import (
    ExperimentalPartitionBuilder,
    compute_dataset_fingerprint,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "component3" / "results" / "adaptation_splits"


def export_dataset_splits(dataset_id: str, split_manifest_filename: str) -> None:
    """Build and save 4-way experimental partitions for a registered dataset."""
    reg = get_dataset_registry()
    meta = reg.get(dataset_id)
    raw_df = reg.load_dataset(dataset_id)

    # Load canonical train indices
    manifest_path = REPO_ROOT / "component3" / "results" / "split_manifests" / split_manifest_filename
    with manifest_path.open("r", encoding="utf-8") as f:
        canon_manifest = json.load(f)

    train_indices = canon_manifest["train_indices"]
    canon_train_df = raw_df.iloc[train_indices].copy().reset_index(drop=True)

    splits, exp_manifest = ExperimentalPartitionBuilder.build_splits(
        canonical_train_df=canon_train_df,
        metadata=meta,
        adaptation_budgets=(50, 100, 200),
        seed=42,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Save JSON manifest
    manifest_out = RESULTS_DIR / f"{dataset_id}_adaptation_split_manifest.json"
    exp_manifest.save(manifest_out)

    # 2. Save partition csv files
    prefix = f"{dataset_id}_"
    splits.source_train.to_csv(RESULTS_DIR / f"{prefix}source_train.csv", index=False)
    splits.source_eval.to_csv(RESULTS_DIR / f"{prefix}source_eval.csv", index=False)
    splits.target_adapt_pool.to_csv(RESULTS_DIR / f"{prefix}target_adapt_pool.csv", index=False)
    splits.target_eval.to_csv(RESULTS_DIR / f"{prefix}target_eval.csv", index=False)

    for b_size, b_df in splits.nested_adapt_budgets.items():
        b_df.to_csv(RESULTS_DIR / f"{prefix}target_adaptation_B{b_size}.csv", index=False)

    # 3. Save class distribution summary
    summary_rows = []
    for part_name, dist in exp_manifest.class_distributions.items():
        summary_rows.append({"partition": part_name, **dist})
    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / f"{prefix}adaptation_class_distribution.csv", index=False)

    print(f"Successfully exported {dataset_id} adaptation splits to {RESULTS_DIR}")
    print(f"  Realized counts: {exp_manifest.realized_counts}")
    print(f"  Nested budgets: {exp_manifest.nested_budget_counts}")


def main() -> None:
    export_dataset_splits("uci_adult_full", "uci_adult_full_split_manifest.json")
    export_dataset_splits("bank_marketing", "bank_marketing_split_manifest.json")


if __name__ == "__main__":
    main()
