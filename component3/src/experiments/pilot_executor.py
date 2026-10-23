"""Pilot Experiment Suite Executor and Checkpointing Engine for Component 3.

Executes a representative, scientifically controlled pilot batch across:
- Multiple datasets (UCI Adult, Telco Customer Churn)
- All 3 generators (CTGAN, TVAE, TabDDPM)
- Multiple model families (XGBoost, Random Forest)
- Full Sprint 05 Multi-Stage Validation Engine
- Atomic Checkpointing & Resume logic
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Any
import pandas as pd

from component3.src.experiments.multi_dataset_runner import MultiDatasetRepairRunner

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = ROOT / "component3" / "results" / "repair_experiments"
DEFAULT_CHECKPOINT_DIR = ROOT / "component3" / "results" / "checkpoints"


@dataclass
class PilotRunSummary:
    """Aggregated execution summary of the pilot batch."""
    total_planned: int
    total_executed: int
    total_skipped_from_checkpoint: int
    total_failed: int
    total_seconds: float
    results_by_generator: dict[str, list[dict[str, Any]]]
    generated_at_utc: str


class PilotExperimentExecutor:
    """Executes and manages checkpointed pilot experiment batches."""

    def __init__(
        self,
        results_dir: Path | str | None = None,
        checkpoint_dir: Path | str | None = None,
    ) -> None:
        self.results_dir = Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else DEFAULT_CHECKPOINT_DIR
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "pilot_checkpoint.json"
        self.runner = MultiDatasetRepairRunner(results_dir=self.results_dir)

    def load_completed_ids(self) -> set[str]:
        """Load completed experiment IDs from disk."""
        if self.checkpoint_file.is_file():
            try:
                with self.checkpoint_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    return set(data.get("completed_ids", []))
            except Exception:
                return set()
        return set()

    def record_completed(self, exp_id: str) -> None:
        """Atomically record an experiment completion into checkpoint file."""
        completed = self.load_completed_ids()
        completed.add(exp_id)
        temp_file = self.checkpoint_dir / "pilot_checkpoint.json.tmp"
        with temp_file.open("w", encoding="utf-8") as f:
            json.dump({
                "completed_ids": sorted(list(completed)),
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            }, f, indent=2)
        temp_file.replace(self.checkpoint_file)

    def run_pilot(
        self,
        datasets: list[str] | None = None,
        generators: list[str] | None = None,
        models: list[str] | None = None,
        n_samples: int = 100,
        epochs: int = 2,
        seed: int = 42,
        max_rows: int = 250,
        force_rerun: bool = False,
    ) -> tuple[list[dict[str, Any]], PilotRunSummary]:
        """Execute the representative multi-dataset, 3-generator pilot matrix."""
        active_datasets = datasets or ["uci_adult", "telco_churn"]
        active_generators = generators or ["tvae", "ctgan", "tabddpm"]
        active_models = models or ["xgboost", "random_forest"]

        planned_experiments: list[dict[str, Any]] = []
        for d in active_datasets:
            for m in active_models:
                for g in active_generators:
                    planned_experiments.append({
                        "dataset_id": d,
                        "model_family": m,
                        "generator_name": g,
                        "scenario_id": "S01",
                        "n_samples": n_samples,
                        "epochs": epochs if g != "tabddpm" else max(1, epochs - 1),
                        "seed": seed,
                        "max_rows": max_rows,
                    })

        t0 = time.perf_counter()
        completed_ids = set() if force_rerun else self.load_completed_ids()
        results: list[dict[str, Any]] = []
        results_by_gen: dict[str, list[dict[str, Any]]] = {g.upper(): [] for g in active_generators}
        skipped_count = 0
        failed_count = 0

        logger.info("Starting Pilot Execution: %d total planned experiments.", len(planned_experiments))

        for item in planned_experiments:
            exp_id = f"{item['dataset_id']}_{item['model_family']}_{item['scenario_id']}_{item['generator_name']}_n{item['n_samples']}_s{item['seed']}"
            if exp_id in completed_ids:
                # Load existing result if available on disk
                res_path = self.results_dir / f"{exp_id}.json"
                if res_path.is_file():
                    with res_path.open("r", encoding="utf-8") as f:
                        saved_rec = json.load(f)
                    results.append(saved_rec)
                    results_by_gen[item["generator_name"].upper()].append(saved_rec)
                    skipped_count += 1
                    continue

            try:
                rec = self.runner.run_experiment(
                    dataset_id=item["dataset_id"],
                    scenario_id=item["scenario_id"],
                    generator_name=item["generator_name"],
                    n_samples=item["n_samples"],
                    epochs=item["epochs"],
                    seed=item["seed"],
                    model_family=item["model_family"],
                    max_rows=item["max_rows"],
                )
                self.record_completed(exp_id)
                results.append(rec)
                results_by_gen[item["generator_name"].upper()].append(rec)
            except Exception as exc:
                failed_count += 1
                logger.error("Pilot experiment %s encountered error: %s", exp_id, exc)

        total_elapsed = time.perf_counter() - t0

        summary = PilotRunSummary(
            total_planned=len(planned_experiments),
            total_executed=len(results) - skipped_count,
            total_skipped_from_checkpoint=skipped_count,
            total_failed=failed_count,
            total_seconds=round(total_elapsed, 2),
            results_by_generator=results_by_gen,
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
        )

        return results, summary
