"""Experimental Matrix Planning Framework for Component 3.

Designs and schedules large-scale multi-dataset, multi-model, multi-drift repair experiments.
Calculates Cartesian products (up to 4,320 combinations), evaluates eligibility constraints,
estimates computational budgets from empirical pilot benchmarks, and produces stratified sample plans.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
import numpy as np

from component3.src.baselines.baseline_trainer import ModelFamily
from component3.src.datasets.dataset_registry import DatasetRegistry, get_dataset_registry
from component3.src.drift.drift_config import DriftCategory

# Empirical benchmark measurements (seconds per run from Sprint 05 pilot experiments)
# Derived from actual runtime logs on CPU:
# - TVAE (5 epochs): ~2.2s training + ~0.05s sampling + ~0.8s validation = ~3.0s total
# - CTGAN (5 epochs): ~3.5s - 18.0s (mean ~8.5s) training + ~0.05s sampling + ~0.8s validation = ~9.5s total
# - TabDDPM (3 epochs, 100 timesteps): ~28.0s training + ~1.5s sampling + ~0.8s validation = ~30.5s total
EMPIRICAL_PILOT_BENCHMARKS_CPU: dict[str, float] = {
    "TVAE": 3.0,
    "CTGAN": 9.5,
    "TABDDPM": 30.5,
}

# GPU acceleration factors (from T4 benchmarks)
GPU_SPEEDUP_FACTORS: dict[str, float] = {
    "TVAE": 1.2,     # TVAE is already fast on CPU, minimal tensor gain
    "CTGAN": 3.5,    # GAN discriminator/generator training benefits moderately
    "TABDDPM": 6.5,  # Diffusion tensor denoising loop benefits heavily from CUDA
}


@dataclass
class PlannedExperiment:
    """Specification of an individual planned repair experiment in the Cartesian grid."""
    experiment_id: str
    dataset_id: str
    model_family: str
    drift_category: str
    generator: str
    repair_quantity: int
    seed: int
    epochs: int
    is_eligible: bool
    eligibility_reason: str
    estimated_seconds_cpu: float
    estimated_seconds_gpu: float
    runtime_provenance: str = "cpu_measured_pilot__gpu_unverified_assumption"

    def to_dict(self) -> dict[str, Any]:
        """Convert planned experiment to dictionary."""
        return asdict(self)


@dataclass
class MatrixPlanSummary:
    """Summary statistics for an experimental matrix plan."""
    total_combinations: int
    eligible_combinations: int
    ineligible_combinations: int
    estimated_total_cpu_hours: float
    estimated_total_gpu_hours: float
    breakdown_by_dataset: dict[str, int]
    breakdown_by_generator: dict[str, int]
    breakdown_by_drift_category: dict[str, int]
    breakdown_by_model: dict[str, int]
    runtime_provenance: str = "CPU estimates measured from Sprint 05 pilot logs; GPU estimates are unverified theoretical projections pending Colab calibration."
    generated_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize summary to dictionary."""
        return asdict(self)


class ExperimentMatrixPlanner:
    """Orchestrates Cartesian grid generation, eligibility checking, and sampling."""

    def __init__(self, registry: DatasetRegistry | None = None) -> None:
        self.registry = registry or get_dataset_registry()

    def generate_full_matrix(
        self,
        dataset_ids: list[str] | None = None,
        model_families: list[str] | None = None,
        drift_categories: list[str] | None = None,
        generators: list[str] | None = None,
        quantities: list[int] | None = None,
        seeds: list[int] | None = None,
        epochs_map: dict[str, int] | None = None,
    ) -> list[PlannedExperiment]:
        """Generate the comprehensive Cartesian matrix across all dimensions."""
        active_datasets = dataset_ids or ["uci_adult", "telco_churn", "home_credit", "ieee_cis", "bank_marketing"]
        active_models = model_families or [m.value for m in ModelFamily]
        active_drifts = drift_categories or [c.value for c in DriftCategory]
        active_gens = generators or ["CTGAN", "TVAE", "TABDDPM"]
        active_quantities = quantities or [200, 500, 1000, 2000]
        active_seeds = seeds or [42, 123, 456]
        default_epochs = epochs_map or {"CTGAN": 5, "TVAE": 5, "TABDDPM": 3}

        experiments: list[PlannedExperiment] = []

        for d_id in sorted(active_datasets):
            for m_fam in sorted(active_models):
                for d_cat in sorted(active_drifts):
                    for gen in sorted(active_gens):
                        gen_upper = gen.upper()
                        epochs = default_epochs.get(gen_upper, 5)

                        for qty in active_quantities:
                            for seed in active_seeds:
                                exp_id = f"{d_id}_{m_fam}_{d_cat[:10]}_{gen_upper.lower()}_n{qty}_s{seed}"

                                # Check eligibility
                                is_eligible = True
                                reasons: list[str] = []

                                # Check dataset registry existence
                                try:
                                    meta = self.registry.get(d_id)
                                    if not meta.is_local_available and d_id not in ("uci_adult", "ieee_cis"):
                                        reasons.append(f"Dataset {d_id} pending download (Colab required)")
                                except KeyError:
                                    is_eligible = False
                                    reasons.append(f"Dataset {d_id} not in registry")

                                # Check generator suitability constraints
                                if d_cat == DriftCategory.GENUINE_CONCEPT_DRIFT.value and gen_upper == "TABDDPM":
                                    # Concept drift requires label guidance
                                    reasons.append("TabDDPM unconditional generation requires label guidance adapter")

                                # Compute estimated runtimes
                                base_cpu = EMPIRICAL_PILOT_BENCHMARKS_CPU.get(gen_upper, 10.0)
                                # Scale runtime slightly with sample quantity
                                qty_scale = 1.0 + (qty / 2000.0) * 0.4
                                est_cpu = base_cpu * qty_scale
                                speedup = GPU_SPEEDUP_FACTORS.get(gen_upper, 2.0)
                                est_gpu = est_cpu / speedup

                                if reasons:
                                    eligibility_str = "; ".join(reasons)
                                else:
                                    eligibility_str = "Eligible for immediate execution"

                                experiments.append(
                                    PlannedExperiment(
                                        experiment_id=exp_id,
                                        dataset_id=d_id,
                                        model_family=m_fam,
                                        drift_category=d_cat,
                                        generator=gen_upper,
                                        repair_quantity=qty,
                                        seed=seed,
                                        epochs=epochs,
                                        is_eligible=is_eligible,
                                        eligibility_reason=eligibility_str,
                                        estimated_seconds_cpu=round(est_cpu, 2),
                                        estimated_seconds_gpu=round(est_gpu, 2),
                                    )
                                )

        return experiments

    def summarize_plan(self, experiments: list[PlannedExperiment]) -> MatrixPlanSummary:
        """Compute aggregated statistics and resource estimations for a plan."""
        total = len(experiments)
        eligible = sum(1 for e in experiments if e.is_eligible)
        ineligible = total - eligible

        total_cpu_sec = sum(e.estimated_seconds_cpu for e in experiments)
        total_gpu_sec = sum(e.estimated_seconds_gpu for e in experiments)

        def count_by(attr: str) -> dict[str, int]:
            d: dict[str, int] = {}
            for e in experiments:
                val = getattr(e, attr)
                d[val] = d.get(val, 0) + 1
            return d

        return MatrixPlanSummary(
            total_combinations=total,
            eligible_combinations=eligible,
            ineligible_combinations=ineligible,
            estimated_total_cpu_hours=round(total_cpu_sec / 3600.0, 4),
            estimated_total_gpu_hours=round(total_gpu_sec / 3600.0, 4),
            breakdown_by_dataset=count_by("dataset_id"),
            breakdown_by_generator=count_by("generator"),
            breakdown_by_drift_category=count_by("drift_category"),
            breakdown_by_model=count_by("model_family"),
        )

    def sample_balanced_subset(
        self,
        experiments: list[PlannedExperiment],
        sample_size: int = 100,
        seed: int = 42,
    ) -> list[PlannedExperiment]:
        """Sample a balanced, representative subset stratified across datasets, drift categories, and generators."""
        if sample_size >= len(experiments):
            return list(experiments)

        rng = np.random.default_rng(seed)

        # Group by (dataset_id, generator, drift_category)
        strata: dict[tuple[str, str, str], list[PlannedExperiment]] = {}
        for exp in experiments:
            key = (exp.dataset_id, exp.generator, exp.drift_category)
            strata.setdefault(key, []).append(exp)

        sampled: list[PlannedExperiment] = []
        n_strata = len(strata)
        per_stratum = max(1, sample_size // n_strata)

        for key, items in sorted(strata.items()):
            n_take = min(per_stratum, len(items))
            chosen_indices = rng.choice(len(items), size=n_take, replace=False)
            for idx in chosen_indices:
                sampled.append(items[idx])
                if len(sampled) >= sample_size:
                    break
            if len(sampled) >= sample_size:
                break

        # If not filled up to sample_size due to rounding, sample remaining without duplicates
        if len(sampled) < sample_size:
            already_chosen = {e.experiment_id for e in sampled}
            remaining = [e for e in experiments if e.experiment_id not in already_chosen]
            needed = sample_size - len(sampled)
            if remaining:
                chosen = rng.choice(len(remaining), size=min(needed, len(remaining)), replace=False)
                for idx in chosen:
                    sampled.append(remaining[idx])

        return sampled

    def filter_by_budget(
        self,
        experiments: list[PlannedExperiment],
        max_budget_hours: float = 5.0,
        use_gpu: bool = True,
    ) -> list[PlannedExperiment]:
        """Select a subset of experiments fitting strictly within a maximum compute time budget."""
        max_seconds = max_budget_hours * 3600.0
        allocated_seconds = 0.0
        selected: list[PlannedExperiment] = []

        for exp in experiments:
            cost = exp.estimated_seconds_gpu if use_gpu else exp.estimated_seconds_cpu
            if allocated_seconds + cost <= max_seconds:
                selected.append(exp)
                allocated_seconds += cost
            else:
                break

        return selected

    def filter_completed_experiments(
        self,
        experiments: list[PlannedExperiment],
        completed_experiment_ids: set[str],
    ) -> list[PlannedExperiment]:
        """Exclude already executed experiment IDs for resumption."""
        return [e for e in experiments if e.experiment_id not in completed_experiment_ids]

    def export_plan(self, experiments: list[PlannedExperiment], filepath: Path | str) -> None:
        """Export plan to JSON file."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        summary = self.summarize_plan(experiments)
        payload = {
            "summary": summary.to_dict(),
            "experiments": [e.to_dict() for e in experiments],
        }
        with p.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Experimental matrix plan exported: {p} ({len(experiments)} runs)")
