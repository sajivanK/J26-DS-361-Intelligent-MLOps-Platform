"""Load and inspect the draft Adult benchmark without executing it."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from c4.data import resolve_data_root
from c4.data.split_manifest import load_adult_split_manifest
from c4.scenarios.config import ScenarioConfig
from c4.scenarios.slices import reference_quantile_slice


class UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate YAML keys instead of silently overwriting them."""


def _unique_mapping(loader, node):
    loader.flatten_mapping(node)
    result = {}

    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)

        if not isinstance(key, str):
            raise ValueError("Benchmark YAML mapping keys must be strings.")

        if key in result:
            raise ValueError(f"Duplicate YAML key: {key}")

        result[key] = loader.construct_object(value_node, deep=True)

    return result


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _unique_mapping,
)


@dataclass(frozen=True)
class DraftBenchmark:
    document: dict
    yaml_sha256: str
    injection_configs: tuple[ScenarioConfig, ...]
    proposed_scenario_count: int


def load_draft_benchmark(path: str | Path) -> DraftBenchmark:
    """Validate the draft's injection grid; never authorize execution."""

    contents = Path(path).read_bytes()
    document = yaml.load(
        contents.decode("utf-8"),
        Loader=UniqueKeyLoader,
    )

    if not isinstance(document, dict):
        raise ValueError("Benchmark YAML must contain a mapping.")

    if document.get("benchmark_schema_version") != "0.1":
        raise ValueError("Unsupported benchmark schema version.")

    if (
        document.get("status") != "draft"
        or document.get("execution_enabled") is not False
    ):
        raise ValueError(
            "This loader accepts non-executable drafts only."
        )

    if document.get("dataset") != "uci_adult":
        raise ValueError("This benchmark supports Adult only.")

    # Reject nonfinite values and non-JSON-compatible configuration data.
    json.dumps(document, allow_nan=False)

    try:
        seeds = document["seed_families"]
        strengths = document["injection_strengths"]
        conditions = document["drift_conditions"]
        diagnoses = document["diagnosis_conditions"]
        repairs = document["repair_conditions"]
        unresolved = document["design"]["unresolved"]
    except (KeyError, TypeError) as error:
        raise ValueError("Required benchmark sections are missing.") from error

    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(seed) is not int for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Seed families must be unique integers.")

    if (
        not isinstance(strengths, dict)
        or set(strengths) != {"low", "medium", "high"}
    ):
        raise ValueError("Define low, medium and high injection strengths.")

    magnitudes = [strengths[name] for name in ("low", "medium", "high")]

    if (
        any(type(value) not in (int, float) for value in magnitudes)
        or not 0 < magnitudes[0] < magnitudes[1] < magnitudes[2] <= 1
    ):
        raise ValueError(
            "Injection strengths must satisfy 0 < low < medium < high <= 1."
        )

    for section_name, entries in [
        ("drift_conditions", conditions),
        ("diagnosis_conditions", diagnoses),
        ("repair_conditions", repairs),
    ]:
        if (
            not isinstance(entries, list)
            or not entries
            or any(not isinstance(entry, dict) for entry in entries)
        ):
            raise ValueError(f"{section_name} must be a nonempty list.")

        names = [entry.get("name") for entry in entries]

        if (
            any(not isinstance(name, str) or not name.strip() for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError(f"{section_name} requires unique nonempty names.")

    if (
        not isinstance(unresolved, list)
        or not unresolved
        or any(not isinstance(item, str) or not item.strip() for item in unresolved)
    ):
        raise ValueError("This draft must record unresolved design choices.")

    configs = []

    for condition in conditions:
        if set(condition) != {
            "name", "mechanism", "features", "interaction_pairs"
        }:
            raise ValueError("Drift condition fields are invalid.")

        for magnitude in magnitudes:
            for seed in seeds:
                configs.append(
                    ScenarioConfig(
                        dataset=document["dataset"],
                        champion_version=document["champion_version"],
                        mechanism=condition["mechanism"],
                        magnitude=magnitude,
                        features=condition["features"],
                        interaction_pairs=condition["interaction_pairs"],
                        seed=seed,
                    )
                )

    hashes = [config.configuration_sha256() for config in configs]

    if len(set(hashes)) != len(hashes):
        raise ValueError("The injection grid contains duplicate specifications.")

    proposed_count = len(configs) * len(diagnoses) * len(repairs)

    if document["design"].get("proposed_factorial_combinations") != proposed_count:
        raise ValueError("Declared scenario count differs from the factor grid.")

    return DraftBenchmark(
        document=document,
        yaml_sha256=hashlib.sha256(contents).hexdigest(),
        injection_configs=tuple(configs),
        proposed_scenario_count=proposed_count,
    )


def verify_draft_sources(
    draft: DraftBenchmark,
    *,
    data_root: str | Path | None = None,
) -> dict[str, object]:
    """Verify the manifest link, role configuration and reference slices."""

    root = resolve_data_root(data_root)
    document = draft.document
    roles = document["data_roles"]

    directory = (
        root / "c4_artifacts" / "splits" / "adult"
        / roles["split_manifest_version"]
    )
    manifest_bytes = (directory / "manifest.json").read_bytes()

    if hashlib.sha256(manifest_bytes).hexdigest() != (
        roles["split_manifest_sha256"]
    ):
        raise ValueError("Draft links to a different split manifest.")

    manifest = load_adult_split_manifest(directory, data_root=root)

    if (directory / "manifest.json").read_bytes() != manifest_bytes:
        raise ValueError("Split manifest changed during verification.")

    expected_config = {
        "repair_fraction": roles["repair_fraction"],
        "release_validation_fraction": roles["release_validation_fraction"],
        "sealed_future_fraction": roles["sealed_future_fraction"],
        "random_state": roles["split_seed"],
    }

    if manifest["split_config"] != expected_config:
        raise ValueError("Draft split settings differ from the manifest.")

    actual_counts = {
        "reference": manifest["reference_rows"],
        "eligible_holdout": manifest["holdout_audit"]["eligible_rows"],
        **{
            role: partition["rows"]
            for role, partition in manifest["partitions"].items()
        },
    }

    if roles["expected_rows"] != actual_counts:
        raise ValueError("Draft row counts differ from the manifest.")

    # Read reference only for threshold verification.
    reference_path = root / "uci_adult_data" / "reference.csv"
    reference_bytes = reference_path.read_bytes()

    if hashlib.sha256(reference_bytes).hexdigest() != (
        manifest["source_checksums"]["reference_csv_sha256"]
    ):
        raise ValueError("Reference CSV changed during verification.")

    import io

    reference = pd.read_csv(io.BytesIO(reference_bytes))
    slice_specs = [document["true_slice"]]

    for condition in document["diagnosis_conditions"]:
        specification = condition["diagnosed_slice"]

        if specification == "same_as_true_slice":
            continue

        if not isinstance(specification, dict):
            raise ValueError("Unsupported diagnosis slice specification.")

        slice_specs.append(specification)

    verified_rules = {}

    for specification in slice_specs:
        if (
            specification["derive_from"] != "reference"
            or specification["rule_feature_space"] != "processed"
        ):
            raise ValueError("Slices must use processed reference features.")

        rule = reference_quantile_slice(
            reference,
            name=specification["name"],
            feature=specification["feature"],
            quantile=specification["reference_quantile"],
            operator=specification["operator"],
        )
        mask = rule.mask(reference)

        if not math.isclose(
            rule.value,
            specification["resolved_threshold"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Recorded slice threshold differs from reference.")

        if int(mask.sum()) != specification["reference_members"]:
            raise ValueError("Recorded slice member count is incorrect.")

        if not math.isclose(
            float(mask.mean()),
            specification["reference_support"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Recorded slice support is incorrect.")

        verified_rules[rule.name] = rule.model_dump(mode="json")

    return {
        "source_verification": "PASS",
        "verified_slice_rules": verified_rules,
        "proposed_scenario_count": draft.proposed_scenario_count,
        "execution_enabled": False,
    }