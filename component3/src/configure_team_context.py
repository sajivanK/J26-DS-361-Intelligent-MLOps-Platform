r"""Validate team artifact locations and write C3's external onboarding file.

Usage: python -m component3.src.configure_team_context [--data-dir C:\J26\data]
Only creates C3 configuration; never modifies a model or dataset.
"""
import argparse
import json
from pathlib import Path

from component3.src.integration_storage import data_root, contexts_file


def configure(root=None, destination=None):
    root = Path(root) if root else data_root()
    destination = Path(destination) if destination else contexts_file()
    template = Path(__file__).resolve().parents[1] / "docs/integration_review/c3_contexts.example.json"
    context = json.loads(template.read_text(encoding="utf-8"))["adult-xgb-v1"]
    context["model_uri"] = str((root / "c2/base_model_adult.pkl").resolve())
    context["reference_dataset_uri"] = str((root / "uci/reference.csv").resolve())
    required = [Path(context["model_uri"]), Path(context["reference_dataset_uri"]), root / "uci/test.csv"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Team artifacts missing; configuration was not written: " + ", ".join(missing))
    # Check the schema without deserializing the model or rewriting any data.
    import csv
    for path in required[1:]:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            columns = next(csv.reader(stream))
        if columns != context["feature_columns"] + ["target"]:
            raise ValueError(f"Unexpected preprocessed team schema: {path}")
    existing = json.loads(destination.read_text(encoding="utf-8")) if destination.is_file() else {}
    if "adult-xgb-v1" in existing and existing["adult-xgb-v1"] != context:
        raise ValueError("Existing adult-xgb-v1 configuration differs; review it before replacing it.")
    existing["adult-xgb-v1"] = context
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--destination", type=Path)
    arguments = parser.parse_args()
    print(configure(arguments.data_dir, arguments.destination))
