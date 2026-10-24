"""Plan or explicitly train/evaluate/save one operational C4 candidate."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from c4.api.runtime_service import prepare_runtime_candidate
from c4.contracts.runtime_compatibility import _json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--upstream-json", type=Path, required=True)
    parser.add_argument("--execute-training", action="store_true")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--version")
    args = parser.parse_args()
    try:
        report = prepare_runtime_candidate(config_path=args.config,
            upstream=_json(args.upstream_json.read_bytes()),
            execute_training=args.execute_training, data_root=args.data_root, version=args.version)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        # Execution may have failed after fitting or during saving; do not claim
        # that training did not happen. Preserve partial output and inspect logs.
        print(json.dumps({"status": "OPERATIONAL_PREPARATION_FAILED",
            "error_type": type(error).__name__, "error": str(error),
            "execution_requested": args.execute_training,
            "partial_execution_possible": args.execute_training,
            "production_release_authorized": False}, indent=2))
        return 1
    print(json.dumps(report, indent=2, allow_nan=False))
    return 2 if report["status"] in (
        "CONFIGURATION_INCOMPLETE", "UPSTREAM_PREREQUISITES_FAILED") else 0


if __name__ == "__main__":
    raise SystemExit(main())
