"""Read-only C4 operational configuration/preflight check; no training or release."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from c4.contracts.runtime_configuration import preflight_runtime


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="Trusted operator configuration; never an HTTP payload.")
    parser.add_argument("--upstream-json", type=Path,
                        help="Platform envelope containing context, c1, c2_to_c4 and c3.")
    args = parser.parse_args()
    try:
        report = preflight_runtime(args.config, args.upstream_json)
    except (OSError, ValueError) as error:
        print(json.dumps({"status": "PREFLIGHT_FAILED", "error": str(error),
            "training_executed": False, "production_release_authorized": False}, indent=2))
        return 1
    print(json.dumps(report, indent=2, allow_nan=False))
    # 2 = incomplete inputs; 0 = binding check completed, not operational approval.
    return 2 if report["status"] in (
        "CONFIGURATION_INCOMPLETE", "UPSTREAM_INPUT_REQUIRED") else 0


if __name__ == "__main__":
    raise SystemExit(main())
