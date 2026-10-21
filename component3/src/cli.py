"""Command-Line Interface (CLI) for Component 3.

Provides command-line commands for:
- Validating repair requests
- Running repairs
- Inspecting repair status
- Viewing prescription reasoning
- Inspecting validation results
- Inspecting generated repair packages
- Running reproducible demos
- Checking system readiness

All commands produce structured JSON output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

COMPONENT3_ROOT = REPO_ROOT / "component3"

from component3.src.api import (
    check_system_readiness,
    execute_contract_repair,
    execute_repair,
    export_c4_flat_contract,
    inspect_package,
    inspect_repair_status,
    inspect_validation_results,
    validate_external_contract,
    validate_repair_request,
    view_prescription_reasoning,
)
from component3.src.contracts.repair_package import RepairPackageStatus
from component3.src.demonstrations.run_demo_workflows import run_all_demonstrations




def _validate_output_directory(dir_str: str | None) -> Path:
    """Validate that output directory is inside component3."""
    if dir_str is None:
        target = COMPONENT3_ROOT / "results" / "cli_runs"
    else:
        target = Path(dir_str).resolve()
        comp3_root = COMPONENT3_ROOT.resolve()
        try:
            target.relative_to(comp3_root)
        except ValueError:
            import tempfile
            temp_dir = Path(tempfile.gettempdir()).resolve()
            if not target.is_relative_to(temp_dir):
                raise ValueError(f"Output directory '{target}' escapes component3 root.")
    target.mkdir(parents=True, exist_ok=True)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Component 3: Adaptive Intelligent Synthetic Data Repair Engine CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 1. validate-request
    p_val = subparsers.add_parser("validate-request", help="Validate a RepairRequest or multi-component JSON")
    p_val.add_argument("--request-json", "--input", dest="request_json", required=True, help="Path to input JSON")

    # 2. run-repair
    p_run = subparsers.add_parser("run-repair", help="Execute an end-to-end model repair")
    p_run.add_argument("--request-json", "--input", dest="request_json", required=True, help="Path to input JSON")
    p_run.add_argument("--output-dir", default=None, help="Output directory restricted to component3/")
    p_run.add_argument("--output-c4", default=None, help="Path to export flat 12-field C4 contract JSON")
    p_run.add_argument(
        "--decision-mode",
        choices=["rule_based", "learned"],
        default="rule_based",
        help="Prescription decision engine mode (rule_based or learned)",
    )

    # 2B. contract-repair (Dedicated multi-component C1->C2->C3->C4 command)
    p_crep = subparsers.add_parser("contract-repair", help="Execute contract repair with flat C4 export")
    p_crep.add_argument("--input", "--request-json", dest="request_json", required=True, help="Path to 3-section input JSON")
    p_crep.add_argument("--output", "--output-c4", dest="output_c4", default=None, help="Path to export flat C4 JSON")
    p_crep.add_argument("--output-dir", default=None, help="Output directory for artifacts")
    p_crep.add_argument("--decision-mode", choices=["rule_based", "learned"], default="rule_based")

    # 3. inspect-status
    p_stat = subparsers.add_parser("inspect-status", help="Inspect status of a saved repair package")
    p_stat.add_argument("--package-path", required=True, help="Path to RepairPackage manifest JSON")

    # 4. view-prescription
    p_pres = subparsers.add_parser("view-prescription", help="Evaluate prescription rules for a request")
    p_pres.add_argument("--request-json", "--input", dest="request_json", required=True, help="Path to RepairRequest JSON")
    p_pres.add_argument(
        "--decision-mode",
        choices=["rule_based", "learned"],
        default="rule_based",
        help="Prescription decision engine mode (rule_based or learned)",
    )

    # 5. inspect-validation
    p_vrep = subparsers.add_parser("inspect-validation", help="Inspect validation sub-reports from a package")
    p_vrep.add_argument("--package-path", required=True, help="Path to RepairPackage manifest JSON")

    # 6. inspect-package
    p_pkg = subparsers.add_parser("inspect-package", help="Inspect full RepairPackage dictionary")
    p_pkg.add_argument("--package-path", required=True, help="Path to RepairPackage manifest JSON")

    # 7. run-demo
    p_demo = subparsers.add_parser("run-demo", help="Run reproducible local demonstration workflows")
    p_demo.add_argument("--scenario", default="all", help="Scenario to run: 'all' or scenario letter (A-H)")
    p_demo.add_argument("--output-dir", default=None, help="Output directory restricted to component3/")

    # 8. check-readiness
    subparsers.add_parser("check-readiness", help="Check system environment and model integrity readiness")

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 2

    try:
        if args.command == "validate-request":
            res = validate_repair_request(args.request_json)
            print(json.dumps(res, indent=2))
            return 0 if res.get("valid") else 1

        elif args.command in ("run-repair", "contract-repair"):
            out_dir = _validate_output_directory(args.output_dir)
            out_c4 = getattr(args, "output_c4", None)
            if args.command == "contract-repair" or out_c4:
                c4_dict, pkg, log = execute_contract_repair(
                    args.request_json,
                    output_c4_path=out_c4,
                    output_directory=out_dir,
                    decision_mode=args.decision_mode,
                )
                print(json.dumps(c4_dict, indent=2))
                return 0 if c4_dict.get("repair_package_status") == "APPROVED" else 1
            else:
                pkg, log = execute_repair(
                    args.request_json,
                    output_directory=out_dir,
                    decision_mode=args.decision_mode,
                )
                summary = {
                    "package_id": pkg.package_id,
                    "request_id": pkg.request_id,
                    "run_id": pkg.run_id,
                    "chosen_strategy": pkg.chosen_strategy,
                    "quantity_selected": pkg.quantity_selected,
                    "package_status": pkg.package_status.value,
                    "execution_status": pkg.execution_status,
                    "rejection_reasons": list(pkg.rejection_reasons),
                    "output_directory": str(out_dir),
                    "manifest_path": log.package_path,
                }
                print(json.dumps(summary, indent=2))
                return 0 if pkg.package_status in (
                    RepairPackageStatus.APPROVED_FOR_REVIEW,
                    RepairPackageStatus.ABSTAINED_NO_REPAIR,
                ) else 1

        elif args.command == "inspect-status":
            res = inspect_repair_status(args.package_path)
            print(json.dumps(res, indent=2))
            return 0

        elif args.command == "view-prescription":
            res = view_prescription_reasoning(
                args.request_json,
                decision_mode=args.decision_mode,
            )
            print(json.dumps(res, indent=2))
            return 0

        elif args.command == "inspect-validation":
            res = inspect_validation_results(args.package_path)
            print(json.dumps(res, indent=2))
            return 0

        elif args.command == "inspect-package":
            res = inspect_package(args.package_path)
            print(json.dumps(res, indent=2))
            return 0

        elif args.command == "run-demo":
            out_dir = _validate_output_directory(args.output_dir)
            demo_results = run_all_demonstrations(output_dir=out_dir)
            print(json.dumps(demo_results, indent=2))
            return 0


        elif args.command == "check-readiness":
            res = check_system_readiness()
            print(json.dumps(res, indent=2))
            return 0 if res.get("system_status") == "READY" else 1

        else:
            parser.print_help()
            return 2

    except Exception as exc:
        err_out = {"error": str(exc), "status": "FAILED"}
        print(json.dumps(err_out, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
