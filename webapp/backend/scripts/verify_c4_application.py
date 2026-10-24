"""Verify the running C4 page API and save a separate application evidence checkpoint.

Run from the repository root. No training, calibration, future-outcome evaluation,
Git mutations or production release. Requires backend at localhost:8000.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

FREEZE = "de09b0220f25b3c82f8849dd2becef4f3301806dfcb1e701d5108b0d252fb4a7"
EXPECTED_INTERVALS = {
    "future_global_margin": [-0.01654219214797924, 0.003968170206024256],
    "future_slice_margin": [-0.012648227271989816, 0.009043439526700975],
    "future_nonaffected_margin": [-0.022121645826075398, 0.0016918876813342238],
}
SOURCES = (
    "webapp/backend/app/components/c4_service.py",
    "webapp/backend/app/components/c4_recorded_results.json",
    "webapp/backend/app/routers/c4.py",
    "webapp/frontend/src/pages/c4/C4Page.jsx",
    "webapp/frontend/src/pages/c4/C4Charts.jsx",
    "webapp/frontend/src/pages/c4/C4Page.css",
)

def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")

def http(base, path, payload=None):
    data = None if payload is None else encoded(payload)
    request = Request(base + path, data=data,
        headers={"Content-Type": "application/json"} if data else {})
    try:
        with urlopen(request, timeout=300) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))
    except URLError as exc:
        raise RuntimeError("Cannot reach the backend. Keep its terminal running.") from exc

def git(*arguments):
    result = subprocess.run(["git", *arguments], capture_output=True,
        text=True, encoding="utf-8", errors="replace", check=True)
    return result.stdout.strip()

def validate_case(payload, method, family):
    result = payload["result"]
    require(payload["controller_freeze_sha256"] == FREEZE, "Controller version differs.")
    require(payload["mode"] == "benchmark_replay", "Incorrect evidence mode.")
    require(payload["upstream_source"] == "simulated", "Unexpected upstream provenance.")
    require(payload["inference_executed"] is True and payload["candidate_retrained"] is False,
        "Expected inference on existing candidate evidence only.")
    require(result["method"] == method and result["family"] == family, "Wrong method/family result.")
    require(result["future_targets_used"] is False and result["production_release_authorized"] is False,
        "Unexpected future-target input or production authorization.")
    require(result["N1_executed"] is True and result["N2_executed"] is True,
        "Model inference did not complete.")
    require(result["selected_policy_kind"] == "review_only" and result["decision"] != "AUTO_DEPLOY",
        "Frozen measured policy changed.")
    require(len(payload["decision_time_features"]) == 75, "Unexpected feature count.")
    details = payload["case_details"]
    require(details["displayed_evidence_reconstructed"] is True, "Evidence reconstruction failed.")
    require(details["champion_model_id"] and details["candidate_model_id"], "Model identifiers missing.")
    require(payload["candidate_evaluation"] == details["candidate_evaluation"], "Displayed evaluation differs.")
    comparisons = details["candidate_evaluation"]
    require(comparisons["affected_slice"]["rows"] + comparisons["non_affected"]["rows"]
        == comparisons["global"]["rows"], "Population support does not reconcile.")
    for name, row in comparisons.items():
        if name not in ("global", "affected_slice", "non_affected"):
            continue
        actual = row["candidate"]["accuracy"] - row["champion"]["accuracy"]
        require(abs(actual-row["candidate_minus_champion"]["accuracy"]) <= 1e-12,
            "Candidate accuracy change differs: " + name)
    if method == "lightgbm":
        require(result["decision"] == "HUMAN_REVIEW", "Primary replay decision changed.")
        for target, expected in EXPECTED_INTERVALS.items():
            actual = result["n2_intervals"][target]
            require(len(actual) == 2 and all(abs(a-e) <= 1e-12 for a,e in zip(actual,expected)),
                "Recorded LightGBM interval differs: " + target)
            require(result["n3"]["dimensions"][target]["status"] == "UNCERTAIN",
                "Recorded primary safety assessment changed.")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    root = Path(git("rev-parse", "--show-toplevel"))
    require(Path.cwd().resolve() == root.resolve(), "Run from the repository root.")
    for name in SOURCES:
        require((root/name).is_file(), "Missing C4 application source: " + name)
    # Always local: no arbitrary remote URL or input-data upload.
    base = "http://127.0.0.1:8000"
    checks, responses = [], {}
    status, info = http(base, "/api/c4/info")
    require(status == 200 and info["production_release_authorized"] is False,
        "C4 information endpoint failed.")
    responses["info"] = info
    for method in ("lightgbm", "linear"):
        for family in ("adaptive", "global"):
            status, response = http(base, "/api/c4/release/evaluate",
                {"case_id":"real-outer-case", "method":method, "family":family})
            require(status == 200, f"{method}/{family}: HTTP {status}: {response}")
            validate_case(response, method, family)
            key = method + "_" + family
            responses[key] = response
            checks.append({"check":key,"passed":True,"decision":response["result"]["decision"]})
            print("CASE PASS:", method, family, response["result"]["decision"], flush=True)
    for label, payload, expected in (
        ("unknown_case", {"case_id":"unknown"}, 404),
        ("invalid_method", {"method":"invalid"}, 422),
        ("reject_future_target", {"future_target":0.1}, 422),
    ):
        status, response = http(base, "/api/c4/release/evaluate", payload)
        require(status == expected, f"{label}: expected {expected}, got {status}")
        checks.append({"check":label,"passed":True,"http_status":status})
        print("INPUT CHECK PASS:", label, status, flush=True)
    status, acceptance = http(base, "/api/c4/acceptance/verify", {})
    require(status == 200, f"Acceptance endpoint failed: {acceptance}")
    require(acceptance["research_observations"] is False
        and acceptance["empirical_auto_deployment_evidence"] is False,
        "Synthetic fixture provenance differs.")
    fixtures = acceptance["fixtures"]
    require(len(fixtures)==7 and all(f["passed"] and f["actual_decision"]==f["expected_decision"] for f in fixtures),
        "Seven passing acceptance checks required.")
    require({f["actual_decision"] for f in fixtures} == {"AUTO_DEPLOY","HUMAN_REVIEW","ESCALATE"},
        "Fixture decision branch coverage incomplete.")
    responses["synthetic_acceptance"] = acceptance
    checks.append({"check":"seven_synthetic_acceptance_fixtures","passed":True})
    print("SYNTHETIC ACCEPTANCE: 7 PASS; all three branches verified", flush=True)
    if not args.save:
        print("VERIFICATION PASS. No files written. Use --save to preserve evidence.")
        return
    data_root = os.environ.get("C4_DATA_ROOT")
    require(data_root, "Set C4_DATA_ROOT before saving application evidence.")
    require(Path(data_root).is_dir(), "C4_DATA_ROOT directory is unavailable.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = Path(data_root)/"c4_artifacts/application_checks/adult/c4-webapp-v2"/stamp
    directory.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, response in responses.items():
        contents=encoded(response)
        (directory/(name+".json")).write_bytes(contents)
        files[name+".json"] = sha(contents)
    record = {
        "purpose":"standalone_C4_API_application_verification", "created_at_utc":stamp,
        "git_revision":git("rev-parse","HEAD"), "working_tree_dirty":bool(git("status","--porcelain")),
        "controller_freeze_sha256":FREEZE, "python":platform.python_version(),
        "checks":checks, "files":files,
        "source_sha256":{name:sha((root/name).read_bytes()) for name in SOURCES},
        "research_metrics_recomputed":False, "models_refitted":False,
        "live_upstream_integration_verified":False, "production_release_authorized":False,
    }
    contents=encoded(record)
    (directory/"verification.json").write_bytes(contents)
    (directory/"verification.sha256").write_text(sha(contents)+"\n",encoding="ascii")
    lines=["# C4 standalone application verification", "",
        f"Created (UTC): {stamp}", f"Git revision: `{record['git_revision']}`",
        f"Working tree dirty: {record['working_tree_dirty']}", "",
        "Four recorded-case method/policy combinations passed API checks.",
        "Unknown cases returned 404; invalid methods and extra target fields returned 422.",
        "Seven synthetic acceptance fixtures passed, including all three decision branches.",
        "Displayed pre-release evaluation was reconstructed from the same 75 decision-time features.",
        "LightGBM replay intervals matched the verified terminal reference within 1e-12.", "",
        "Scope: existing candidate evidence and frozen controller inference. No new candidate training,",
        "future-outcome evaluation, recalibration or production deployment was executed.",
        "Synthetic branch checks do not establish empirical autonomous deployment performance.",
        "Real C1/C2/C3 integration and full-system testing remain pending.", "",
        f"Verification SHA256: `{sha(contents)}`", ""]
    note="\n".join(lines).encode("utf-8")
    (directory/"C4_APPLICATION_VERIFICATION.md").write_bytes(note)
    # Keep the full API payloads outside the repo. Only a concise evidence note belongs in Git.
    doc=root/"components/c4/docs"/("C4_APPLICATION_VERIFICATION_"+stamp+".md")
    doc.parent.mkdir(parents=True,exist_ok=True)
    with doc.open("xb") as stream: stream.write(note)
    print("EVIDENCE:",directory)
    print("DOCUMENT:",doc)
    print("VERIFICATION SHA256:",sha(contents))

if __name__ == "__main__":
    main()
