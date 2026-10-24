"""
Demo client for the C2 API: what C3/C4 integration will look like.
J26-DS-361 | C2 XAI Root Cause Diagnosis Engine | Tharmika N (IT22247940)

1. Start the API in one terminal:   python3.11 -m uvicorn c2.api:app --port 8002
2. Run this in a second terminal:    python3.11 -m c2.api_demo

It sends 2 clean warm-up batches, then 3 drifted cycles (capital_gain drift)
with C1 output from the team's shared v1.1 fake builder, exactly as the real C1
API will send it during integration week. Every cycle is stored in the run
history; the demo then reads the history and the business report of the
latest run (FR7, written by OpenAI when OPENAI_API_KEY is set on the server).
"""
import json
import warnings

import requests

from c2.src.config import TARGET_COL  # noqa: I001  (adds shared/ to the path first)
from contracts.c1_fake_output import build_fake_c1_output
from c2.src.scenarios import make_temporal_stream

warnings.filterwarnings("ignore", category=UserWarning)
URL = "http://localhost:8002"


def rows(df):
    """Production rows: the label column is never sent (C2 works without labels)."""
    return df.drop(columns=[TARGET_COL]).to_dict(orient="records")


def main():
    print("health:", requests.get(f"{URL}/health", timeout=30).json())
    requests.post(f"{URL}/reset", timeout=30)          # start a fresh run history
    stream = make_temporal_stream("marginal_single", ["capital_gain"], 0.6, seed=22)

    r = requests.post(f"{URL}/warmup", timeout=300,
                      json={"batches": [rows(s["drifted_df"]) for s in stream[:2]]})
    print("warmup:", r.status_code, r.json())

    for s in stream[2:]:
        c1 = build_fake_c1_output(s["ground_truth"], seed=s["spec"]["seed"])
        r = requests.post(f"{URL}/diagnose", timeout=300,
                          json={"c1_output": c1, "c1_trusted": False,      # fake C1: hints only
                                "production_data": rows(s["drifted_df"]),
                                "source": "demo · capital_gain shift 0.6"})
        out = r.json()
        print(f"\nrun {out.get('run')}: HTTP {r.status_code}  "
              f"root={out['to_c3']['root_cause_feature']}  "
              f"stability={out['to_c3']['diagnosis_stability']}  "
              f"latency={out['details']['latency_s']}s  "
              f"contract problems={out['contract_problems']}")

    print("\nLast repair specification sent to C3:")
    print(json.dumps(out["to_c3"], indent=2))
    ok = r.status_code == 200 and not any(out["contract_problems"].values())

    # NFR7: missing C1 output still gives a diagnosis, flagged in details
    s = stream[-1]
    r2 = requests.post(f"{URL}/diagnose", timeout=300,
                       json={"production_data": rows(s["drifted_df"])})
    print(f"\nwithout C1: HTTP {r2.status_code}, c1_status="
          f"{r2.json()['details']['c1_status']}")
    # NFR7: bad production data gives a structured error, not a crash
    r3 = requests.post(f"{URL}/diagnose", timeout=60,
                       json={"production_data": [{"age": 1.0}] * 5})
    print(f"bad data:   HTTP {r3.status_code}, error={r3.json().get('error')}")
    # FR9: audit trail
    a = requests.get(f"{URL}/audit", params={"last": 2}, timeout=30).json()
    print(f"audit:      {len(a['records'])} latest records, last version "
          f"{a['records'][-1]['diagnosis_version']}")
    # Run history and business report (FR7)
    h = requests.get(f"{URL}/runs", timeout=30).json()
    print(f"history:    {len(h['runs'])} runs stored, newest first: "
          + ", ".join(f"#{x['run']} {x['state']}" for x in h["runs"]))
    n = h["runs"][0]["run"]
    rep = requests.get(f"{URL}/runs/{n}/report", timeout=120).json()
    print(f"\nreport for run #{n} (written by {rep['writer']}"
          f"{', ' + rep['fallback_reason'] if rep['fallback_reason'] else ''}):")
    print(f"  {rep['title']}")
    print(f"  decision: {rep['decision']['head']}")
    print(f"  summary:  {rep['sections']['summary']}")
    print(f"  claim check passed: {rep['claim_check']['passed']}")
    md = requests.get(f"{URL}/runs/{n}/report.md", timeout=60)
    print(f"  markdown export: HTTP {md.status_code}, {len(md.text)} characters")
    ok = (ok and r2.status_code == 200 and r3.status_code == 422
          and rep["claim_check"]["passed"] and md.status_code == 200)
    print("\nAPI DEMO OK" if ok else "\nAPI DEMO FAILED")


if __name__ == "__main__":
    main()