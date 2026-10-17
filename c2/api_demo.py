"""
Demo client for the C2 API: what C3/C4 integration will look like.
J26-DS-361 | C2 XAI Root Cause Diagnosis Engine | Tharmika N (IT22247940)

1. Start the API in one terminal:   python3.11 -m uvicorn c2.api:app --port 8002
2. Run this in a second terminal:    python3.11 -m c2.api_demo

It sends 2 clean warm-up batches, then 3 drifted cycles (capital_gain drift)
with C1 output from the team's shared v1.1 fake builder, exactly as the real C1
API will send it during integration week.
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
    return df.to_dict(orient="records")


def main():
    print("health:", requests.get(f"{URL}/health", timeout=30).json())
    stream = make_temporal_stream("marginal_single", ["capital_gain"], 0.6, seed=22)

    r = requests.post(f"{URL}/warmup", timeout=300,
                      json={"batches": [rows(s["drifted_df"]) for s in stream[:2]]})
    print("warmup:", r.status_code, r.json())

    for s in stream[2:]:
        c1 = build_fake_c1_output(s["ground_truth"], seed=s["spec"]["seed"])
        r = requests.post(f"{URL}/diagnose", timeout=300,
                          json={"c1_output": c1, "production_data": rows(s["drifted_df"])})
        out = r.json()
        print(f"\ncycle {s['cycle']}: HTTP {r.status_code}  "
              f"root={out['to_c3']['root_cause_feature']}  "
              f"stability={out['to_c3']['diagnosis_stability']}  "
              f"latency={out['details']['latency_s']}s  "
              f"contract problems={out['contract_problems']}")

    print("\nLast repair specification sent to C3:")
    print(json.dumps(out["to_c3"], indent=2))
    ok = r.status_code == 200 and not any(out["contract_problems"].values())
    print("\nAPI DEMO OK" if ok else "\nAPI DEMO FAILED")


if __name__ == "__main__":
    main()