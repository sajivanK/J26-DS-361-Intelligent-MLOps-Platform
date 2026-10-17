# C2 engine output check (contracts v1.1)

Streams: 51 (scenario suite + clean), 2 warm-up + 3 diagnosed cycles each, C1 input from shared/contracts/c1_fake_output.py.

| Check | Result |
|---|---|
| C2 -> C3 outputs valid | 51/51 |
| C2 -> C4 outputs valid | 51/51 |
| root_cause_category accuracy (C2 evidence only) | 92.2% |
| root_cause_category accuracy (C2 + C1 hints) | 98.0% |
| Diagnosis latency per cycle, median | 0.16 s |
| Diagnosis latency per cycle, max | 0.36 s |

The fake C1 is built from the injection answer key, so the C2-only category accuracy is the honest figure until real C1 output is connected.


### Category confusion (rows = truth, columns = C2 + C1 hints)

| truth \ predicted | combined | interaction | marginal | unknown |
|---|---|---|---|---|
| combined | 2 | 0 | 0 | 0 |
| interaction | 0 | 4 | 0 | 0 |
| marginal | 0 | 0 | 40 | 0 |
| unknown | 0 | 0 | 1 | 4 |
