# Intelligent MLOps Platform: web app

J26-DS-361 | Sajivan K (C1, lead), Tharmika N (C2), Mathusan K (C3), Kesigan M (C4)

One FastAPI backend runs all four components in-process; a React frontend (`frontend/`, next step) shows them.

## Structure

```
webapp/backend/app/
  main.py            creates the app, loads every component at startup        (rarely edited)
  settings.py        data folder, dataset list, model paths (env variables)     (add your paths)
  contracts.py       checks every output against shared/contracts/*.json        (do not edit)
  orchestrator.py    runs C1 -> C2 -> C3 -> C4 and the repair gate              (lead)
  uploads.py         reads an uploaded CSV batch, removes label columns
  components/
    base.py          the interface every component follows                     (do not edit)
    c1.py            C1 Drift Detection, REAL                                  (Sajivan)
    c2.py            C2 Root Cause Diagnosis, FAKE until replaced               (Tharmika)
    c3.py            C3 Synthetic Data Repair, FAKE until replaced              (Mathusan)
    c4.py            C4 Release Controller, FAKE until replaced                 (Kesigan)
    fakes.py         fake outputs built from each member's contract example
  routers/
    health.py        GET  /api/health, /api/datasets
    c1.py            GET  /api/c1/engine-info, POST /api/c1/detect
    pipeline.py      POST /api/pipeline/run
```

## The interface (components/base.py)

```python
class Component:
    def load(self): ...                          # load models once at startup
    def run(self, batch, upstream) -> dict: ...  # {output_name: contract_json}
```
`upstream` holds everything produced before you: `context` (dataset, model_id, run_id), `c1`, `c2_to_c3`,
`c2_to_c4`, `c3`. Return one entry per contract you produce (C2 returns both `c2_to_c3` and `c2_to_c4`).

## Plugging in your real component (C2, C3, C4)

1. Put your code in your own folder (`c2/`, `c3/`, `c4/` at the repo root, like `c1/`).
2. In `components/cX.py`: load your models in `load()`, call your code in `run()`, set `source = "real"`.
3. Model and data paths go in `settings.py` as environment variables; the files stay outside the repo.
4. Add your libraries to your own `cX/requirements.txt`; install with `-c c1/requirements.txt` so shared
   libraries (numpy, pandas, scikit-learn, xgboost, torch) keep the pinned versions.
5. Run the backend and `POST /api/pipeline/run`: your step must show `"contract_ok": true`.
6. Edit only your own files; open a PR to `main`.

## Run (Windows, from the repo root, venv active)

```powershell
pip install -r webapp\backend\requirements.txt -c c1\requirements.txt
uvicorn app.main:app --app-dir webapp/backend --port 8000
```
Open http://127.0.0.1:8000/docs. Data folder: `C:\J26\data` by default (set `J26_DATA_DIR` to change).
