# Component 3: Adaptive Data Repair

Component 3 receives C1 drift evidence, C2 diagnosis, and the configured model/data context. It first attempts a validated synthetic repair. If that candidate fails a quality gate or does not beat the authentic control, C3 evaluates an authentic weighted adaptation candidate (`source=1.0`, `target=2.0`). Nothing is approved unless independent utility, paired-bootstrap, source-retention, and overall-retention gates all pass.

## Configure team artifacts

Run this once from the repository root after the team model and UCI data have been placed under the configured data directory:

```powershell
python -m component3.src.configure_team_context
```

The default data root is `C:\J26\data`. To use another location:

```powershell
python -m component3.src.configure_team_context --data-dir D:\path\to\team-data
```

The command checks the required model/datasets and their columns before writing C3's external context file. It does not alter model or dataset artifacts.

## Run the platform

From the repository root, start the backend:

```powershell
python -m uvicorn app.main:app --app-dir webapp/backend --port 8000
```

In a second terminal, start the frontend:

```powershell
Set-Location webapp/frontend
npm install
npm run dev
```

Open `http://localhost:5173`. API documentation is at `http://127.0.0.1:8000/docs`.

## Decision meanings

- `APPROVED` + `TARGETED_SYNTHETIC_REPAIR`: the synthetic candidate passed every quality and utility gate and beat the authentic weighted candidate.
- `APPROVED` + `AUTHENTIC_WEIGHTED_ADAPTATION`: synthetic repair was unsuitable, but weighted real labelled adaptation passed every utility and retention gate.
- `REJECTED`: both candidate paths failed; reasons for each are retained.
- `BLOCKED`: required labels, diagnosis stability, slice support, model compatibility, or independent evaluation evidence were unavailable.
- `ABSTAINED_NO_REPAIR`: C1 predicted no meaningful harm.

An approval means “ready for C4 review”; it is not a production deployment.
