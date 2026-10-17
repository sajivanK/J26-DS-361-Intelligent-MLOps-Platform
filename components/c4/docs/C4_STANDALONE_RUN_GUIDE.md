# C4 Standalone Application Run Guide

## Scope
Recorded Adult case assessment using frozen C4 models and simulated upstream evidence.
Evaluation runs N1/N2 inference and the controller against saved pre-release evidence.
It does not retrain the candidate or deploy a model.

## Prerequisites
- Existing verified Python 3.12 environment and C4 dependencies.
- Node.js/npm compatible with the shared frontend.
- Shared webapp and C1 source available locally.
- External C4 artifacts accessible through C4_DATA_ROOT.
- C4 router, page and API registrations present.

The current C4 branch does not yet contain all shared application dependencies.
Fresh-checkout reproducibility remains pending shared-app reconciliation.

## Terminal 1: backend
Run from the repository root:

```powershell
& ".\.venv\Scripts\Activate.ps1"

$env:C4_DATA_ROOT = '\\?\G:\.shortcut-targets-by-id\1xCM4Ry8l8ads1Xn8rBneZ_AavyLqYC3u\J26-DS-361-Datasets'

python -m uvicorn app.main:app --app-dir webapp/backend --port 8000
```

Keep this terminal running.
The dataset path is machine-specific; use the appropriate artifact root on another machine.

## Terminal 2: frontend
Run from the repository root:

```powershell
Set-Location webapp/frontend
npm ci
npm run dev
```

Open http://localhost:5173/c4

## Verification
Run from the repository root with the Python environment active:

```powershell
python -m unittest discover -s webapp/backend/tests/c4 -v

Push-Location webapp/frontend
npm run build
Pop-Location

python webapp/backend/scripts/verify_c4_application.py --save
```

The application verifier requires the backend running and C4_DATA_ROOT set.
It saves timestamped evidence externally and a Markdown summary under components/c4/docs.

## Expected recorded-case output
- Decision: HUMAN_REVIEW.
- Five mandatory gates pass.
- All three future-safety dimensions are uncertain.
- Selected policy: review-only.
- Observed performance is separate from future forecasts.
- Production release remains unauthorized.

## Synthetic verification
Seven acceptance fixtures verify controller behaviour, including all three decision branches.
Synthetic AUTO_DEPLOY is technical verification, not measured deployment-safety evidence.

## Troubleshooting
- Backend unavailable: start it on port 8000, reload the page and retry.
- Artifact unavailable: verify C4_DATA_ROOT and preserve checksum checks.
- Missing C1 engines: shared C1 functionality remains unavailable; recorded C4 assessment is separate.
- Request failure: inspect backend logs; no new release decision is produced.
- A retained result must be identified as a previous successful result.

## Pending integration
After the leader merges webapp into main:
1. Back up local untracked shared files before reconciling branches.
2. Review and apply the C4 registration patch if still required.
3. Verify setup from a fresh checkout with external artifacts.
4. Implement and test actual C1/C2/C3 adapters and operational orchestration.
