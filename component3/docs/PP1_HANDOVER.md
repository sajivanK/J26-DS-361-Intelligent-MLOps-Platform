# Component 3: Synthetic Data Repair Engine — PP1 Handover Guide
**Project:** Intelligent MLOps Platform (J26-DS-361)  
**Component:** Component 3 — Adaptive Intelligent Synthetic Data Repair Engine  
**Lead / Owner:** Mathusan K (IT23174580)  
**Git Branch:** `c3-mathusan`  
**Milestone:** PP1 (Progress Presentation 1)  

---

## 1. Project Overview & Component Scope

Component 3 is the **Synthetic Data Repair Engine** of the Intelligent MLOps Platform. When incoming production data drifts and is diagnosed as harmful, Component 3 is triggered to restore model performance without requiring costly human re-labeling or waiting for delayed production ground truth.

### Position in Research Pipeline
1. **C1 (Drift Detection):** Identifies covariate/concept drift and quantifies harm severity (predicted drop in accuracy/F1).
2. **C2 (Root Cause Diagnosis):** Pinpoints the root cause features and defines the specific `affected_slice_rule` (e.g., `Contract == 'Month-to-month' and tenure <= 12`).
3. **C3 (Synthetic Data Repair - Ours):**
   - Ingests diagnosed slice rule and drift severity.
   - Evaluates data context and prescribes optimal generative model (TVAE, TabDDPM, CTGAN).
   - Generates targeted synthetic candidate samples for the affected slice.
   - Rigorously validates candidates across 4 quality gates (Structural, Fidelity KS, Privacy DCR, Retraining Utility).
   - Produces an approved or rejected repair package adhering strictly to `shared/contracts/c3_output_schema.json`.
4. **C4 (Release Control):** Consumes approved repair package for candidate model retraining and safety gate evaluation before canary release.

---

## 2. Architecture & Data Flow

```
[Upstream C1 & C2 Evidence]
            │
            ▼
┌────────────────────────────────────────────────────────┐
│ External Contract Ingestion & Validation               │
│ (safe AST slice parsing, numeric bounds, security)     │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Adaptive Prescription Engine                           │
│ - Rule-based & Learned Ranker (14.8% lower regret)     │
│ - Selects Generator: TVAE, TabDDPM, or CTGAN           │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Generative Sampling on Diagnosed Slice                 │
│ - Conditioned on affected slice features               │
│ - Evaluates candidate quantities (K=25, 50, 100, 200)  │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Multi-Stage Quality Gate Audit                         │
│ 1. Structural Integrity Gate (Column schema, types)    │
│ 2. Fidelity Gate (Kolmogorov-Smirnov distance <= 0.35) │
│ 3. Privacy Gate (DCR 5th-percentile >= 0.05, 0 copies) │
│ 4. Utility Gate (Independent slice F1 gain >= +0.0100) │
└───────────────────────────┬────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
      [ALL PASS]                      [ANY FAIL]
┌───────────────────────────┐   ┌───────────────────────────┐
│ Status: APPROVED          │   │ Status: REJECTED          │
│ Minimum Effective Repair  │   │ Synthetic rows: 0         │
│ Dataset URI: Available    │   │ Dataset URI: "NONE"       │
└─────────────┬─────────────┘   └─────────────┬─────────────┘
              └───────────────┬───────────────┘
                              │
                              ▼
        ┌───────────────────────────────────────────┐
        │ C4 Retraining Adapter                     │
        │ Flat 12-Field Contract (JSON Schema v1.0) │
        └───────────────────────────────────────────┘
```

---

## 3. Installation & Verified Environment Setup

### Prerequisites
- Python 3.11+ (Python 3.13 verified on Windows 11)
- Node.js 20+ and npm 10+
- Git branch: `c3-mathusan`

### Backend Setup (Repository Root)
```powershell
# Activate project virtual environment
.\.venv\Scripts\Activate.ps1

# Install webapp requirements with C1 pinned constraints
pip install -r webapp\backend\requirements.txt -c c1\requirements.txt

# Verify C3 package imports
python -c "import component3; print('C3 Package loaded successfully')"
```

### Frontend Setup (`webapp/frontend`)
```powershell
cd webapp\frontend
npm install
npm run build
```

---

## 4. How to Run the Platform & Component 3

### Terminal 1: Backend Server (FastAPI on Port 8000)
Run from the repository root:
```powershell
uvicorn app.main:app --app-dir webapp/backend --port 8000 --reload
```
- API Documentation: http://127.0.0.1:8000/docs
- Health status: http://127.0.0.1:8000/api/health
- C3 Info: http://127.0.0.1:8000/api/c3/info

### Terminal 2: Frontend Web Application (Vite on Port 5173)
Run from `webapp/frontend`:
```powershell
npm run dev
```
- Open browser at: http://localhost:5173
- Component 3 Dedicated UI: http://localhost:5173/c3

---

## 5. Component 3 API Endpoints

| Method | Path | Description | Response / Behavior |
|---|---|---|---|
| `GET` | `/api/c3/info` | Engine operational status & capabilities | JSON with generators, decision modes, quality gates |
| `GET` | `/api/c3/benchmarks` | Committed research evaluation results | TVAE vs TabDDPM vs CTGAN comparison metrics |
| `POST` | `/api/c3/demo/repair` | Verified demonstration repair | Generates validated 12-field contract response |
| `POST` | `/api/c3/repair` | Live repair from uploaded batch CSV | Validates against `c3_output_schema.json` |
| `GET` | `/api/c3/download/{file}` | Safe CSV download of approved synthetic data | Streaming CSV response (traversal protected) |

### Sample Contract Response (12 Flat Fields)
```json
{
  "repair_dataset_uri": "component3/results/runs/manual/ddd9525e49b8459e899441ef7350abff/pkg-c225c1fcf77a_candidates_k25.csv",
  "repair_package_status": "APPROVED",
  "generator": "TVAE",
  "n_synthetic_rows": 25,
  "augmentation_ratio": 0.0071,
  "fidelity_score": 0.791,
  "utility_score": 0.0002,
  "privacy_status": "PASS",
  "target_coverage": 1.0,
  "prescription_confidence": 0.95,
  "repair_version": "c3-demo-164315",
  "repair_timestamp": "2026-10-22T11:13:15Z"
}
```

---

## 6. Frontend Implementation & UI Guide Compliance

The Component 3 user interface in `webapp/frontend/src/pages/c3/C3Page.jsx` strictly follows all rules in `webapp/UI_GUIDE.md`:
1. **Design Tokens:** Zero hardcoded hex colors. Exclusively uses `var(--accent)`, `var(--bg)`, `var(--surface)`, `var(--line)`, `var(--ok-*)`, `var(--warn-*)`, `var(--bad-*)`.
2. **Typography:** IBM Plex Sans for labels and text, IBM Plex Mono for numbers and metrics.
3. **8-Part Layout:**
   - **PageHeader:** Eyebrow `Component 3 · Mathusan K (IT23174580)`, title, subtitle, and version chip.
   - **Run Bar:** Dataset selector (`telco_churn` / `uci`), execution mode toggle, file upload, and primary action buttons.
   - **Answer Banner:** `Verdict` (APPROVED / REJECTED) with 3-column Hero (`Distribution Fidelity`, `Retraining Utility`, `Slice Coverage`).
   - **Why It Matters:** 4-card Quality Gate Validation Audit (Structural, Fidelity KS <= 0.35, Privacy DCR >= 0.05, Utility Gain >= +0.0100).
   - **Session History:** Interactive SVG timeline tracking all runs executed in current tab session.
   - **Why Trust It:** Committed research benchmark charts comparing TVAE, TabDDPM, and CTGAN across slice coverage, KS distance, latency, and minimum effective volume search.
   - **Details:** Operational specifications and configuration KV cards.
   - **Raw Output:** Collapsible `JsonBox` containing the exact JSON payload passed to C4.
4. **Namespace Isolation:** All CSS classes prefixed with `.c3-` (`.c3-hero`, `.c3-gate-card`, `.c3-runbar`, etc.).
5. **No External Chart Libraries:** Pure SVG charts (`C3Charts.jsx`) rendering responsive vector visualizations.

---

## 7. Automated Testing & Verification

Execute the complete test suite:
```powershell
# External contracts, schema validation, and server tests
pytest component3/tests/test_external_contracts.py component3/tests/test_server.py component3/tests/test_webapp_c3_router.py -v

# Frontend production build verification
cd webapp\frontend
npm run build
```

---

## 8. Reproducible PP1 Demonstration Guide

Follow these exact steps during the PP1 presentation to demonstrate Component 3:

1. **Open Frontend:**
   - Navigate to http://localhost:5173.
   - Note the left sidebar: Component 3 shows a green/teal `real` badge with title **C3 Synthetic Data Repair** and owner **Mathusan K**.
2. **Navigate to Component 3 Page:**
   - Click **C3 Synthetic Data Repair** (`/c3`).
   - Point out the 8-part UI layout matching the team's standardized design system.
3. **Execute Verified Demonstration Repair:**
   - In the Run Bar, ensure **Telco Churn** and **Verified Demonstration** are selected.
   - Click **Run C3 Demo Repair**.
   - Observe the live Verdict banner change to **APPROVED** (green).
   - Walk the panel through the 3 hero metrics:
     - **Fidelity: 79.1%** (high multivariate feature similarity).
     - **Retraining Utility: +3.8 pp F1** (unseen test slice performance recovery).
     - **Slice Coverage: 100%** (diagnosed contract rule satisfaction).
4. **Show Quality Gate Validation:**
   - Highlight the 4 validation cards: Structural (PASS), Fidelity KS (PASS), Privacy DCR (PASS, 0 copies), Utility (PASS).
5. **Demonstrate Synthetic CSV Download:**
   - Click **Download Synthetic CSV** in the approved banner.
   - Show that the downloaded file is a real candidate dataset generated for the diagnosed slice.
6. **Show Benchmark Evidence ("Why Trust It"):**
   - Point to the SVG generator comparison chart showing why TVAE was prescribed over CTGAN and TabDDPM.
   - Point to the Minimum Effective Repair curve showing that K=100 achieves optimal performance without dataset inflation.
7. **Show Raw Contract Output:**
   - Expand the bottom **Raw output (JSON, contract v1.0)** section.
   - Show that the payload contains exactly the 12 fields required by Kesigan's C4 contract.
8. **Demonstrate Pipeline Integration:**
   - Navigate to **Drift Lab** (`/lab`).
   - Run the full pipeline.
   - Show that Component 3 card displays **real** and **contract ok** alongside C1, C2, and C4.

---

## 9. Scientific Limitations & Future C4 Integration

### Known Limitations
1. **Sample Size for Online Training:** Full online training of TabDDPM or CTGAN from scratch requires 15–30 seconds. For real-time interactive demo, cached pre-generated candidate datasets are utilized.
2. **Multi-Label Concept Drift:** Current repair engine targets single or multi-feature covariate drift on tabular classification. Concept drift with complete posterior inversion requires active human re-annotation.

### Handover to C4 Owner (Kesigan M)
- **Contract Interface:** C3 outputs the exact 12 flat fields specified in `shared/contracts/c3_output_schema.json`.
- **Approved Dataset Location:** When `repair_package_status == "APPROVED"`, `repair_dataset_uri` points to the valid CSV file ready to be concatenated with the training set.
- **When Rejected:** When `repair_package_status == "REJECTED"`, `repair_dataset_uri` is `"NONE"` and `n_synthetic_rows` is 0. C4 should trigger `ESCALATE` or `HUMAN_REVIEW` without retraining.
