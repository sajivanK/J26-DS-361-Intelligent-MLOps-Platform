# Component 3: Google Colab Workflow Guide

**Project:** J26-DS-361 Intelligent MLOps Platform  
**Component:** Component 3 — Adaptive Intelligent Synthetic Data Repair Engine  
**Branch:** `c3-mathusan`  

This directory contains standalone, path-agnostic Google Colab notebooks and utility modules for executing high-throughput experiments for Component 3.

---

## 1. Quickstart in Google Colab (2-Minute Setup)

### Step 1: Open Google Colab
Go to [Google Colab](https://colab.research.google.com).

### Step 2: Select Hardware Accelerator
Navigate to **Runtime > Change runtime type**:
- **Hardware accelerator:** Select **T4 GPU** (or V100/A100).
- Click **Save**.

### Step 3: Clone Repository in Colab
In the first code cell of Colab, run:
```python
# Clone branch c3-mathusan
!git clone -b c3-mathusan https://github.com/sajivanK/J26-DS-361-Intelligent-MLOps-Platform.git
%cd J26-DS-361-Intelligent-MLOps-Platform

# Verify working directory and importability
import sys
from pathlib import Path
if str(Path(".").resolve()) not in sys.path:
    sys.path.insert(0, str(Path(".").resolve()))
print("Repository successfully cloned on branch c3-mathusan!")
```

---

## 2. Notebook Catalog

| Notebook | Purpose | Recommended Hardware | Output Artifacts |
|---|---|---|---|
| `01_colab_baseline_training.ipynb` | Train & evaluate baseline classifiers (XGBoost, RF, LightGBM) across multi-dataset benchmarks | Standard CPU / T4 GPU | Fitted models, preprocessors, evaluation metrics |
| `02_colab_drift_experiments.ipynb` | Inject 8 drift categories and measure model performance degradation | Standard CPU | Drifted datasets, slice labels, drift manifests |
| `03_colab_synthetic_repair.ipynb` | Execute generative models (CTGAN, TVAE, TabDDPM) and 4-stage validation | **T4 GPU** (Mandatory for high-epoch CTGAN/TabDDPM) | JSON validation reports, synthetic batches |
| `04_colab_prescription_training.ipynb` | Train and calibrate the Intelligent Repair Prescription Engine with pre-decision features | Standard CPU / T4 GPU | Prescription model weights, evaluation reports |

---

## 3. Google Drive Directory Structure

When running in Colab, mounting Google Drive persists all training results across runtime disconnects:
```text
My Drive/
└── J26-DS-361-MLOps/
    └── component3/
        ├── data/              <- Dataset CSVs (UCI Adult, Telco, Bank Marketing)
        ├── models/            <- Trained baseline & generative model weights
        ├── checkpoints/       <- Atomic experiment resume state files (pilot_checkpoint.json)
        ├── results/           <- Experiment JSON reports (compatible with Sprint 05 contract)
        ├── figures/           <- Publication-quality evaluation plots
        └── exports/           <- Packaged zip archives for git transfer
```

---

## 4. Hardware & GPU Acceleration Guidelines

1. **CTGAN:** Benefits significantly from CUDA tensor acceleration during discriminator/generator alternating gradient updates.
2. **TabDDPM:** Requires CUDA acceleration for iterative reverse diffusion sampling (100–1000 timesteps).
3. **TVAE:** Runs efficiently on CPU (~2 seconds) or GPU.
4. **Validation Engine:** Multi-stage validation runs on CPU (scipy, numpy, scikit-learn).

---

## 5. Transferring Artifacts Back to Component 3

1. In Notebook `03_colab_synthetic_repair.ipynb`, execute the final cell calling `plot_colab_summary()` and exporting the results.
2. Download the resulting experiment JSON reports from Google Drive (`results/repair_experiments/*.json`).
3. Place them into `component3/results/repair_experiments/`.
4. Run:
```bash
python component3/src/experiments/plot_pilot_results.py
```
This will automatically generate the 4 panel-ready PNG figures in `component3/results/figures/`.
