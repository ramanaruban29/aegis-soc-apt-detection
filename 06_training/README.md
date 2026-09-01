================================================================================

 Student : Rangan Ramanaruban (RGU ID: 2506630,IIT ID: 20241605)
 Module Code  : CMM799
================================================================================ 
 
# AEGIS SOC — CICAPT-IIoT Model Training Pipeline

This folder contains the canonical training pipeline for the deployed
AEGIS hybrid Autoencoder + LightGBM model, consistent with Chapter 6,
Section 6.5 of the thesis. The four scripts are run in sequence; each
one's output is consumed by the next.

## Dataset

- CICAPT-IIoT Phase 2 (Ghiasvand et al., 2026): 9,536,823 flows, 1,004
  real attacks (9,498:1 imbalance).
- Split: stratified 60/20/20 train/validation/test, random_state=42
  (Section 6.1.3).

## Pipeline (run in order)

### 1. `01_train_autoencoder.py` — Stage 1 (Section 6.5.0)

Trains the dense Autoencoder (63-48-32-16-8-16-32-48-63, with batch
normalisation and dropout) exclusively on benign flows, sampled by
reservoir sampling across the full dataset. Preprocessing: RobustScaler
fitted on benign flows only, features clipped to +/-10 IQR. Computes the
benign reconstruction-error distribution and saves the 99th-percentile
score clip. Includes a separation gate that aborts training if the
attack/benign reconstruction-error ratio falls below 1.5x, guarding
against a degenerate Autoencoder.

**Outputs:** `ae_phase2.h5`, `ae_scaler_phase2.pkl`, `ae_clip_meta.json`
(scale_clip_value, score_clip_p99, separation_ratio).

### 2. `02_generate_stage2_features.py` — Feature Fusion (Section 6.5.0-6.5.1)

Loads the Stage 1 artefacts and streams the full CICAPT-IIoT Phase 2 CSV
in chunks, applying the same RobustScaler, feature clip, and Autoencoder
inference to every flow. Assembles the 64-dimensional fused feature
vector (anomaly_score + 63 scaled raw features), re-verifies the
attack/benign separation on the full dataset, and performs the
stratified 60/20/20 train/validation/test split (Section 6.1.3). Also
writes `feature_names.json` — the single source of truth for the named
64-column schema (`anomaly_score`, `feat_flow_duration`, ...) used by
every downstream training, evaluation, and explainability script.

**Outputs:** `X_train.npy`, `y_train.npy`, `X_val.npy`, `y_val.npy`,
`X_test.npy`, `y_test.npy`, `feature_names.json`.

### 3. `03_train_lgbm.py` — Stage 2 (Sections 6.5.1-6.5.3)

Loads the fused train/validation splits, balances the training set via
50:1 random undersampling of the benign class (Section 6.5.2, resolving
the degenerate scale_pos_weight failure documented in Log Excerpt 6.1),
and trains the LightGBM classifier with scale_pos_weight=1 (since the
data is already balanced by undersampling). Selects the deployment
threshold via an MCC-optimised grid search over 1,000 candidates on the
validation set (Section 6.5.3), also recording the best-F1 and zero-FP
thresholds for comparison.

**Outputs:** `cicapt_lgbm_v2.pkl`, `training_metadata.json` (thresholds,
validation metrics, class counts).

### 4. `04_evaluate.py` — Held-Out Test Evaluation (Section 6.5.4 / Chapter 7)

Loads the trained model and the held-out test split, verifies the saved
feature names match the model's own feature names exactly, and reports
the final test-set metrics at the deployment threshold: confusion
matrix, TPR, FPR, precision, recall, F1, AUC-ROC, AUC-PR, and MCC. Also
saves ROC and Precision-Recall curve data and a naive threshold=0.5
baseline for comparison.

**Outputs:** `final_test_metrics.json`, ROC/PR curve arrays.

## From Staging to Deployment

Scripts 1-4 write to a staging directory during training/evaluation.
Once the held-out test metrics are confirmed (Script 4), the validated
artefacts are copied to the live deployment location used by the
inference service and by every evaluation/explainability script in this
submission:

```bash
cp ae_phase2.h5            application/backend/models/cicapt_autoencoder.h5
cp ae_scaler_phase2.pkl    application/backend/models/cicapt_ae_scaler.pkl
cp ae_clip_meta.json       application/backend/models/ae_clip_meta.json
cp cicapt_lgbm_v2.pkl      application/backend/models/cicapt_lgbm.pkl
cp feature_names.json      application/backend/models/feature_names.json
```

## Deployed Model Summary

| Property | Value |
|---|---|
| Features | 64 (63 raw + anomaly_score) |
| Autoencoder architecture | 63-48-32-16-8-16-32-48-63 (BatchNorm + Dropout) |
| Scaler | RobustScaler (fitted on benign flows only) |
| Imbalance handling | 50:1 random undersampling (no SMOTE) |
| Deployment threshold | 0.9930 (best MCC on validation set) |
| Held-out test recall | 85.57% |
| Held-out test FPR | 0.0025% |
| Held-out test MCC | 0.8198 |

## Re-running

```bash
cd 06_training
python3 01_train_autoencoder.py
python3 02_generate_stage2_features.py
python3 03_train_lgbm.py
python3 04_evaluate.py
```

Each script reads the dataset path and staging directory from constants
near the top of the file; adjust these to your local CICAPT-IIoT CSV
location and preferred output directory before running. Scripts must be
run in order, as each depends on artefacts produced by the previous one.

