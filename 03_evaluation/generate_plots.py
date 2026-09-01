#!/usr/bin/env python3
"""
================================================================================
 AEGIS SOC — Results Plots Generator  (for Chapter 7)
================================================================================
 Generates FOUR publication-quality plots from the deployed model, all at once:

 Uses the SAME held-out TEST set as Table 7.1, so the plots match your reported
 numbers (Recall 85.57%, FPR 0.0025%, MCC 0.8198).

 SETUP:  pip install matplotlib seaborn
 RUN:    python3 generate_plots.py
================================================================================
"""
import warnings, os, json
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import joblib
import pyarrow.parquet as pq
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             average_precision_score, confusion_matrix)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── PATHS ──
CSV         = '/home/ramanan/apt_project/data/raw/phase2/phase2_NetworkData.csv'
PARQUET     = '/home/ramanan/apt_project/data/processed/phase2/train_normal.parquet'
AE_PATH     = '/home/ramanan/SOC_New/backend/models/cicapt_autoencoder.h5'
SCALER_PATH = '/home/ramanan/SOC_New/backend/models/cicapt_ae_scaler.pkl'
CLIP_META   = '/home/ramanan/SOC_New/backend/models/ae_clip_meta.json'
LGBM_PATH   = '/home/ramanan/SOC_New/backend/models/cicapt_lgbm.pkl'

AE_HISTORY_CANDIDATES = [
    '/home/ramanan/SOC_New/backend/models/ae_history.json',
    '/home/ramanan/SOC_New/backend/models/cicapt_ae_history.json',
    '/home/ramanan/SOC_New/training/ae_history.json',
]

CHUNK_SIZE   = 200_000
RANDOM_STATE = 42
AE_BATCH     = 512
THRESHOLD    = 0.9930           # deployment threshold
OUT_DIR      = 'plots'
os.makedirs(OUT_DIR, exist_ok=True)

# Consistent styling
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 150,
})
DEEP='#065A82'; TEAL='#1C7293'; ACCENT='#02C39A'; CRED='#C0392B'; GREY='#5A6472'

# ── 1. rebuilding held-out TEST set ──
print("[1/6] Reading labels + test split...")
labels = []
for chunk in pd.read_csv(CSV, usecols=['label'], chunksize=CHUNK_SIZE, low_memory=False):
    labels.append(chunk['label'].values)
y_all = np.concatenate(labels); del labels
idx = np.arange(len(y_all))
_, test_idx = train_test_split(idx, test_size=0.2, stratify=y_all, random_state=RANDOM_STATE)
test_sorted = np.sort(test_idx)
print(f"    test rows: {len(test_idx):,}  attacks: {int(y_all[test_idx].sum())}")

# ── 2. load*ing artefacts ──
print("[2/6] Loading scaler + AE + LightGBM...")
ae = tf.keras.models.load_model(AE_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)
with open(CLIP_META) as f:
    cm = json.load(f)
feat_clip  = float(cm.get('scale_clip_value', 10.0))
score_clip = float(cm.get('score_clip_p99', 1.0))
FLOW_COLS = [c for c in pq.read_table(PARQUET).schema.names if c != 'ts']
obj = joblib.load(LGBM_PATH)
lgbm = obj['model'] if isinstance(obj, dict) else obj

# ── 3. stream test set, collect y_true + probabilities ──
print("[3/6] Scoring test set (probabilities)...")
y_true_parts, prob_parts = [], []
row_offset = 0
for chunk in pd.read_csv(CSV, chunksize=CHUNK_SIZE, low_memory=False):
    g_idx = np.arange(row_offset, row_offset + len(chunk))
    mask = np.isin(g_idx, test_sorted, assume_unique=True)
    row_offset += len(chunk)
    if not mask.any():
        continue
    sub = chunk.loc[mask]
    y = sub['label'].values.astype(int)
    X_raw = pd.DataFrame({c: pd.to_numeric(sub[c], errors='coerce').fillna(0)
                          if c in sub.columns else 0.0 for c in FLOW_COLS})
    X_raw = X_raw.replace([np.inf, -np.inf], 0).fillna(0).values.astype(np.float32)
    Xs = np.clip(scaler.transform(X_raw), -feat_clip, feat_clip).astype(np.float32)
    recon = ae.predict(Xs, batch_size=AE_BATCH, verbose=0)
    score = np.clip(np.mean((Xs - recon)**2, axis=1), 0.0, score_clip).astype(np.float32)
    X64 = np.column_stack([score, Xs]).astype(np.float32)
    prob = lgbm.predict_proba(X64)[:, 1]
    y_true_parts.append(y); prob_parts.append(prob)
    del sub, X_raw, Xs, recon, score, X64
y_true = np.concatenate(y_true_parts)
y_prob = np.concatenate(prob_parts)
y_pred = (y_prob >= THRESHOLD).astype(int)

# ── 4. ROC curve ──
print("[4/6] Plotting ROC curve...")
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color=DEEP, lw=2.2, label=f'AEGIS (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], '--', color=GREY, lw=1, label='Random')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve — Held-out Test Set', fontweight='bold', color='#21295C')
ax.legend(loc='lower right', frameon=False)
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02]); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(f'{OUT_DIR}/roc_curve.png', bbox_inches='tight'); plt.close(fig)

# ── 5. Precision-Recall curve (primary under imbalance) ──
print("[5/6] Plotting Precision-Recall curve...")
prec, rec, _ = precision_recall_curve(y_true, y_prob)
ap = average_precision_score(y_true, y_prob)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(rec, prec, color=TEAL, lw=2.2, label=f'AEGIS (AP = {ap:.4f})')
baseline = y_true.mean()
ax.axhline(baseline, ls='--', color=GREY, lw=1, label=f'Baseline ({baseline:.5f})')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve — Held-out Test Set\n(primary metric under extreme imbalance)',
             fontweight='bold', color='#21295C', fontsize=11)
ax.legend(loc='center left', frameon=False)
ax.set_xlim([0, 1.02]); ax.set_ylim([0, 1.02]); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(f'{OUT_DIR}/pr_curve.png', bbox_inches='tight'); plt.close(fig)

# ── 6. Confusion matrix heatmap ──
print("[6/6] Plotting confusion matrix + loss curve...")
cm_arr = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm_arr.ravel()
fig, ax = plt.subplots(figsize=(5.4, 4.6))
im = ax.imshow(cm_arr, cmap='Blues', aspect='auto')
labels_cell = [[f'TN\n{tn:,}', f'FP\n{fp:,}'], [f'FN\n{fn:,}', f'TP\n{tp:,}']]
for i in range(2):
    for j in range(2):
        val = cm_arr[i, j]
        colour = 'white' if val > cm_arr.max()/2 else '#1A2233'
        ax.text(j, i, labels_cell[i][j], ha='center', va='center',
                color=colour, fontsize=12, fontweight='bold')
ax.set_xticks([0, 1]); ax.set_xticklabels(['Benign', 'Attack'])
ax.set_yticks([0, 1]); ax.set_yticklabels(['Benign', 'Attack'])
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix — Held-out Test Set', fontweight='bold', color='#21295C')
fig.tight_layout(); fig.savefig(f'{OUT_DIR}/confusion_matrix.png', bbox_inches='tight'); plt.close(fig)

# ── AE training-loss curve (only if history exists) ──
hist_file = next((h for h in AE_HISTORY_CANDIDATES if os.path.exists(h)), None)
if hist_file:
    with open(hist_file) as f:
        hist = json.load(f)
    loss = hist.get('loss', []); val_loss = hist.get('val_loss', [])
    if loss:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(range(1, len(loss)+1), loss, color=DEEP, lw=2, label='Training loss')
        if val_loss:
            ax.plot(range(1, len(val_loss)+1), val_loss, color=CRED, lw=2, label='Validation loss')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Reconstruction Loss (MSE)')
        ax.set_title('Autoencoder Training Loss', fontweight='bold', color='#21295C')
        ax.legend(frameon=False); ax.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(f'{OUT_DIR}/ae_loss_curve.png', bbox_inches='tight'); plt.close(fig)
        print(f"    AE loss curve saved (from {hist_file})")
else:
    print("    NOTE: no AE training-history file found — loss curve skipped.")
    print("          To enable it, save the Keras history during AE training as JSON, e.g.:")
    print("          json.dump(history.history, open('backend/models/ae_history.json','w'))")

print("\n" + "="*60)
print("DONE. Plots written to ./plots/")
print(f"  roc_curve.png          (AUC-ROC = {roc_auc:.4f})")
print(f"  pr_curve.png           (AUC-PR  = {ap:.4f})")
print(f"  confusion_matrix.png   (TP={tp} FP={fp} FN={fn} TN={tn:,})")
print(f"  ae_loss_curve.png      ({'saved' if hist_file else 'skipped — no history file'})")
print("="*60)
print("\nInsert these into Chapter 7. The ROC/PR AUC values here are computed on")
print("the held-out test set and should match your reported figures.")