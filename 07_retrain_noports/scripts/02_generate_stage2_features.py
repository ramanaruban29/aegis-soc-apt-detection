import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

"""
02_generate_stage2_features.py
=======================================================
Step 2 of 4 — Generate 64-feature matrix for LightGBM.

  - DROPPED fake roll columns (roll_mean/roll_std/roll_max/roll_min/
    score_diff/abs_diff).  Those were just copies of anomaly_score or zeros
    because the chunk loop has no cross-chunk memory — they caused LightGBM
    to early-stop at iteration 2.
  - Feature matrix is now 64 columns: anomaly_score + 63 scaled raw features.
  - feature_names.json written here (not in Script 03) so the column list is
    defined once and shared by all downstream scripts.
  - Stratified 60/20/20 split with verification (unchanged).
"""

import gc
import sys
import json
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split

# ── Paths ─────────────────────────────────────────────────────────────────────
CSV       = "/home/ramanaruban.online/soc.ramanaruban.online/Soc/application/backend/data/CICAPT-IIoT/phase2_NetworkData.csv"
MODEL_DIR = Path("/home/ramanaruban.online/soc.ramanaruban.online/Soc/07_retrain_noports/models")
LOG_DIR   = Path("/home/ramanaruban.online/soc.ramanaruban.online/Soc/07_retrain_noports/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

RAW_FEATURE_COLS = [
    'flow_duration', 'Header_Length',
    'Protocol Type', 'Duration', 'Rate', 'Srate', 'Drate',
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
    'psh_flag_number', 'ack_flag_number', 'urg_flag_number',
    'ece_flag_number', 'cwr_flag_number', 'ack_count', 'syn_count',
    'fin_count', 'urg_count', 'rst_count', 'max_duration',
    'min_duration', 'sum_duration', 'average_duration', 'std_duration',
    'CoAP', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC',
    'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP', 'IPv', 'LLC',
    'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT',
    'Number', 'MAC', 'Magnitue', 'Radius', 'Covariance', 'Variance',
    'Weight', 'DS status', 'Fragments', 'Sequence number',
    'Protocol Version', 'flow_idle_time', 'flow_active_time',
]  # 61 features (ports excluded as identity)

# 62-column feature names: 1 AE score + 61 scaled raw features
FEATURE_NAMES = ['anomaly_score'] + [f'feat_{c.replace(" ", "_")}' for c in RAW_FEATURE_COLS]
assert len(FEATURE_NAMES) == 62

CHUNKSIZE   = 50_000
AE_BATCH    = 2048
RANDOM_SEED = 42

# ── Pre-flight checks ─────────────────────────────────────────────────────────
print("Pre-flight checks...")
for p, name in [
    (MODEL_DIR / "ae_scaler_phase2.pkl", "RobustScaler"),
    (MODEL_DIR / "ae_phase2.h5",         "AE model"),
    (MODEL_DIR / "ae_clip_meta.json",    "Clip metadata"),
    (Path(CSV),                           "Dataset CSV"),
]:
    if not p.exists():
        print(f"  ✗ Missing: {name} → {p}")
        sys.exit(1)
    print(f"  ✓ Found: {name}")

# ── Load artifacts ────────────────────────────────────────────────────────────
print("\nLoading artifacts...")
scaler = joblib.load(MODEL_DIR / "ae_scaler_phase2.pkl")
ae     = tf.keras.models.load_model(str(MODEL_DIR / "ae_phase2.h5"), compile=False)

with open(MODEL_DIR / "ae_clip_meta.json") as f:
    clip_meta = json.load(f)

SCORE_CLIP = clip_meta["score_clip_p99"]
FEAT_CLIP  = clip_meta["scale_clip_value"]

print(f"   AE input shape  : {ae.input_shape}")
print(f"   Score clip (p99): {SCORE_CLIP:.6f}")
print(f"   Feature clip    : ±{FEAT_CLIP}")
print(f"   Separation ratio: {clip_meta['separation_ratio']}x")
print(f"   Output features : {len(FEATURE_NAMES)}  (anomaly_score + 63 scaled raw)")

# ── Count rows ────────────────────────────────────────────────────────────────
print("\nCounting rows...")
total_rows = sum(1 for _ in open(CSV)) - 1
print(f"  Total rows: {total_rows:,}")

# ── Process CSV in chunks ─────────────────────────────────────────────────────
print("\nProcessing CSV in chunks...")

tmp_X = MODEL_DIR / "_tmp_X.npy"
tmp_y = MODEL_DIR / "_tmp_y.npy"

rows_done   = 0
first_chunk = True

for i, chunk in enumerate(pd.read_csv(CSV, chunksize=CHUNKSIZE, low_memory=False)):

    missing = [c for c in RAW_FEATURE_COLS + ['label'] if c not in chunk.columns]
    if missing:
        print(f"  ! Chunk {i}: missing {missing} — skipping")
        continue

    X_raw = (chunk[RAW_FEATURE_COLS]
             .apply(pd.to_numeric, errors='coerce')
             .fillna(0).replace([np.inf, -np.inf], 0))
    X_arr = X_raw.values.astype(np.float32)
    y_arr = (chunk['label'] != 0).astype(np.uint8).values
    del X_raw

    # 1. RobustScaler
    X_scaled = scaler.transform(X_arr).astype(np.float32)
    del X_arr

    # 2. Feature clip ±10 IQR
    X_scaled = np.clip(X_scaled, -FEAT_CLIP, FEAT_CLIP)

    # 3. AE inference
    recon_parts = []
    for start in range(0, len(X_scaled), AE_BATCH):
        recon_parts.append(
            ae.predict(X_scaled[start:start + AE_BATCH], verbose=0))
    recon = np.vstack(recon_parts).astype(np.float32)
    del recon_parts

    # 4. Anomaly score
    anomaly_scores = np.mean((X_scaled - recon) ** 2, axis=1).astype(np.float32)
    del recon

    # 5. Score clip at benign p99
    anomaly_scores = np.clip(anomaly_scores, 0, SCORE_CLIP).astype(np.float32)

    # 6. Assembling 64-feature matrix: [anomaly_score | 63 scaled raw features]
    #    LightGBM to early-stop at iteration 2.
    X_64 = np.column_stack([
        anomaly_scores,  # col 0
        X_scaled,        # cols 1–63
    ]).astype(np.float32)
    del X_scaled, anomaly_scores

   
    if first_chunk:
        np.save(tmp_X, X_64)
        np.save(tmp_y, y_arr)
        first_chunk = False
    else:
        prev_X = np.load(tmp_X)
        prev_y = np.load(tmp_y)
        np.save(tmp_X, np.vstack([prev_X, X_64]))
        np.save(tmp_y, np.concatenate([prev_y, y_arr]))
        del prev_X, prev_y

    del X_64, y_arr
    gc.collect()

    rows_done += len(chunk)
    if (i + 1) % 5 == 0 or rows_done >= total_rows:
        attacks = int(np.load(tmp_y).sum())
        pct     = 100 * rows_done / total_rows
        print(f"  [{pct:5.1f}%] {rows_done:>10,} / {total_rows:,} | attacks: {attacks:,}")

# ── Load assembled arrays ─────────────────────────────────────────────────────
print("\nLoading assembled arrays...")
X_full = np.load(tmp_X)
y_full = np.load(tmp_y)
tmp_X.unlink(missing_ok=True)
tmp_y.unlink(missing_ok=True)

print(f"  ✓ Full dataset : {X_full.shape}  (expected N × 64)")
print(f"    Benign       : {(y_full == 0).sum():,}")
print(f"    Attacks      : {(y_full == 1).sum():,}")
print(f"    Imbalance    : {(y_full==0).sum() // max((y_full==1).sum(), 1):,}:1")

assert X_full.shape[1] == 62, f"Expected 62 cols, got {X_full.shape[1]}"

# ── Verify anomaly score separation ──────────────────────────────────────────
ae_scores = X_full[:, 0]
b_mean    = ae_scores[y_full == 0].mean()
a_mean    = ae_scores[y_full == 1].mean()
sep       = a_mean / max(b_mean, 1e-9)
print(f"\n  Post-clip separation check:")
print(f"    Benign  score mean : {b_mean:.6f}")
print(f"    Attack  score mean : {a_mean:.6f}")
print(f"    Separation ratio   : {sep:.2f}x")

if sep < 1.5:
    print(f"  ✗ Separation insufficient ({sep:.2f}x) — retrain AE")
    sys.exit(1)
print(f"  ✓ Separation OK")

if (y_full == 1).sum() == 0:
    print("  ✗ No attacks found — check label column")
    sys.exit(1)

# ── Saving feature names (single source of truth for Scripts 03 & 04) ──────────
with open(MODEL_DIR / "feature_names.json", "w") as f:
    json.dump(FEATURE_NAMES, f)
print(f"\n  ✓ feature_names.json saved → {MODEL_DIR}/feature_names.json")

# ── Stratified 60/20/20 split ─────────────────────────────────────────────────
print("\nCreating stratified 60/20/20 split...")

X_tv, X_test, y_tv, y_test = train_test_split(
    X_full, y_full, test_size=0.2,
    random_state=RANDOM_SEED, stratify=y_full)
del X_full, y_full
gc.collect()

X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv, test_size=0.25,
    random_state=RANDOM_SEED, stratify=y_tv)
del X_tv, y_tv
gc.collect()

print(f"  Train : {X_train.shape} | attacks: {y_train.sum():,}")
print(f"  Val   : {X_val.shape}   | attacks: {y_val.sum():,}")
print(f"  Test  : {X_test.shape}  | attacks: {y_test.sum():,}")

# ── Saving splits ───────────────────────────────────────────────────────────────
print(f"\nSaving splits to {MODEL_DIR}...")
for arr, name in [
    (X_train, "X_train"), (y_train, "y_train"),
    (X_val,   "X_val"),   (y_val,   "y_val"),
    (X_test,  "X_test"),  (y_test,  "y_test"),
]:
    p = MODEL_DIR / f"{name}.npy"
    np.save(p, arr)
    print(f"  ✓ {name}.npy  ({p.stat().st_size/1e6:.1f} MB)")

print("\n✓ Done — run 03_train_lgbm.py next")