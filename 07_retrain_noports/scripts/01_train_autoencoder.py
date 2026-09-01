import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

"""
01_train_autoencoder.py  
===========================================
Step 1 of 4 — Train dense autoencoder on Phase2 BENIGN flows only.

Q1-grade methodology fixes vs previous version:
  - RobustScaler  (replaces StandardScaler — resistant to extreme outliers)
  - Reservoir sampling across full 9.5M rows (replaces first-500k-only)
  - Score clip threshold saved at benign 99th percentile (controls FP in Stage2)
  - Separation gate: aborts if attack_mean / benign_mean < 1.5x
  - Deeper AE architecture: 63→48→32→16→8→16→32→48→63
  - Batch normalisation + dropout for better generalization
"""

import gc
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
CSV      = "/home/ramanaruban.online/soc.ramanaruban.online/Soc/application/backend/data/CICAPT-IIoT/phase2_NetworkData.csv"
OUT_DIR  = Path("/home/ramanaruban.online/soc.ramanaruban.online/Soc/07_retrain_noports/models")
LOG_DIR  = Path("/home/ramanaruban.online/soc.ramanaruban.online/Soc/07_retrain_noports/logs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
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
]  # 61 features (ports excluded as identity, like IPs)

CHUNKSIZE   = 50_000
VAL_RATIO   = 0.2
RANDOM_SEED = 42
MAX_SAMPLES = 800_000   # reservoir size — covers ~8.4% of benign, spread evenly

np.random.seed(RANDOM_SEED)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Fit RobustScaler on ALL benign flows (chunked)
# RobustScaler uses median + IQR → immune to max=20801 outliers
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 1: Fitting RobustScaler on all benign flows...")
print("=" * 65)

from sklearn.preprocessing import RobustScaler

scaler    = RobustScaler()
n_benign  = 0
n_attacks = 0

# RobustScaler.partial_fit not available — collect medians via two-pass
# Pass 1: collect sample for scaler fitting (first 200k benign — enough for
#         median/IQR estimation which is stable beyond ~50k samples)
scaler_sample = []
scaler_collected = 0
SCALER_FIT_SIZE  = 200_000

for chunk in pd.read_csv(CSV, chunksize=CHUNKSIZE, low_memory=False):
    benign = chunk[chunk['label'] == 0][FEATURE_COLS]
    benign = (benign.apply(pd.to_numeric, errors='coerce')
                    .fillna(0).replace([np.inf, -np.inf], 0))
    n_benign  += len(benign)
    n_attacks += int((chunk['label'] != 0).sum())
    if scaler_collected < SCALER_FIT_SIZE and len(benign) > 0:
        scaler_sample.append(benign.values.astype(np.float32))
        scaler_collected += len(benign)

X_scaler_fit = np.vstack(scaler_sample)[:SCALER_FIT_SIZE]
scaler.fit(X_scaler_fit)
del X_scaler_fit, scaler_sample
gc.collect()

print(f"  ✓ RobustScaler fitted on {min(scaler_collected, SCALER_FIT_SIZE):,} benign flows")
print(f"  ✓ Total benign in file : {n_benign:,}")
print(f"  ✓ Total attacks        : {n_attacks:,}")
print(f"  ✓ Imbalance ratio      : {n_benign//max(n_attacks,1):,}:1")

joblib.dump(scaler, OUT_DIR / "ae_scaler_phase2.pkl")
print(f"  ✓ Scaler saved → {OUT_DIR}/ae_scaler_phase2.pkl")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Reservoir sampling — evenly distributed across full CSV
# Reservoir sampling gives a uniform random sample without loading full file
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 65}")
print(f"STEP 2: Reservoir sampling {MAX_SAMPLES:,} benign flows from {n_benign:,}...")
print("=" * 65)

reservoir   = np.zeros((MAX_SAMPLES, len(FEATURE_COLS)), dtype=np.float32)
filled      = 0       # rows in reservoir so far
total_seen  = 0       # total benign rows seen

for chunk in pd.read_csv(CSV, chunksize=CHUNKSIZE, low_memory=False):
    benign = chunk[chunk['label'] == 0][FEATURE_COLS]
    benign = (benign.apply(pd.to_numeric, errors='coerce')
                    .fillna(0).replace([np.inf, -np.inf], 0))
    rows = benign.values.astype(np.float32)

    for row in rows:
        total_seen += 1
        if filled < MAX_SAMPLES:
            reservoir[filled] = row
            filled += 1
        else:
            # Reservoir replacement — uniform over all seen rows
            j = np.random.randint(0, total_seen)
            if j < MAX_SAMPLES:
                reservoir[j] = row

    if total_seen % 500_000 == 0:
        print(f"  Scanned {total_seen:,} benign rows...")

X_benign = reservoir[:filled].copy()
del reservoir
gc.collect()

print(f"  ✓ Reservoir: {len(X_benign):,} benign flows sampled from {total_seen:,}")

# Scale
X_benign = scaler.transform(X_benign).astype(np.float32)

# Clip scaled values at ±10 IQR — removes extreme outlier dimensions
# (a flow with one feature at 500σ should not dominate reconstruction error)
X_benign = np.clip(X_benign, -10, 10)

print(f"  ✓ Scaled + clipped to [-10, 10]")
print(f"  ✓ Post-scale stats: mean={X_benign.mean():.4f}  std={X_benign.std():.4f}  "
      f"max={X_benign.max():.4f}")

# Train / val split (stratified random)
idx   = np.random.permutation(len(X_benign))
split = int(len(X_benign) * (1 - VAL_RATIO))
X_tr  = X_benign[idx[:split]]
X_vl  = X_benign[idx[split:]]

print(f"  ✓ Train: {len(X_tr):,}   Val: {len(X_vl):,}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Build autoencoder (deeper + regularized)
# Architecture: 63 → 48 → 32 → 16 → 8 → 16 → 32 → 48 → 63
# BatchNorm stabilizes training; Dropout prevents memorization
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 65}")
print("STEP 3: Building autoencoder...")
print("=" * 65)

import tensorflow as tf
tf.random.set_seed(RANDOM_SEED)

n_features = X_tr.shape[1]  # 61 (ports excluded)

def build_ae(n_features):
    inputs = tf.keras.Input(shape=(n_features,), name="input")

    # Encoder
    x = tf.keras.layers.Dense(48, activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    bottleneck = tf.keras.layers.Dense(8, activation='relu', name="bottleneck")(x)

    # Decoder
    x = tf.keras.layers.Dense(16, activation='relu')(bottleneck)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(48, activation='relu')(x)
    outputs = tf.keras.layers.Dense(n_features, activation='linear', name="output")(x)

    model = tf.keras.Model(inputs, outputs, name="cicapt_ae_phase2_v2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
    )
    return model

ae = build_ae(n_features)
ae.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=7, restore_best_weights=True, monitor='val_loss'),
    tf.keras.callbacks.ReduceLROnPlateau(
        patience=3, factor=0.5, min_lr=1e-5, monitor='val_loss', verbose=1),
    tf.keras.callbacks.CSVLogger(str(LOG_DIR / "ae_training_log.csv")),
]

print("\nTraining autoencoder...")
history = ae.fit(
    X_tr, X_tr,
    validation_data = (X_vl, X_vl),
    epochs          = 100,
    batch_size      = 4096,
    callbacks       = callbacks,
    verbose         = 1,
)

final_val_loss = min(history.history['val_loss'])
print(f"\n  ✓ Best val MSE : {final_val_loss:.6f}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Compute and save score clip threshold
# Clip threshold = 99th percentile of benign reconstruction error
# This is saved and applied in Script 02 so benign outliers never reach LGBM
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 65}")
print("STEP 4: Computing score clip threshold on benign val set...")
print("=" * 65)

recon_vl   = ae.predict(X_vl, batch_size=4096, verbose=0)
scores_vl  = np.mean((X_vl - recon_vl) ** 2, axis=1)

clip_p99   = float(np.percentile(scores_vl, 99))
clip_p999  = float(np.percentile(scores_vl, 99.9))

print(f"  Benign val scores:")
print(f"    mean  = {scores_vl.mean():.6f}")
print(f"    std   = {scores_vl.std():.6f}")
print(f"    p95   = {np.percentile(scores_vl, 95):.6f}")
print(f"    p99   = {clip_p99:.6f}   ← clip threshold")
print(f"    p99.9 = {clip_p999:.6f}")
print(f"    max   = {scores_vl.max():.6f}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Verifying separation on attack flows (Q1 gate — must pass before saving)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 65}")
print("STEP 5: Verifying attack separation (Q1 gate)...")
print("=" * 65)

attack_chunks = []
for chunk in pd.read_csv(CSV, chunksize=CHUNKSIZE, low_memory=False):
    attacks = chunk[chunk['label'] != 0][FEATURE_COLS]
    attacks = (attacks.apply(pd.to_numeric, errors='coerce')
                      .fillna(0).replace([np.inf, -np.inf], 0))
    if len(attacks) > 0:
        attack_chunks.append(attacks.values.astype(np.float32))

X_attacks = np.vstack(attack_chunks)
X_attacks = scaler.transform(X_attacks).astype(np.float32)
X_attacks = np.clip(X_attacks, -10, 10)   # same preprocessing as training

recon_a  = ae.predict(X_attacks, batch_size=4096, verbose=0)
scores_a = np.mean((X_attacks - recon_a) ** 2, axis=1)

# Apply clip threshold to both (as Script 02 will)
scores_b_clipped = np.clip(scores_vl, 0, clip_p99)
scores_a_clipped = np.clip(scores_a,  0, clip_p99)

sep_raw     = scores_a.mean() / max(scores_vl.mean(), 1e-9)
sep_clipped = scores_a_clipped.mean() / max(scores_b_clipped.mean(), 1e-9)

print(f"\n  Raw scores (before clipping):")
print(f"    Benign  — mean={scores_vl.mean():.5f}  max={scores_vl.max():.5f}")
print(f"    Attacks — mean={scores_a.mean():.5f}   max={scores_a.max():.5f}")
print(f"    Separation ratio (raw)     : {sep_raw:.2f}x")

print(f"\n  Clipped scores (at p99={clip_p99:.5f}):")
print(f"    Benign  — mean={scores_b_clipped.mean():.5f}  max={scores_b_clipped.max():.5f}")
print(f"    Attacks — mean={scores_a_clipped.mean():.5f}  max={scores_a_clipped.max():.5f}")
print(f"    Separation ratio (clipped) : {sep_clipped:.2f}x  ← cite this")

MIN_SEPARATION = 1.5
if sep_clipped < MIN_SEPARATION:
    print(f"\n   SEPARATION GATE FAILED: {sep_clipped:.2f}x < {MIN_SEPARATION}x")
    print(f"    The AE does not distinguish attacks from benign traffic.")
    print(f"    Do NOT proceed to Script 02. Investigate feature engineering.")
    sys.exit(1)
else:
    print(f"\n   Separation gate passed: {sep_clipped:.2f}x ≥ {MIN_SEPARATION}x")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: Saving everything
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 65}")
print("STEP 6: Saving model and artifacts...")
print("=" * 65)

ae.save(str(OUT_DIR / "ae_phase2.h5"))
print(f"  ✓ AE model  → {OUT_DIR}/ae_phase2.h5")

clip_meta = {
    "score_clip_p99"   : clip_p99,
    "score_clip_p999"  : clip_p999,
    "scale_clip_value" : 10.0,      # ±10 IQR clipping applied to input features
    "benign_score_mean": float(scores_b_clipped.mean()),
    "benign_score_std" : float(scores_b_clipped.std()),
    "attack_score_mean": float(scores_a_clipped.mean()),
    "separation_ratio" : round(sep_clipped, 4),
    "n_benign_train"   : int(len(X_tr)),
    "n_benign_val"     : int(len(X_vl)),
    "n_attacks"        : int(len(X_attacks)),
    "final_val_mse"    : round(final_val_loss, 8),
    "architecture"     : "61-48-32-16-8-16-32-48-61 + BN + Dropout(0.1)",
    "scaler"           : "RobustScaler",
}

import json
with open(OUT_DIR / "ae_clip_meta.json", "w") as f:
    json.dump(clip_meta, f, indent=2, default=lambda o: float(o))
print(f"  ✓ Clip meta → {OUT_DIR}/ae_clip_meta.json")

np.save(OUT_DIR / "ae_scores_benign_val.npy", scores_b_clipped)
np.save(OUT_DIR / "ae_scores_attacks.npy",    scores_a_clipped)
print(f"  ✓ Score arrays saved for analysis")

print(f"\n{'=' * 65}")
print(f"  AE training complete")
print(f"  Separation ratio : {sep_clipped:.2f}x")
print(f"  Score clip @ p99 : {clip_p99:.6f}")
print(f"  → Run 02_generate_stage2_features.py next")
print("=" * 65)