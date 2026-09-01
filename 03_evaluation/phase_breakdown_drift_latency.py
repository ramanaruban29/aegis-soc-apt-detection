"""
ISOLATED diagnostic — three additional ML testing items:
  A. Confusion matrix / false-negative breakdown by attack phase (subLabel)
  B. Temporal drift check — first half vs second half of the test stream
  C. Single-flow inference latency (not batch throughput)
"""
import warnings, os, json, time
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np, pandas as pd, joblib
import pyarrow.parquet as pq
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, confusion_matrix

# ---- Inputs: READ ONLY, never modified ----
CSV         = '/home/ramanan/apt_project/data/raw/phase2/phase2_NetworkData.csv'
PARQUET     = '/home/ramanan/apt_project/data/processed/phase2/train_normal.parquet'
AE_PATH     = '/home/ramanan/SOC_New/backend/models/cicapt_autoencoder.h5'
SCALER_PATH = '/home/ramanan/SOC_New/backend/models/cicapt_ae_scaler.pkl'
LGBM_PATH   = '/home/ramanan/SOC_New/backend/models/cicapt_lgbm.pkl'
CLIP_META   = '/home/ramanan/SOC_New/backend/models/ae_clip_meta.json'
CHUNK_SIZE  = 200_000

# ---- Output: NEW, isolated folder only ----
OUT_DIR = Path("/home/ramanan/thesis_extra_tests")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("[1/7] Pass 1 — reading only label column to compute the split...")
labels = []
for chunk in pd.read_csv(CSV, usecols=['label'], chunksize=CHUNK_SIZE, low_memory=False):
    labels.append(chunk['label'].values)
y_all = np.concatenate(labels)
n_total = len(y_all)
idx_all = np.arange(n_total)
trainval_idx, test_idx = train_test_split(idx_all, test_size=0.2, stratify=y_all, random_state=42)
train_idx, val_idx = train_test_split(trainval_idx, test_size=0.25, stratify=y_all[trainval_idx], random_state=42)
print(f"    Test split: {len(test_idx):,} rows, {int(y_all[test_idx].sum())} attacks")

print("[2/7] Loading model artefacts (read-only)...")
ae = tf.keras.models.load_model(AE_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)
lgbm = joblib.load(LGBM_PATH)
with open(CLIP_META) as f:
    clip_meta = json.load(f)
feat_clip = float(clip_meta.get('scale_clip_value', 10.0))
score_clip = float(clip_meta.get('score_clip_p99', 1.0))
FLOW_COLS = [c for c in pq.read_table(PARQUET).schema.names if c != 'ts']

print("[3/7] Pass 2 — streaming CSV, keeping test-split rows + subLabel...")
test_rows = []
row_offset = 0
for chunk in pd.read_csv(CSV, chunksize=CHUNK_SIZE, low_memory=False):
    chunk_global_idx = np.arange(row_offset, row_offset + len(chunk))
    mask = np.isin(chunk_global_idx, test_idx)
    if mask.any():
        sub = chunk.loc[mask].copy()
        sub['_global_idx'] = chunk_global_idx[mask]  # preserve original row order for temporal split
        test_rows.append(sub)
    row_offset += len(chunk)
    if row_offset % 2_000_000 < CHUNK_SIZE:
        print(f"    ...scanned {row_offset:,} / {n_total:,} rows")

test_df = pd.concat(test_rows, ignore_index=True)
del test_rows
test_df = test_df.sort_values('_global_idx').reset_index(drop=True)
y_test = test_df['label'].values
sub_label_col = 'subLabel' if 'subLabel' in test_df.columns else None
print(f"    Collected {len(test_df):,} test rows, {int(y_test.sum())} attacks; subLabel column found: {sub_label_col is not None}")

print("[4/7] Running the verified 64-feature pipeline: scale -> clip -> AE -> score clip -> fuse -> predict...")
X_raw = pd.DataFrame({c: pd.to_numeric(test_df[c], errors='coerce').fillna(0) if c in test_df.columns else 0.0 for c in FLOW_COLS})
X_raw = X_raw.replace([np.inf, -np.inf], 0).fillna(0).values.astype(np.float32)
X_scaled = scaler.transform(X_raw).astype(np.float32)
X_scaled = np.clip(X_scaled, -feat_clip, feat_clip).astype(np.float32)
recon = ae.predict(X_scaled, batch_size=512, verbose=0)
anomaly_score = np.mean((X_scaled - recon) ** 2, axis=1).astype(np.float32)
anomaly_score = np.clip(anomaly_score, 0.0, score_clip).astype(np.float32)
X_64 = np.column_stack([anomaly_score, X_scaled]).astype(np.float32)
THRESHOLD = 0.993006
probs = lgbm.predict_proba(X_64)[:, 1]
preds = (probs >= THRESHOLD).astype(int)

results = {}

# ---- A. Confusion matrix / FN breakdown by attack phase ----
print("[5/7] Task A — false-negative breakdown by attack phase...")
if sub_label_col:
    fn_mask = (y_test == 1) & (preds == 0)
    tp_mask = (y_test == 1) & (preds == 1)
    fn_phases = test_df.loc[fn_mask, sub_label_col].value_counts().to_dict()
    tp_phases = test_df.loc[tp_mask, sub_label_col].value_counts().to_dict()
    print("    False negatives by phase:", fn_phases)
    print("    True positives by phase:", tp_phases)
    results['confusion_by_phase'] = {'false_negatives': fn_phases, 'true_positives': tp_phases}
else:
    print("    subLabel column not found — skipping phase breakdown")
    results['confusion_by_phase'] = None

# ---- B. Temporal drift check ----
print("[6/7] Task B — temporal drift: first half vs second half of test stream...")
midpoint = len(test_df) // 2
halves = {'first_half': (0, midpoint), 'second_half': (midpoint, len(test_df))}
drift_results = {}
for name, (start, end) in halves.items():
    y_h = y_test[start:end]
    p_h = preds[start:end]
    tn, fp, fn, tp = confusion_matrix(y_h, p_h, labels=[0, 1]).ravel()
    mcc = matthews_corrcoef(y_h, p_h) if len(set(y_h)) > 1 else float('nan')
    tpr = tp / (tp + fn) if (tp + fn) else float('nan')
    fpr = fp / (fp + tn) if (fp + tn) else float('nan')
    drift_results[name] = {'n': int(end - start), 'attacks': int(y_h.sum()), 'tp': int(tp), 'fp': int(fp),
                            'fn': int(fn), 'tn': int(tn), 'tpr': tpr, 'fpr': fpr, 'mcc': mcc}
    print(f"    {name}: n={end-start}, attacks={int(y_h.sum())}, TPR={tpr:.2%}, FPR={fpr:.4%}, MCC={mcc:.4f}")
results['temporal_drift'] = drift_results

# ---- C. Single-flow inference latency ----
print("[7/7] Task C — single-flow inference latency (1000 individual predictions)...")
sample_idx = np.random.RandomState(42).choice(len(X_64), size=min(1000, len(X_64)), replace=False)
latencies_ms = []
for i in sample_idx:
    row = X_64[i:i+1]
    t0 = time.perf_counter()
    _ = lgbm.predict_proba(row)
    latencies_ms.append((time.perf_counter() - t0) * 1000)
latencies_ms = np.array(latencies_ms)
latency_stats = {
    'n_samples': len(latencies_ms),
    'mean_ms': float(latencies_ms.mean()),
    'median_ms': float(np.median(latencies_ms)),
    'p95_ms': float(np.percentile(latencies_ms, 95)),
    'p99_ms': float(np.percentile(latencies_ms, 99)),
    'max_ms': float(latencies_ms.max()),
}
print(f"    Mean: {latency_stats['mean_ms']:.3f}ms  Median: {latency_stats['median_ms']:.3f}ms  "
      f"P95: {latency_stats['p95_ms']:.3f}ms  P99: {latency_stats['p99_ms']:.3f}ms")
results['inference_latency'] = latency_stats

# ---- Save all results to the isolated output folder ----
out_file = OUT_DIR / "ml_testing_extras_results.json"
with open(out_file, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nAll results saved to: {out_file}")
print("No existing files were read-write touched. No existing project folder was modified.")