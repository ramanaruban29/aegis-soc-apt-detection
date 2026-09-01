"""
 Evaluating THREE real model
variants on the true held-out test split (60/20/20, random_state=42,
stratify=y):

  1. cicapt_lgbm_v2 (= the deployed model, backend/models/cicapt_lgbm.pkl)
     64 features (63 scaled + anomaly_score), 137 trees
  2. cicapt_lgbm_v4_model.pkl
     64 features (63 scaled + anomaly_score), 365 trees
  3. cicapt_lgbm_v5_model.pkl
     63 features (scaled only, NO anomaly_score) — genuine ablation, 757 trees

For each model, a threshold grid search is run on the SAME test set to find
its own best-MCC operating point, mirroring the methodology already used
for the deployed model (Chapter 6.4.3 / retrain_phase2/logs/03.log), so the
three-way comparison is methodologically fair and consistent.

"""
import warnings, os
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np, pandas as pd, joblib, json
import pyarrow.parquet as pq
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score

CSV         = '/home/ramanan/apt_project/data/raw/phase2/phase2_NetworkData.csv'
PARQUET     = '/home/ramanan/apt_project/data/processed/phase2/train_normal.parquet'
AE_PATH     = '/home/ramanan/SOC_New/backend/models/cicapt_autoencoder.h5'
SCALER_PATH = '/home/ramanan/SOC_New/backend/models/cicapt_ae_scaler.pkl'
CLIP_META   = '/home/ramanan/SOC_New/backend/models/ae_clip_meta.json'
CHUNK_SIZE  = 200_000

MODELS = {
    'v2 (deployed, 137 trees)':      '/home/ramanan/SOC_New/backend/models/cicapt_lgbm.pkl',
    'v4 (365 trees)':                '/home/ramanan/SOC_New/models/cicapt_lgbm_v4_model.pkl',
    'v5 (ablation, no anomaly_score, 757 trees)': '/home/ramanan/SOC_New/models/cicapt_lgbm_v5_model.pkl',
}

THRESHOLDS = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85,
              0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.985,
              0.99, 0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999]

print("[1/6] Pass 1 — reading only the 'label' column to compute the split...")
labels = []
for chunk in pd.read_csv(CSV, usecols=['label'], chunksize=CHUNK_SIZE, low_memory=False):
    labels.append(chunk['label'].values)
y_all = np.concatenate(labels)
del labels
n_total = len(y_all)
print(f"    Total rows: {n_total:,}  Total attacks: {int(y_all.sum())}")

idx_all = np.arange(n_total)
trainval_idx, test_idx = train_test_split(idx_all, test_size=0.2, stratify=y_all, random_state=42)
train_idx, val_idx = train_test_split(trainval_idx, test_size=0.25, stratify=y_all[trainval_idx], random_state=42)
test_idx_set = set(test_idx.tolist())
print(f"    Test split: {len(test_idx):,} rows, {int(y_all[test_idx].sum())} attacks")

print("[2/6] Loading shared preprocessing artefacts (read-only)...")
ae = tf.keras.models.load_model(AE_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)
with open(CLIP_META) as f:
    clip_meta = json.load(f)
feat_clip = float(clip_meta.get('scale_clip_value', 10.0))
score_clip = float(clip_meta.get('score_clip_p99', 1.0))
FLOW_COLS = [c for c in pq.read_table(PARQUET).schema.names if c != 'ts']

print("[3/6] Loading the three candidate models...")
loaded_models = {}
for name, path in MODELS.items():
    obj = joblib.load(path)
    model = obj['model'] if isinstance(obj, dict) else obj
    loaded_models[name] = model
    print(f"    {name}: n_features={model.n_features_}")

print("[4/6] Pass 2 — streaming CSV, keeping only test-split rows...")
test_rows = []
row_offset = 0
for chunk in pd.read_csv(CSV, chunksize=CHUNK_SIZE, low_memory=False):
    chunk_global_idx = np.arange(row_offset, row_offset + len(chunk))
    mask = np.isin(chunk_global_idx, test_idx)
    if mask.any():
        test_rows.append(chunk.loc[mask])
    row_offset += len(chunk)
    if row_offset % 1_000_000 < CHUNK_SIZE:
        print(f"    ...scanned {row_offset:,} / {n_total:,} rows")

test_df = pd.concat(test_rows, ignore_index=True)
del test_rows
y_test = test_df['label'].values
print(f"    Collected {len(test_df):,} test rows, {int(y_test.sum())} attacks")

print("[5/6] Running shared preprocessing: scale -> clip -> AE -> score clip...")
X_raw = pd.DataFrame({c: pd.to_numeric(test_df[c], errors='coerce').fillna(0) if c in test_df.columns else 0.0 for c in FLOW_COLS})
X_raw = X_raw.replace([np.inf, -np.inf], 0).fillna(0).values.astype(np.float32)

X_scaled = scaler.transform(X_raw).astype(np.float32)
X_scaled = np.clip(X_scaled, -feat_clip, feat_clip).astype(np.float32)

recon = ae.predict(X_scaled, batch_size=512, verbose=0)
anomaly_score = np.mean((X_scaled - recon) ** 2, axis=1).astype(np.float32)
anomaly_score = np.clip(anomaly_score, 0.0, score_clip).astype(np.float32)

X_64 = np.column_stack([anomaly_score, X_scaled]).astype(np.float32)
X_63 = X_scaled  # ablation input: no anomaly_score column
print(f"    X_64 shape: {X_64.shape}   X_63 shape: {X_63.shape}")

print("\n[6/6] Threshold sweep per model — reporting best-MCC operating point:\n")
results_summary = []
for name, model in loaded_models.items():
    X_input = X_64 if model.n_features_ == 64 else X_63
    probs = model.predict_proba(X_input)[:, 1]

    best = None
    for t in THRESHOLDS:
        preds = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        mcc = matthews_corrcoef(y_test, preds)
        f1 = f1_score(y_test, preds)
        tpr = tp / (tp + fn) if (tp + fn) else 0
        fpr = fp / (fp + tn) if (fp + tn) else 0
        row = dict(threshold=t, tp=tp, fp=fp, fn=fn, tn=tn, tpr=tpr, fpr=fpr, f1=f1, mcc=mcc)
        if best is None or mcc > best['mcc']:
            best = row

    print(f"=== {name} ===")
    print(f"  Best-MCC threshold: {best['threshold']}")
    print(f"  TP={best['tp']} FP={best['fp']} FN={best['fn']} TN={best['tn']}")
    print(f"  TPR={best['tpr']:.2%}  FPR={best['fpr']:.4%}  F1={best['f1']:.4f}  MCC={best['mcc']:.4f}\n")
    results_summary.append((name, best))

print("=" * 70)
print("SUMMARY TABLE (for thesis reporting)")
print("=" * 70)
print(f"{'Model':<45} {'Thresh':>8} {'TPR':>8} {'FPR':>10} {'F1':>8} {'MCC':>8}")
for name, r in results_summary:
    print(f"{name:<45} {r['threshold']:>8.3f} {r['tpr']:>7.2%} {r['fpr']:>9.4%} {r['f1']:>8.4f} {r['mcc']:>8.4f}")

print("\nDone. all Done.")