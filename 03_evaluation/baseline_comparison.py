#!/usr/bin/env python3
"""
================================================================================
 AEGIS SOC — 10-Model Baseline Comparison  (Chapter 7 / Table B) 
================================================================================
 Reproduces the Sahu et al. (ICDCN 2026) "Table 1" benchmark design on the
 CICAPT-IIoT Phase-2 NETWORK-FLOW modality, using the SAME preprocessing, split
 and feature construction as evaluate_ablation_models.py.

 MEMORY-SAFE DESIGN:
   * TRAIN set is stratified-subsampled to TRAIN_SUBSAMPLE rows (default 300k),
     preserving the attack:benign ratio. Standard, disclosable practice.
   * TEST set is used in FULL.

================================================================================
"""

import warnings, os, json, time, gc
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import joblib
import pyarrow.parquet as pq
import tensorflow as tf

# ── PATHS  ─────────────────────────
CSV         = '/home/ramanan/apt_project/data/raw/phase2/phase2_NetworkData.csv'
PARQUET     = '/home/ramanan/apt_project/data/processed/phase2/train_normal.parquet'
AE_PATH     = '/home/ramanan/SOC_New/backend/models/cicapt_autoencoder.h5'
SCALER_PATH = '/home/ramanan/SOC_New/backend/models/cicapt_ae_scaler.pkl'
CLIP_META   = '/home/ramanan/SOC_New/backend/models/ae_clip_meta.json'
DEPLOYED_AEGIS_PKL = '/home/ramanan/SOC_New/backend/models/cicapt_lgbm.pkl'

CHUNK_SIZE       = 200_000
TRAIN_SUBSAMPLE  = 300_000     
AE_BATCH         = 512
RANDOM_STATE     = 42
OUT_DIR          = 'baseline_comparison'


USE_DEPLOYED_AEGIS = True

#KNN excluded as slowest and weakest baseline
SKIP_MODELS = {"KNN"}   

os.makedirs(OUT_DIR, exist_ok=True)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (matthews_corrcoef, confusion_matrix, f1_score,
                             accuracy_score, precision_score, recall_score)

THRESHOLDS = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85,
              0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.985,
              0.99, 0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999]

from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier, GradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import (QuadraticDiscriminantAnalysis,
                                          LinearDiscriminantAnalysis)
import lightgbm as lgb
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    print("[warn] xgboost not installed — skipping XGBoost. Install: pip install xgboost")


# ── 1.  60/20/20 split ──────────────
print("[1/7] Pass 1 — reading 'label' column for the split...")
labels = []
for chunk in pd.read_csv(CSV, usecols=['label'], chunksize=CHUNK_SIZE, low_memory=False):
    labels.append(chunk['label'].values)
y_all = np.concatenate(labels); del labels; gc.collect()
n_total = len(y_all)
print(f"    total rows: {n_total:,}  attacks: {int(y_all.sum())}")

idx_all = np.arange(n_total)
trainval_idx, test_idx = train_test_split(
    idx_all, test_size=0.2, stratify=y_all, random_state=RANDOM_STATE)
train_idx, val_idx = train_test_split(
    trainval_idx, test_size=0.25, stratify=y_all[trainval_idx], random_state=RANDOM_STATE)

# ── stratified subsample of TRAIN indices ─────
y_train_full = y_all[train_idx]
if len(train_idx) > TRAIN_SUBSAMPLE:
    sub_idx, _ = train_test_split(
        np.arange(len(train_idx)), train_size=TRAIN_SUBSAMPLE,
        stratify=y_train_full, random_state=RANDOM_STATE)
    train_idx = train_idx[sub_idx]
    print(f"    train subsampled -> {len(train_idx):,} rows "
          f"({int(y_all[train_idx].sum())} attacks, stratified)")
else:
    print(f"    train kept full -> {len(train_idx):,} rows")

train_idx_set = set(train_idx.tolist())
val_idx_set   = set(val_idx.tolist())
test_idx_set  = set(test_idx.tolist())
print(f"    val:  {len(val_idx):,} rows, {int(y_all[val_idx].sum())} attacks")
print(f"    test (full): {len(test_idx):,} rows, {int(y_all[test_idx].sum())} attacks")
del y_train_full, trainval_idx, idx_all; gc.collect()


# ── 2. Shared preprocessing artefacts ────────────────────────────
print("[2/7] Loading scaler + autoencoder + clip meta (read-only)...")
ae = tf.keras.models.load_model(AE_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)
with open(CLIP_META) as f:
    clip_meta = json.load(f)
feat_clip  = float(clip_meta.get('scale_clip_value', 10.0))
score_clip = float(clip_meta.get('score_clip_p99', 1.0))
FLOW_COLS = [c for c in pq.read_table(PARQUET).schema.names if c != 'ts']
print(f"    flow features: {len(FLOW_COLS)}  feat_clip={feat_clip}  score_clip={score_clip}")


# ── helper: streaming CSV once, build fused 64-feature matrix for an index set ──
def build_matrix(keep_set, tag):
    keep_arr = np.fromiter(keep_set, dtype=np.int64, count=len(keep_set))
    keep_sorted = np.sort(keep_arr)
    rows = []
    row_offset = 0
    for chunk in pd.read_csv(CSV, chunksize=CHUNK_SIZE, low_memory=False):
        g_idx = np.arange(row_offset, row_offset + len(chunk))
        mask = np.isin(g_idx, keep_sorted, assume_unique=True)
        if mask.any():
            rows.append(chunk.loc[mask])
        row_offset += len(chunk)
    df = pd.concat(rows, ignore_index=True); del rows; gc.collect()
    y = df['label'].values.astype(int)

    X_raw = pd.DataFrame({c: pd.to_numeric(df[c], errors='coerce').fillna(0)
                          if c in df.columns else 0.0 for c in FLOW_COLS})
    del df; gc.collect()
    X_raw = X_raw.replace([np.inf, -np.inf], 0).fillna(0).values.astype(np.float32)

    X_scaled = scaler.transform(X_raw).astype(np.float32)
    del X_raw; gc.collect()
    X_scaled = np.clip(X_scaled, -feat_clip, feat_clip).astype(np.float32)

    recon = ae.predict(X_scaled, batch_size=AE_BATCH, verbose=0)
    score = np.mean((X_scaled - recon) ** 2, axis=1).astype(np.float32)
    del recon; gc.collect()
    score = np.clip(score, 0.0, score_clip).astype(np.float32)

    X_64 = np.column_stack([score, X_scaled]).astype(np.float32)
    del X_scaled, score; gc.collect()
    print(f"    {tag}: X_64={X_64.shape}  attacks={int(y.sum())}")
    return X_64, y


print("[3/7] Building TRAIN matrix (subsampled, fused 64-feature)...")
X_tr, y_tr = build_matrix(train_idx_set, "train")
print("[4/7] Building VALIDATION matrix (for threshold selection)...")
X_val, y_val = build_matrix(val_idx_set, "val")
gc.collect()
# NOTE: TEST matrix is built later, AFTER training + threshold selection,
# so I never hold val and test matrices at the same time to be memory consisitent


# ── 3. Baselines ─────────────────────────────────────────────────────────────
def make_models():
    m = {}
    if HAS_XGB:
        m["XGBoost"] = XGBClassifier(
            n_estimators=400, max_depth=8, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", tree_method="hist",
            n_jobs=-1, random_state=RANDOM_STATE)
    m["Random Forest"]       = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)
    m["Extra Trees"]         = ExtraTreesClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)
    m["Decision Tree"]       = DecisionTreeClassifier(random_state=RANDOM_STATE)
    m["AdaBoost"]            = AdaBoostClassifier(n_estimators=200, random_state=RANDOM_STATE)
    m["Gradient Boosting"]   = GradientBoostingClassifier(random_state=RANDOM_STATE)
    m["KNN"]                 = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    m["Naive Bayes"]         = GaussianNB()
    m["Logistic Regression"] = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=RANDOM_STATE)
    m["LDA"]                 = LinearDiscriminantAnalysis()
    m["QDA"]                 = QuadraticDiscriminantAnalysis()
    return m

def make_aegis():
    return lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.05, max_depth=8, num_leaves=63,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, objective='binary',
        random_state=RANDOM_STATE, n_jobs=-1)


def select_threshold_on_val(model, X_val, y_val):
    """Proper standard: picking the best-MCC threshold on the VALIDATION set."""
    probs = model.predict_proba(X_val)[:, 1]
    best_t, best_mcc = THRESHOLDS[0], -2.0
    for t in THRESHOLDS:
        preds = (probs >= t).astype(int)
        mcc = matthews_corrcoef(y_val, preds)
        if mcc > best_mcc:
            best_mcc, best_t = mcc, t
    return best_t, round(best_mcc, 4)

def evaluate_on_test(name, model, X_te, y_te, chosen_t, val_mcc, train_s):
    """Reporting final metrics on the TEST set at the frozen validation threshold."""
    t0 = time.time()
    probs = model.predict_proba(X_te)[:, 1]
    infer_s = time.time() - t0
    preds = (probs >= chosen_t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "Model": name, "Val_threshold": chosen_t, "Val_MCC": val_mcc,
        "Accuracy":  round(accuracy_score(y_te, preds), 4),
        "Precision": round(precision_score(y_te, preds, zero_division=0), 4),
        "Recall":    round(recall_score(y_te, preds, zero_division=0), 4),
        "F1":        round(f1_score(y_te, preds, zero_division=0), 4),
        "MCC":       round(matthews_corrcoef(y_te, preds), 4),
        "FPR":       round(fpr, 6),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "Train_s": round(train_s, 2), "Infer_s": round(infer_s, 4),
    }


# ── 4. PHASE 1: train each model, pick threshold on VALIDATION, keep model ──

print("\n[5/7] Training baselines + selecting thresholds on VALIDATION...\n")
trained = {}          # name -> (model, chosen_threshold, val_mcc, train_s)
for name, clf in make_models().items():
    if name in SKIP_MODELS:
        print(f"   - {name:22s} skipped (SKIP_MODELS)")
        continue
    try:
        t0 = time.time(); clf.fit(X_tr, y_tr); train_s = time.time() - t0
        chosen_t, val_mcc = select_threshold_on_val(clf, X_val, y_val)
        trained[name] = (clf, chosen_t, val_mcc, train_s)
        print(f"   ✓ {name:22s} val-MCC={val_mcc:.4f}  chosen t={chosen_t}")
    except MemoryError:
        print(f"   ✗ {name:22s} MemoryError — add to SKIP_MODELS and re-run")
    except Exception as e:
        print(f"   ✗ {name:22s} failed: {e}")

# Freeing training + validation matrices before building the (large) test matrix
del X_tr, y_tr, X_val, y_val; gc.collect()

# ── PHASE 2: building TEST matrix, evaluate every model at its frozen threshold ─
print("\n[6/7] Building TEST matrix (full) and evaluating at frozen thresholds...")
X_te, y_te = build_matrix(test_idx_set, "test")
gc.collect()

rows = []
for name, (clf, chosen_t, val_mcc, train_s) in trained.items():
    try:
        row = evaluate_on_test(name, clf, X_te, y_te, chosen_t, val_mcc, train_s)
        rows.append(row)
        print(f"   ✓ {name:22s} TEST F1={row['F1']:.4f} MCC={row['MCC']:.4f} "
              f"FPR={row['FPR']*100:.4f}% @t={chosen_t}")
    except Exception as e:
        print(f"   ✗ {name:22s} test eval failed: {e}")


# ── 5. AEGIS row: deployed model at its REAL deployed threshold (0.9930) ────

print("\n[7/7] AEGIS Hybrid (deployed) at its real threshold 0.9930...")
DEPLOY_THRESHOLD = 0.9930
try:
    obj = joblib.load(DEPLOYED_AEGIS_PKL)
    aegis = obj['model'] if isinstance(obj, dict) else obj
    row = evaluate_on_test("AEGIS Hybrid (AE + LightGBM)", aegis, X_te, y_te,
                           DEPLOY_THRESHOLD, val_mcc=None, train_s=0.0)
    rows.append(row)
    print(f"   ★ AEGIS (deployed)      TEST F1={row['F1']:.4f} MCC={row['MCC']:.4f} "
          f"FPR={row['FPR']*100:.4f}% @t={DEPLOY_THRESHOLD}")
except Exception as e:
    print(f"   ✗ AEGIS deployed load/eval failed: {e}")
n_test_rows = len(y_te)
del X_te, y_te; gc.collect()


# ── 6. Saving Baseline Comparison Results ──────────────────────────────────────────────────────────────────
res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT_DIR, "baseline_comparison_results.csv"), index=False)

view = res.copy()
for c in ["Accuracy", "Precision", "Recall", "F1"]:
    view[c] = (view[c] * 100).round(2).astype(str) + "%"
view["FPR"] = (res["FPR"] * 100).round(4).astype(str) + "%"
view = view[["Model", "Accuracy", "Precision", "Recall", "F1", "MCC", "FPR", "Infer_s"]]
view = view.sort_values("MCC", ascending=False)
view.to_csv(os.path.join(OUT_DIR, "baseline_table_thesis.csv"), index=False)

print("\n[7/7] " + "=" * 66)
print("DONE. Written to", OUT_DIR + "/")
print("  baseline_comparison_results.csv  (full metrics + confusion + threshold)")
print("  baseline_table_thesis.csv        (clean %, sorted by MCC)")
print("=" * 72)
print(f"\nNOTE for caption: baselines TRAINED on up to {TRAIN_SUBSAMPLE:,}-row "
      f"stratified sample; thresholds SELECTED on validation; all models "
      f"EVALUATED on the full held-out test split ({n_test_rows:,} rows).")
print("\nThesis-ready table (sorted by MCC):\n")
print(view.to_string(index=False))