import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import joblib
import json
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, matthews_corrcoef, precision_recall_curve,
    average_precision_score,
)

MODEL_DIR   = Path("/home/ramanaruban.online/soc.ramanaruban.online/Soc/07_retrain_noports/models")
RESULTS_DIR = Path("/home/ramanaruban.online/soc.ramanaruban.online/Soc/07_retrain_noports/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED  = 42
BENIGN_RATIO = 50

with open(MODEL_DIR / "feature_names.json") as f:
    FEATURE_NAMES = json.load(f)

print(f"Feature names: {FEATURE_NAMES[:3]} … ({len(FEATURE_NAMES)} total)")

def to_df(arr):
    assert arr.shape[1] == len(FEATURE_NAMES)
    return pd.DataFrame(arr, columns=FEATURE_NAMES)

print("Loading train/val splits...")
X_train_full = np.load(MODEL_DIR / "X_train.npy")
y_train_full = np.load(MODEL_DIR / "y_train.npy")
X_val        = np.load(MODEL_DIR / "X_val.npy")
y_val        = np.load(MODEL_DIR / "y_val.npy")

n_attacks_train = int(y_train_full.sum())
n_benign_train  = int((y_train_full == 0).sum())
print(f"  Full train: {X_train_full.shape} | attacks={n_attacks_train:,}  benign={n_benign_train:,}")

n_keep_benign = min(n_attacks_train * BENIGN_RATIO, n_benign_train)
rng           = np.random.default_rng(RANDOM_SEED)
attack_idx    = np.where(y_train_full == 1)[0]
benign_idx    = np.where(y_train_full == 0)[0]
benign_sample = rng.choice(benign_idx, size=n_keep_benign, replace=False)
train_idx     = np.concatenate([attack_idx, benign_sample])
rng.shuffle(train_idx)

X_train = X_train_full[train_idx]
y_train = y_train_full[train_idx]
del X_train_full, y_train_full

print(f"  Undersampled train: {X_train.shape} | attacks={y_train.sum():,}  benign={(y_train==0).sum():,}  ratio={BENIGN_RATIO}:1")

df_train = to_df(X_train)
df_val   = to_df(X_val)

print("\nTraining LightGBM...")
model = lgb.LGBMClassifier(
    n_estimators      = 2000,
    learning_rate     = 0.05,
    max_depth         = 8,
    num_leaves        = 63,
    min_child_samples = 5,
    scale_pos_weight  = 1,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    reg_alpha         = 0.1,
    reg_lambda        = 0.1,
    random_state      = RANDOM_SEED,
    n_jobs            = -1,
    verbose           = -1,
)

model.fit(
    df_train, y_train,
    eval_set    = [(df_val, y_val)],
    eval_metric = "auc",
    callbacks   = [
        lgb.early_stopping(stopping_rounds=100, verbose=True),
        lgb.log_evaluation(50),
    ],
)

print(f"\n✓ Best iteration : {model.best_iteration_}")
if model.best_iteration_ <= 5:
    print("WARNING: still stopping early — check feature matrix")

# Rewrite feature_names.json from model to guarantee match
with open(MODEL_DIR / "feature_names.json", "w") as f:
    json.dump(list(model.feature_name_), f)
print(f"✓ feature_names.json rewritten from model")

print("\nScore distribution on val set:")
y_prob_val    = model.predict_proba(df_val)[:, 1]
attack_scores = y_prob_val[y_val == 1]
benign_scores = y_prob_val[y_val == 0]

print(f"  Attacks — mean={attack_scores.mean():.4f}  p5={np.percentile(attack_scores,5):.4f}  p50={np.percentile(attack_scores,50):.4f}")
print(f"  Benign  — mean={benign_scores.mean():.4f}  p95={np.percentile(benign_scores,95):.4f}  p99={np.percentile(benign_scores,99):.4f}")

unique_scores = len(np.unique(np.round(y_prob_val, 3)))
print(f"  Unique scores (3dp): {unique_scores}")
if unique_scores < 10:
    print("  Still near-binary — reduce BENIGN_RATIO to 20 and rerun")
else:
    print(" Score distribution healthy")

auc_roc = roc_auc_score(y_val, y_prob_val)
auc_pr  = average_precision_score(y_val, y_prob_val)
print(f"\n  AUC-ROC : {auc_roc:.4f}")
print(f"  AUC-PR  : {auc_pr:.4f}  ← primary metric")

best_f1_thresh = best_mcc_thresh = 0.5
best_f1_val = best_mcc_val = 0.0
zero_fp_thresh = None
results = []

for t in np.linspace(0.001, 0.999, 1000):
    y_pred         = (y_prob_val >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[0, 1]).ravel()
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    fpr  = fp / max(fp + tn, 1)
    mcc  = matthews_corrcoef(y_val, y_pred)
    results.append((float(t), rec, fpr, prec, rec, f1, mcc, int(tp), int(fp)))
    if f1  > best_f1_val:  best_f1_val,  best_f1_thresh  = f1,  float(t)
    if mcc > best_mcc_val: best_mcc_val, best_mcc_thresh = mcc, float(t)
    if zero_fp_thresh is None and fpr < 0.0001 and tp > 0:
        zero_fp_thresh = float(t)

print(f"\n  Best F1  threshold : {best_f1_thresh:.4f}  (F1={best_f1_val:.4f})")
print(f"  Best MCC threshold : {best_mcc_thresh:.4f}  (MCC={best_mcc_val:.4f})")
print(f"  Zero-FP  threshold : {zero_fp_thresh:.4f}" if zero_fp_thresh else "  Zero-FP threshold : not found")

if best_mcc_val > 0.1:
    deploy_thresh = best_mcc_thresh
elif best_f1_val > 0.05:
    deploy_thresh = best_f1_thresh
else:
    deploy_thresh = 0.5
print(f"\n  → Deploy threshold : {deploy_thresh:.4f}")

y_pred_val     = (y_prob_val >= deploy_thresh).astype(int)
tn, fp, fn, tp = confusion_matrix(y_val, y_pred_val, labels=[0, 1]).ravel()

val_metrics = {
    "threshold" : round(deploy_thresh, 6), "split": "validation",
    "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    "TPR"       : round(tp / max(tp + fn, 1), 6),
    "FPR"       : round(fp / max(fp + tn, 1), 6),
    "precision" : round(precision_score(y_val, y_pred_val, zero_division=0), 6),
    "recall"    : round(recall_score(y_val, y_pred_val, zero_division=0), 6),
    "f1"        : round(f1_score(y_val, y_pred_val, zero_division=0), 6),
    "auc_roc"   : round(auc_roc, 6),
    "auc_pr"    : round(auc_pr, 6),
    "mcc"       : round(matthews_corrcoef(y_val, y_pred_val), 6),
}

print(f"\nValidation metrics at threshold={deploy_thresh:.4f}:")
for k, v in val_metrics.items():
    print(f"  {k:12s}: {v}")

print("\nThreshold comparison (for paper):")
print(f"  {'Threshold':>10}  {'TPR':>6}  {'FPR':>8}  {'F1':>6}  {'MCC':>6}  {'TP':>4}  {'FP':>5}")
print(f"  {'-'*60}")
for row in results[::100]:
    t, tpr, fpr, prec, rec, f1, mcc, tp_i, fp_i = row
    print(f"  {t:10.4f}  {tpr:6.4f}  {fpr:8.6f}  {f1:6.4f}  {mcc:6.4f}  {tp_i:4d}  {fp_i:5d}")

joblib.dump(model, MODEL_DIR / "cicapt_lgbm_v2.pkl")
print(f"\n✓ Model saved → {MODEL_DIR}/cicapt_lgbm_v2.pkl")

precision_arr, recall_arr, _ = precision_recall_curve(y_val, y_prob_val)
np.save(RESULTS_DIR / "pr_precision.npy", precision_arr)
np.save(RESULTS_DIR / "pr_recall.npy",    recall_arr)

metadata = {
    "model": "cicapt_lgbm_v2",
    "trained_on": "CICAPT-IIoT Phase2 (train 60%, undersampled 50:1)",
    "threshold_tuned_on": "CICAPT-IIoT Phase2 (val 20%, full distribution)",
    "test_split": "CICAPT-IIoT Phase2 (test 20%) — held out",
    "n_features": int(X_train.shape[1]),
    "benign_ratio": BENIGN_RATIO,
    "scale_pos_weight": 1,
    "deploy_threshold": round(deploy_thresh, 6),
    "best_f1_threshold": round(best_f1_thresh, 6),
    "best_mcc_threshold": round(best_mcc_thresh, 6),
    "zero_fp_threshold": round(zero_fp_thresh, 6) if zero_fp_thresh else None,
    "val_metrics": val_metrics,
    "class_counts": {
        "train_attacks": n_attacks_train,
        "train_benign_kept": int(n_keep_benign),
        "train_benign_total": n_benign_train,
        "val_attacks": int(y_val.sum()),
    },
}
with open(RESULTS_DIR / "training_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Metadata saved → {RESULTS_DIR}/training_metadata.json")
print(f"Run 04_evaluate.py next")
