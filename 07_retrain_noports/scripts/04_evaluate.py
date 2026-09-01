import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

"""
04_evaluate.py  
===============================================
Step 4 of 4 — Final evaluation on HELD-OUT test set.

"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, matthews_corrcoef, classification_report,
    roc_curve, precision_recall_curve, average_precision_score,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR   = Path("/home/ramanaruban.online/soc.ramanaruban.online/Soc/07_retrain_noports/models")
RESULTS_DIR = Path("/home/ramanaruban.online/soc.ramanaruban.online/Soc/07_retrain_noports/results")

# ── Loading model, metadata, feature names ──────────────────────────────────────
print("Loading model and metadata...")
model = joblib.load(MODEL_DIR / "cicapt_lgbm_v2.pkl")

with open(RESULTS_DIR / "training_metadata.json") as f:
    metadata = json.load(f)

with open(MODEL_DIR / "feature_names.json") as f:
    feature_names = json.load(f)

deploy_thresh = metadata["deploy_threshold"]
print(f"  Deploy threshold (tuned on val): {deploy_thresh:.4f}")
print(f"  Feature names loaded: {feature_names[:3]} … ({len(feature_names)} total)")

# Verifying names match what the model was trained on
assert list(model.feature_name_) == feature_names, (
    "feature_names.json does not match model.feature_name_.\n"
    "Re-run Scripts 02 and 03 to regenerate both artefacts from the same run."
)
print(f"   feature_names.json matches model.feature_name_")

# ── Loading test set ─────────────────────────────────────────────────────────────
print("\nLoading held-out test set...")
X_test_arr = np.load(MODEL_DIR / "X_test.npy")
y_test     = np.load(MODEL_DIR / "y_test.npy")

print(f"  Test : {X_test_arr.shape} | attacks: {y_test.sum():,} | benign: {(y_test==0).sum():,}")

# Wrap in DataFrame — same columns as training → no feature-name warning
X_test = pd.DataFrame(X_test_arr, columns=feature_names)
del X_test_arr
print(f"   Wrapped in DataFrame ({X_test.shape})")

# ── Prediction ───────────────────────────────────────────────────────────────────
print("\nRunning inference on test set...")
y_prob = model.predict_proba(X_test)[:, 1]

# ── Evaluate at deploy threshold ─────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"FINAL TEST RESULTS (threshold={deploy_thresh:.4f})")
print(f"{'='*60}")

y_pred = (y_prob >= deploy_thresh).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

test_metrics = {
    "model"    : "cicapt_lgbm_v2",
    "dataset"  : "CICAPT-IIoT Phase2 (held-out test set, 20%)",
    "threshold": deploy_thresh,
    "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    "TPR"      : round(tp / max(tp + fn, 1), 6),
    "FPR"      : round(fp / max(fp + tn, 1), 6),
    "TNR"      : round(tn / max(tn + fp, 1), 6),
    "precision": round(precision_score(y_test, y_pred, zero_division=0), 6),
    "recall"   : round(recall_score(y_test, y_pred, zero_division=0), 6),
    "f1"       : round(f1_score(y_test, y_pred, zero_division=0), 6),
    "auc_roc"  : round(roc_auc_score(y_test, y_prob), 6),
    "auc_pr"   : round(average_precision_score(y_test, y_prob), 6),
    "mcc"      : round(matthews_corrcoef(y_test, y_pred), 6),
}

print("\nConfusion Matrix:")
print(f"  TP={tp:,}  FP={fp:,}")
print(f"  FN={fn:,}  TN={tn:,}")
print()
for k, v in test_metrics.items():
    print(f"  {k:12s}: {v}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Benign', 'Attack']))

# ── ROC / PR curve data ───────────────────────────────────────────────────────
fpr_arr, tpr_arr, roc_thresholds = roc_curve(y_test, y_prob)
np.save(RESULTS_DIR / "roc_fpr.npy",        fpr_arr)
np.save(RESULTS_DIR / "roc_tpr.npy",        tpr_arr)
np.save(RESULTS_DIR / "roc_thresholds.npy", roc_thresholds)

prec_arr, rec_arr, pr_thresholds = precision_recall_curve(y_test, y_prob)
np.save(RESULTS_DIR / "pr_precision.npy",  prec_arr)
np.save(RESULTS_DIR / "pr_recall.npy",     rec_arr)
np.save(RESULTS_DIR / "pr_thresholds.npy", pr_thresholds)

# ── Baseline comparison ───────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("BASELINE COMPARISON (threshold=0.5)")
print(f"{'='*60}")

y_pred_base = (y_prob >= 0.5).astype(int)
tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_test, y_pred_base, labels=[0, 1]).ravel()

baseline = {
    "threshold": 0.5,
    "TP": int(tp_b), "FP": int(fp_b), "TN": int(tn_b), "FN": int(fn_b),
    "TPR": round(tp_b / max(tp_b + fn_b, 1), 6),
    "FPR": round(fp_b / max(fp_b + tn_b, 1), 6),
    "f1" : round(f1_score(y_test, y_pred_base, zero_division=0), 6),
}

print(f"  TPR={baseline['TPR']:.4f}  FPR={baseline['FPR']:.4f}  F1={baseline['f1']:.4f}")
print(f"\nImprovement over baseline:")
print(f"  FPR reduction: {baseline['FPR']:.4f} → {test_metrics['FPR']:.6f} "
      f"({(1 - test_metrics['FPR'] / max(baseline['FPR'], 1e-9)) * 100:.1f}% reduction)")
print(f"  F1 change:     {baseline['f1']:.4f} → {test_metrics['f1']:.4f}")

# ── Save final results ────────────────────────────────────────────────────────
final_results = {
    "proposed_method": test_metrics,
    "baseline"       : baseline,
    "curves_saved"   : ["roc_fpr.npy", "roc_tpr.npy", "pr_precision.npy", "pr_recall.npy"],
}
with open(RESULTS_DIR / "final_test_metrics.json", "w") as f:
    json.dump(final_results, f, indent=2)

print(f"\n Final results saved → {RESULTS_DIR}/final_test_metrics.json")
print(f" ROC and PR curve data saved to {RESULTS_DIR}/")
print(f"\n→ Copy model to deployment:")
print(f"  cp {MODEL_DIR}/cicapt_lgbm_v2.pkl       /home/ramanaruban.online/soc.ramanaruban.online/Soc/application/backend/models/NEWPORTS_cicapt_lgbm.pkl")
print(f"  cp {MODEL_DIR}/ae_phase2.h5              /home/ramanaruban.online/soc.ramanaruban.online/Soc/application/backend/models/NEWPORTS_cicapt_autoencoder.h5")
print(f"  cp {MODEL_DIR}/ae_scaler_phase2.pkl      /home/ramanaruban.online/soc.ramanaruban.online/Soc/application/backend/models/NEWPORTS_cicapt_ae_scaler.pkl")
print(f"  cp {MODEL_DIR}/feature_names.json        /home/ramanaruban.online/soc.ramanaruban.online/Soc/application/backend/models/NEWPORTS_feature_names.json")