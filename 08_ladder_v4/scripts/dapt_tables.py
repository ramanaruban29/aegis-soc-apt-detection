import json, glob, sys, collections, numpy as np
sys.path.insert(0, "../aptevalkit")
from aptevalkit import paired_summary
folder = sys.argv[1]
R = [json.load(open(f)) for f in sorted(glob.glob(folder + "/seed*.json"))]
J = len(R); print(f"{J} runs in {folder}")
FP = ("fpr0.002", "fpr0.005", "fpr0.01")
ROWS = [("Deployed (full)", "lgbm_full"), ("No-score", "lgbm_noscore"), ("Full, unclipped score", "lgbm_full_unclip"),
        ("LightGBM tuned", "lgbm_tuned"), ("Extra Trees", "et"), ("Gradient Boosting", "gb"), ("Random Forest", "rf"),
        ("XGBoost", "xgb"), ("No-ports", "lgbm_noports"), ("AdaBoost", "ada"), ("MLP", "mlp"),
        ("Autoencoder alone (clipped)", "ae_only"), ("Autoencoder alone (unclipped)", "ae_only_unclipped"),
        ("Isolation Forest", "iforest"), ("One-Class SVM", "ocsvm")]
col = lambda tag, f: np.array([r["res"][tag][f]["mcc"] for r in R])
print("\n### Table A: test MCC, mean (sd) over runs")
print("| Model | FPR 0.2% | FPR 0.5% | FPR 1% | MCC rule |\n|---|---|---|---|---|")
for label, tag in ROWS:
    if tag not in R[0]["res"] or not R[0]["res"][tag]: continue
    c = [col(tag, f) for f in FP + ("mccrule",)]
    fm = [f"{x.mean():.3f} ({x.std(ddof=1):.3f})" for x in c]
    if abs(c[0].mean() - c[1].mean()) < 1e-9 and abs(c[1].mean() - c[2].mean()) < 1e-9: fm[:3] = [fm[0] + "†"] * 3
    print(f"| {label} | " + " | ".join(fm) + " |")
print("† identical at all three targets: scores are tied, use the MCC-rule column only")

n_tr = np.mean([r["info"]["attacks"][0] + r["info"]["benign"][0] for r in R])
n_te = np.mean([r["info"]["attacks"][2] + r["info"]["benign"][2] for r in R])
others = ["lgbm_noscore", "lgbm_full_unclip", "lgbm_noports", "lgbm_tuned", "et", "gb", "rf", "xgb", "ada"]
print(f"\n### Table B: deployed minus other, MCC at FPR 0.5% ({len(others)} comparisons; Bonferroni alpha = {0.05/len(others):.4f})")
print("| Other | mean diff | 95% run bootstrap | corrected p |\n|---|---|---|---|")
for t in others:
    s = paired_summary(col("lgbm_full", "fpr0.005"), col(t, "fpr0.005"), n_tr, n_te)
    print(f"| {t} | {s['mean']:+.3f} | [{s['ci_runs'][0]:+.3f}, {s['ci_runs'][1]:+.3f}] | {s.get('p_corrected', float('nan')):.4f} |")

SUP = ["lgbm_full", "lgbm_noscore", "lgbm_noports", "lgbm_tuned", "xgb", "rf", "et", "gb", "ada"]
FAM = dict(lgbm_full="LightGBM", lgbm_noscore="LightGBM", lgbm_noports="LightGBM", lgbm_tuned="LightGBM",
           xgb="XGBoost", rf="RandomForest", et="ExtraTrees", gb="GradBoost", ada="AdaBoost")
M = np.array([[r["res"][t]["fpr0.005"]["mcc"] for t in SUP] for r in R])
rk = (-M).argsort(1).argsort(1) + 1
print("\n### Table C: mean rank at FPR 0.5% (9 models, 1 = best)")
for t, v in sorted(zip(SUP, rk.mean(0)), key=lambda x: x[1]): print(f"  {t:14s} {v:.2f}")
w = M.argmax(1)
print("winners by model :", dict(collections.Counter(SUP[i] for i in w)))
print("winners by family:", dict(collections.Counter(FAM[SUP[i]] for i in w)))

te = np.array([r["info"]["attacks"][2] for r in R]); m = col("lgbm_full", "fpr0.005")
print(f"\n### Split lottery: corr(test attacks, MCC) = {np.corrcoef(te, m)[0, 1]:+.2f}")
print("test attacks -> MCC:", "  ".join(f"{te[i]}:{m[i]:.2f}" for i in np.argsort(te)))
