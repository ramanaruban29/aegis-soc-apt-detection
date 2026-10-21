import json, glob, sys, collections, numpy as np
sys.path.insert(0, "../aptevalkit")
from aptevalkit import paired_summary
R = [json.load(open(f)) for f in sorted(glob.glob(sys.argv[1] + "/*.json"))]
J = len(R); n_tr = np.mean([r["info"]["attacks"][0] for r in R]); n_te = np.mean([r["info"]["attacks"][2] for r in R])
nb = np.mean([r["info"]["benign_sizes"][2] for r in R])
print(f"{J} runs | mean attacks train/test {n_tr:.0f}/{n_te:.0f} | mean test benign {nb:.0f}")
SUP = ["lgbm_full", "lgbm_noscore", "lgbm_noports", "lgbm_tuned", "xgb", "rf", "et", "gb", "ada"]
FAM = dict(lgbm_full="LightGBM", lgbm_noscore="LightGBM", lgbm_noports="LightGBM", lgbm_tuned="LightGBM", xgb="XGBoost", rf="RandomForest", et="ExtraTrees", gb="GradBoost", ada="AdaBoost")
F = ("fpr2e-05", "fpr4e-05", "fpr8e-05", "mccrule")
col = lambda m, f, k="mcc": np.array([r["res"][m][f][k] for r in R])
print("\nMedian MCC [median FP] (target FP: %s)" % ", ".join(f"{x*nb:.0f}" for x in (2e-5, 4e-5, 8e-5)))
print(f"{'model':14s}" + "".join(f"{f:>22s}" for f in F))
for m in SUP:
    print(f"{m:14s}" + "".join(f"{np.median(col(m, f)):>12.3f} [{np.median(col(m, f, 'fp')):>6.0f}]  " for f in F))
others = [m for m in SUP if m != "lgbm_full"]
for f in ("fpr4e-05", "fpr2e-05", "fpr8e-05"):
    print(f"\nDeployed minus other, {f} ({len(others)} comparisons, Bonferroni alpha {0.05/len(others):.4f})")
    for o in others:
        a, b = col("lgbm_full", f), col(o, f); s = paired_summary(a, b, n_tr, n_te)
        print(f"  {o:14s} {s['mean']:+.3f} runs-CI [{s['ci_runs'][0]:+.3f}, {s['ci_runs'][1]:+.3f}] corrected p {s.get('p_corrected', float('nan')):.3f} | deployed better {(a > b).sum()}/{J}")
M = np.array([[col(m, "fpr4e-05")[i] for m in SUP] for i in range(J)]); w = M.argmax(1)
print("\nwinners by model :", dict(collections.Counter(SUP[i] for i in w)))
print("winners by family:", dict(collections.Counter(FAM[SUP[i]] for i in w)))
