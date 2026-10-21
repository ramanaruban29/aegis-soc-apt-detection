import sys, numpy as np
sys.path.insert(0, ".")
import aegis_v2_updated as A
runs = A.load_runs("experiments_v2/runs")
tags = ["full", "noscore", "noports", "ada_tuned", "rf_tuned", "et_tuned", "xgb_tuned", "lgbm_tuned", "gb_tuned"]
seeds = sorted(set.intersection(*[set(runs[t]) for t in tags]))
M = np.array([[A.test_metrics(runs[t][s], A.thr_fpr(runs[t][s]))["mcc"] for s in seeds] for t in tags])
best = M.argmax(0); ranks = (-M).argsort(0).argsort(0) + 1
print("shared seeds:", seeds)
print("how often a single run would have ranked each model first (FPR rule, 0.004%):")
for i, t in enumerate(tags): print(f"  {t:12s} first in {(best == i).sum():2d}/{len(seeds)} seeds | mean rank {ranks[i].mean():.1f}")
