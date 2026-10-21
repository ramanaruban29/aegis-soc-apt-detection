import sys, numpy as np
sys.path.insert(0, ".")
import aegis_v2_updated as A
D = A.Data(".")
bv = D.att["val"]["burst"].values
big = np.bincount(bv).argmax()
runs = A.load_runs("experiments_v2/runs")
for tag in ["full", "noscore", "noports", "noports_same"]:
    low, rt, rf, mm = 0, [], [], []
    for s, r in sorted(runs[tag].items()):
        t = A.thr_valmcc(r); a = np.asarray(r["att_val"]); assert len(a) == len(bv)
        low += t < 0.5
        rt.append((a[bv == big] >= t).mean()); rf.append((a[bv == big] >= A.thr_fpr(r)).mean())
        mm.append(A.test_metrics(r, t)["mcc"])
    print(f"{tag:13s} low MCC-rule threshold in {low:2d}/{len(runs[tag])} seeds | recall on big val burst: at MCC-rule thr {np.mean(rt):.2f}, at FPR thr {np.mean(rf):.2f} | MCC-rule MCC {np.mean(mm):.3f} (sd {np.std(mm, ddof=1):.3f})")
