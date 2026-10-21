import sys, glob, numpy as np, pandas as pd
sys.path.insert(0, ".")
import aegis_v2_updated as A
d = pd.read_csv("all_attack_bursts.csv")
ts = d["ts"].values.astype(float)
cl = (d["subLabel"].str.lower() == "cleanup").values
for gap in ["row", "0.5s", "1s", "5s", "30s", "120s"]:
    v = []
    for f in sorted(glob.glob(f"experiments_v2/runs_burst/gap{gap}_seed*.npz")):
        z = np.load(f, allow_pickle=True); r = {k: z[k] for k in z.files}
        v.append(float(A.test_metrics(r, A.thr_fpr(r))["mcc"]))
    v = np.array(v)
    print(f"{gap:>5s} per-seed {np.round(v, 3).tolist()}")
    print(f"      mean {v.mean():.3f}  median {np.median(v):.3f}  lowest {v.min():.3f}")
for gap in [5, 120]:
    ids = A.bursts_from_gap(ts, gap)
    for s in range(10):
        sp = A.partition(ids, np.random.default_rng(s))
        z = np.load(f"experiments_v2/runs_burst/gap{gap}s_seed{s:02d}.npz", allow_pickle=True); r = {k: z[k] for k in z.files}
        m = A.test_metrics(r, A.thr_fpr(r))["mcc"]
        print(f"gap {gap:>3}s seed {s} cleanup train {int((cl & (sp == 0)).sum()):3d} val {int((cl & (sp == 1)).sum()):3d} test {int((cl & (sp == 2)).sum()):3d} | test MCC {m:.3f}")
