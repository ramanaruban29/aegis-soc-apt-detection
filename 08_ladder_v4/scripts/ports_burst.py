import sys, numpy as np
sys.path.insert(0, ".")
import aegis_v2_updated as A
D = A.Data(".")
v = D.att["val"]
big = np.bincount(v["burst"].values).argmax()
rows = np.sort(v.loc[v["burst"] == big, "orig"].values)
X = np.load("X_full.npy", mmap_mode="r")
ben = np.sort(D.idx["val"][D.y["val"] == 0])
ben = np.sort(np.random.default_rng(0).choice(ben, 300_000, replace=False))
for c, n in [(3, "Source_Port"), (4, "Destination_Port")]:
    b, bb = np.asarray(X[rows, c]), np.asarray(X[ben, c])
    vals, cnt = np.unique(b, return_counts=True); top = vals[cnt.argmax()]
    print(f"{n}: {len(b)} burst flows, {len(vals)} distinct values; most common value {top:.6g} covers {cnt.max()/len(b):.0%} of the burst and {(bb == top).mean():.2%} of benign validation flows")
