import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
import aegis_v2_updated as A
D = A.Data(".")
X = np.load("X_full.npy", mmap_mode="r")
v = D.att["val"]
big = np.bincount(v["burst"].values).argmax()
vr = np.sort(v.loc[v["burst"] == big, "orig"].values)
pairs, cnt = np.unique(np.asarray(X[vr])[:, [3, 4]], axis=0, return_counts=True)
ben = np.sort(D.idx["val"][D.y["val"] == 0])
bp = np.asarray(X[ben])[:, [3, 4]]
tr = D.att["train"]
tp = np.asarray(X[np.sort(tr["orig"].values)])[:, [3, 4]]
tstage = tr.sort_values("orig")["sub"].str.lower().values
for p, c in zip(pairs, cnt):
    m_b = ((bp == p).all(1)).mean(); m_t = (tp == p).all(1)
    print(f"port pair {p}: {c} of {cnt.sum()} burst flows | {m_b:.3%} of the {len(bp):,} benign validation flows | training attacks with this pair: {int(m_t.sum())}", pd.Series(tstage[m_t]).value_counts().to_dict())
