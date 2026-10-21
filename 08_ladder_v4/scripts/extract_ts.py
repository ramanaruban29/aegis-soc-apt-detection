import numpy as np, pandas as pd, pathlib
CSV = "/home/ramanan/CICAPT_IIoT/Dataset/CICAPT-IIoT Dataset/Network_Traffic/Phase2/phase2_NetworkData.csv"
OUT = pathlib.Path("experiments_v4"); OUT.mkdir(exist_ok=True)
ts, lab = [], []
for ch in pd.read_csv(CSV, usecols=["ts", "label"], chunksize=1_000_000, dtype={"ts": "float64", "label": "str"}):
    ts.append(ch["ts"].values); lab.append(ch["label"].values)
ts = np.concatenate(ts); lab = np.concatenate(lab)
print("rows:", len(ts), "(X_full has 9,536,823)")
print("label values:", pd.Series(lab).value_counts().head(6).to_dict())
print("ts range:", pd.to_datetime(ts.min(), unit="s"), "->", pd.to_datetime(ts.max(), unit="s"))
print("share of rows where time goes backwards:", float((np.diff(ts) < 0).mean()))
np.save(OUT / "ts_full.npy", ts)

b = pd.read_csv("all_attack_bursts.csv")
for base in (0, 1):
    ok = np.isclose(ts[b["csv_row"].values - base], b["ts"].values).mean()
    print(f"attack rows match csv_row-{base}: {ok:.4f}")
b["t"] = pd.to_datetime(b["ts"], unit="s")
g = b.groupby("subLabel").agg(n=("ts", "size"), bursts=("burst_id", "nunique"), first=("t", "min"), last=("t", "max"))
print(g.sort_values("first").to_string())
print("attack rows:", len(b), "| attack time span covers", f"{(b['ts'].max()-b['ts'].min())/(ts.max()-ts.min()):.1%} of the capture")
