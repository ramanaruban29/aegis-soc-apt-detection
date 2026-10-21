import glob, pandas as pd
fs = sorted(glob.glob("/mnt/d/DAPT2020/csv/*.csv"))
ref = next(list(pd.read_csv(f, nrows=0).columns) for f in fs if "Flow ID" in pd.read_csv(f, nrows=0).columns)
parts = []
for f in fs:
    d = pd.read_csv(f, low_memory=False); fix = ""
    if "Flow ID" not in d.columns:
        d = pd.read_csv(f, header=None, names=ref, low_memory=False); fix = "(header was missing)"
    d["__file"] = f.split("/")[-1].replace(".pcap_Flow.csv", ""); parts.append(d)
    print(f"{d['__file'].iloc[0]:28s} rows {len(d):6d} {fix}")
d = pd.concat(parts, ignore_index=True)
print("shape:", d.shape, "| missing Activity:", int(d["Activity"].isna().sum()))
act = d["Activity"].astype(str).str.strip()
d["y"] = (~act.str.lower().isin(["normal", "benign"])).astype(int)
ts = pd.to_datetime(d["Timestamp"], format="%d/%m/%Y %I:%M:%S %p", errors="coerce")
print("unparsable timestamps:", int(ts.isna().sum()))
d["sec"] = (ts - pd.Timestamp("1970-01-01")).dt.total_seconds()
print("attack flows:", int(d.y.sum()), f"({d.y.mean():.1%}) | benign:", int((1 - d.y).sum()))
print(d.groupby("__file")["sec"].agg(["min", "max"]).apply(lambda c: pd.to_datetime(c, unit="s")).to_string())
a = d[d.y == 1].sort_values("sec").copy()
a["burst"] = (a["sec"].diff().fillna(1e9) > 5).cumsum()
g = a.groupby("burst").agg(n=("sec", "size"), start=("sec", "min"), acts=("Activity", lambda s: s.value_counts().head(2).to_dict()))
print("attack bursts (5 s gap):", len(g), "| largest burst share of attack flows:", f"{g.n.max()/g.n.sum():.1%}")
print(g.sort_values("n", ascending=False).head(8).assign(start=lambda x: pd.to_datetime(x.start, unit="s")).to_string())
print("bursts with >=10 flows:", int((g.n >= 10).sum()), "| single-flow bursts:", int((g.n == 1).sum()))
d.to_csv("/mnt/d/dapt_work/dapt_clean.csv", index=False)
