#!/usr/bin/env python3
"""Cross-dataset test for the paper: train on one dataset, test on the other (both directions).

  python3 cross_dataset.py --check                      # only print the shared feature list, then stop
  python3 cross_dataset.py --seeds 10                   # full run, both directions, with the AE score
  python3 cross_dataset.py --seeds 10 --noae            # faster run without the autoencoder (LightGBM etc. only)
  python3 cross_dataset.py --summarize                  # print the table from the saved JSON files

Run it from the folder that holds X_full.npy (CICAPT-IIoT arrays).  Edit the PATHS block below if your files are elsewhere.
Design: the source dataset gives the training data AND the threshold (false-positive-rate rule on source benign validation flows,
no target labels are used).  The whole target dataset is the test set.  Ports are always dropped here, because they are a
testbed shortcut (paper, Section 7.5).  Two scalings are reported: 'src' (scaler fitted on source benign flows) and 'tgt'
(scaler fitted on UNLABELLED target flows that look benign by construction, i.e. all target flows -- no labels used)."""
import argparse, json, os, re, sys, time, glob
import numpy as np, pandas as pd
# ------------------------------------------------------------------ PATHS (edit if needed)
PATHS = dict(
    cic_X="../X_full.npy",                         # col 0 = old anomaly score (ignored), cols 1..63 = features
    cic_names="../feature_names.json",             # the file you passed as --feature_names to cicapt_runner_v4.py
    cic_att_csv="../all_attack_bursts.csv",        # has csv_row (row index of each attack flow)
    dapt_csv="/mnt/d/dapt_work/dapt_clean.csv", # has y, sec and the numeric feature columns
    out="/mnt/d/dapt_work/results_cross")
ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--noae", action="store_true")
ap.add_argument("--check", action="store_true"); ap.add_argument("--summarize", action="store_true")
ap.add_argument("--map", default=None, help="optional JSON {cicapt_feature_name: dapt_feature_name} to add manual pairs")
ap.add_argument("--jobs", type=int, default=8); ap.add_argument("--under", type=int, default=50)
ap.add_argument("--cic_benign_cap", type=int, default=2_000_000, help="benign CICAPT flows kept as TARGET test set (all attacks kept)")
a = ap.parse_args(); os.makedirs(PATHS["out"], exist_ok=True)

def mcc(tp, fp, fn, tn):
    d = float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return 0.0 if d == 0 else (tp * tn - fp * fn) / np.sqrt(d)
def metrics(score, y, thr):
    p = score >= thr; tp = int((p & (y == 1)).sum()); fp = int((p & (y == 0)).sum())
    fn = int(((~p) & (y == 1)).sum()); tn = int(((~p) & (y == 0)).sum())
    return dict(mcc=float(mcc(tp, fp, fn, tn)), recall=tp / max(1, tp + fn), fp=fp, tp=tp, fpr=fp / max(1, fp + tn))
def best_mcc(score, y, n=400):          # ORACLE threshold on the target: shows what threshold transfer loses
    qs = np.unique(np.quantile(score, np.linspace(0.5, 0.99999, n))); best = (-1, None)
    for t in qs:
        m = metrics(score, y, t)
        if m["mcc"] > best[0]: best = (m["mcc"], m)
    return best[1]
def ap_score(score, y):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, score))

# ------------------------------------------------------------------ feature matching
REP = [("total", "tot"), ("packets", "pkts"), ("packet", "pkt"), ("forward", "fwd"), ("backward", "bwd"), ("length", "len"),
       ("average", "avg"), ("segment", "seg"), ("bytes", "byts"), ("byte", "byt"), ("per", ""), ("subflow", "subflow"), ("size", "size"),
       ("std", "std"), ("standard deviation", "std"), ("deviation", "std"), ("mean", "avg"), ("iat", "iat"), ("window", "win"), ("initial", "init")]
def norm(s):
    s = str(s).lower().strip()
    for long, short in REP: s = s.replace(long, short)
    return re.sub(r"[^a-z0-9]", "", s)
def load_names():
    n = json.load(open(PATHS["cic_names"])); n = n["feature_names"] if isinstance(n, dict) else n
    return list(n[1:64])
def dapt_features(d):
    skip = ("label", "activity", "stage")
    return [c for c in d.select_dtypes(include=[np.number]).columns if c not in ("y", "sec") and not any(k in c.lower() for k in skip)]
def match(cn, dn, extra=None):
    cmap = {norm(c): i for i, c in enumerate(cn)}; pairs = []
    for j, c in enumerate(dn):
        k = norm(c)
        if k in cmap and "port" not in k: pairs.append((cmap[k], j, cn[cmap[k]], c))
    if extra:
        for c, dcol in extra.items():
            if c in cn and dcol in dn: pairs.append((cn.index(c), dn.index(dcol), c, dcol))
    seen, out = set(), []
    for p in pairs:
        if p[0] in seen: continue
        seen.add(p[0]); out.append(p)
    return out

if a.summarize:
    R = {}
    for f in sorted(glob.glob(PATHS["out"] + "/*.json")):
        j = json.load(open(f)); R.setdefault(j["direction"], []).append(j)
    for dirn, rs in R.items():
        print(f"\n=== {dirn}: {len(rs)} seeds, {rs[0]['n_shared']} shared features, source FPR rule -> target")
        print(f"{'model':22s}{'scaling':8s}{'rule':9s}{'MCC med':>9s}{'recall':>8s}{'FP med':>10s}{'FPR med':>10s}{'oracleMCC':>10s}{'AP':>7s}")
        keys = sorted({(m, s, r) for r0 in rs for m, v in r0["res"].items() for s, v2 in v.items() for r in v2 if r != "oracle" and r != "ap"})
        for m, s, r in keys:
            g = lambda fn: float(np.median([fn(r0["res"][m][s]) for r0 in rs if m in r0["res"] and s in r0["res"][m] and r in r0["res"][m][s]]))
            print(f"{m:22s}{s:8s}{r:9s}{g(lambda x: x[r]['mcc']):9.3f}{g(lambda x: x[r]['recall']):8.3f}{g(lambda x: x[r]['fp']):10.0f}{g(lambda x: x[r]['fpr']):10.5f}{g(lambda x: x['oracle']['mcc']):10.3f}{g(lambda x: x['ap']):7.3f}")
    sys.exit()

# ------------------------------------------------------------------ load both datasets
cn = load_names(); dd = pd.read_csv(PATHS["dapt_csv"], low_memory=False); dn = dapt_features(dd)
extra = json.load(open(a.map)) if a.map else None
pairs = match(cn, dn, extra)
print(f"CICAPT {len(cn)} features, DAPT {len(dn)} features, SHARED (ports excluded): {len(pairs)}")
for p in pairs: print(f"   {p[2]!r:45s} <-> {p[3]!r}")
print("CICAPT unmatched:", [c for i, c in enumerate(cn) if i not in {p[0] for p in pairs}][:60])
print("DAPT unmatched  :", [c for j, c in enumerate(dn) if j not in {p[1] for p in pairs}][:60])
if a.check or len(pairs) < 10:
    if len(pairs) < 10: print("\nToo few shared features (<10). Make a --map JSON pairing the names by hand, e.g. {\"CIC name\": \"DAPT name\"}, and rerun.")
    sys.exit()
ci = [p[0] + 1 for p in pairs]; di = [p[1] for p in pairs]          # +1: CICAPT col 0 is the old score
Xc_all = np.load(PATHS["cic_X"], mmap_mode="r"); Nc = Xc_all.shape[0]
att = pd.read_csv(PATHS["cic_att_csv"])["csv_row"].values.astype(np.int64)
yc_all = np.zeros(Nc, np.uint8); yc_all[att] = 1
Xd = dd[dn].iloc[:, di].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32); yd = dd["y"].values.astype(int)
# target/source subsets of CICAPT: every attack + a random benign sample (reproducible)
rng0 = np.random.default_rng(123); benc = np.where(yc_all == 0)[0]
benc = np.sort(rng0.choice(benc, min(a.cic_benign_cap, len(benc)), replace=False)); rows_c = np.sort(np.concatenate([att, benc]))
Xc = np.asarray(Xc_all[rows_c][:, ci], dtype=np.float32); yc = yc_all[rows_c].astype(int)
print(f"CICAPT subset {len(Xc)} flows ({int(yc.sum())} attacks) | DAPT {len(Xd)} flows ({int(yd.sum())} attacks)")

# ------------------------------------------------------------------ models
import lightgbm as lgb
from sklearn import ensemble
from sklearn.preprocessing import RobustScaler
if not a.noae:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
def fit_ae(Z, seed):
    n = Z.shape[1]; tf.keras.utils.set_random_seed(seed)
    inp = keras.Input((n,)); x = layers.Dense(48, activation="relu")(inp); x = layers.BatchNormalization()(x); x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x); x = layers.Dense(8, activation="relu")(x); x = layers.Dense(16, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x); x = layers.Dense(48, activation="relu")(x)
    ae = keras.Model(inp, layers.Dense(n)(x)); ae.compile(optimizer="adam", loss="mse")
    ae.fit(Z, Z, validation_split=0.1, epochs=100, batch_size=256, verbose=0,
           callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)]); return ae
def ae_err(ae, Z, ch=500_000):
    return np.concatenate([((Z[i:i + ch] - ae.predict(Z[i:i + ch], batch_size=8192, verbose=0)) ** 2).mean(1) for i in range(0, len(Z), ch)])
def make(name, seed):
    if name.startswith("lgbm"): return lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31, min_child_samples=5, subsample=1.0,
                                                          colsample_bytree=0.8, reg_lambda=1.0, scale_pos_weight=1, verbose=-1, n_jobs=a.jobs, random_state=seed)
    if name == "rf": return ensemble.RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=2, n_jobs=a.jobs, random_state=seed)
    if name == "et": return ensemble.ExtraTreesClassifier(n_estimators=300, min_samples_leaf=1, max_features=0.5, n_jobs=a.jobs, random_state=seed)
    if name == "ada": return ensemble.AdaBoostClassifier(n_estimators=300, learning_rate=0.5, random_state=seed)
MODELS = ["lgbm_noscore", "rf", "et", "ada"] + ([] if a.noae else ["lgbm_full"])
def prob(m, F, ch=500_000): return np.concatenate([m.predict_proba(F[i:i + ch])[:, 1] for i in range(0, len(F), ch)])

def run(direction, Xs, ys, Xt, yt, fprs, seed):
    rng = np.random.default_rng(seed); t0 = time.time()
    ib = np.where(ys == 0)[0]; ia = np.where(ys == 1)[0]; rng.shuffle(ib)
    nval = min(len(ib) // 5, 300_000); iv = ib[:nval]; ib_tr = ib[nval:]        # source benign validation (for the FPR rule)
    ntr_b = min(len(ib_tr), a.under * len(ia)) if len(ia) < 5000 else len(ib_tr)  # 50:1 undersample when attacks are rare (CICAPT), else keep all
    itr = np.concatenate([ia, ib_tr[:ntr_b]])
    out = dict(direction=direction, seed=seed, n_shared=Xs.shape[1], n_train=int(len(itr)), n_train_att=int(len(ia)), n_val_benign=int(len(iv)), res={})
    for scaling in ("src", "tgt"):
        sc = RobustScaler().fit(Xs[ib_tr] if scaling == "src" else Xt)               # 'tgt' uses unlabelled target flows only
        S = lambda X, sc=sc: np.clip(sc.transform(X), -10, 10).astype(np.float32)
        # the classifier is trained on the SOURCE scaled with the SOURCE scaler in both cases; 'tgt' only changes how the target is scaled
        scs = RobustScaler().fit(Xs[ib_tr]); SS = lambda X: np.clip(scs.transform(X), -10, 10).astype(np.float32)
        Zs_tr, Zs_v = SS(Xs[itr]), SS(Xs[iv]); Zt = S(Xt) if scaling == "tgt" else SS(Xt)
        if not a.noae:
            ae = fit_ae(SS(Xs[ib_tr[: min(len(ib_tr), 400_000)]]), seed)
            Es_tr, Es_v, Et = ae_err(ae, Zs_tr), ae_err(ae, Zs_v), ae_err(ae, Zt)
        for name in MODELS:
            full = name == "lgbm_full"
            Ftr = np.column_stack([Zs_tr, Es_tr]) if full else Zs_tr
            Fv = np.column_stack([Zs_v, Es_v]) if full else Zs_v
            Ft = np.column_stack([Zt, Et]) if full else Zt
            m = make(name, seed).fit(Ftr, ys[itr]); pv, pt = prob(m, Fv), prob(m, Ft)
            r = {}
            for f in fprs:
                thr = float(np.quantile(pv, 1 - f)); r[f"fpr{f:g}"] = metrics(pt, yt, thr)
            r["oracle"] = best_mcc(pt, yt); r["ap"] = ap_score(pt, yt); out["res"].setdefault(name, {})[scaling] = r
            print(f"  [{direction} seed {seed} {scaling}] {name:13s} " + " | ".join(f"{k}: MCC {v['mcc']:.3f} rec {v['recall']:.3f} FP {v['fp']}" for k, v in r.items() if k.startswith("fpr")) + f" | oracle {r['oracle']['mcc']:.3f}", flush=True)
        if not a.noae:                       # AE alone (benign-only detector) on the target
            pv, pt = Es_v, Et; r = {f"fpr{f:g}": metrics(pt, yt, float(np.quantile(pv, 1 - f))) for f in fprs}
            r["oracle"] = best_mcc(pt, yt); r["ap"] = ap_score(pt, yt); out["res"].setdefault("ae_only", {})[scaling] = r
    out["secs"] = time.time() - t0; return out

for seed in range(a.seeds):
    for direction, (Xs, ys, Xt, yt, fprs) in {"DAPT_to_CICAPT": (Xd, yd, Xc, yc, [0.002, 0.005, 0.01]),
                                              "CICAPT_to_DAPT": (Xc, yc, Xd, yd, [2e-5, 4e-5, 8e-5])}.items():
        fo = f"{PATHS['out']}/{direction}_seed{seed:02d}.json"
        if os.path.exists(fo): continue
        json.dump(run(direction, Xs, ys, Xt, yt, fprs, seed), open(fo, "w")); print(f"saved {fo}", flush=True)
print("all done. Summary:  python3 cross_dataset.py --summarize")