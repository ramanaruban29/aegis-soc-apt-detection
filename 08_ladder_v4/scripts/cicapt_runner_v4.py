import os, sys, json, time, argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, pandas as pd, joblib

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, choices=["row", "random5", "random120", "chrono", "loso"])
ap.add_argument("--benign", default="block", choices=["random", "block"])
ap.add_argument("--seeds", type=int, default=1)
ap.add_argument("--feature-names", required=True)
ap.add_argument("--jobs", type=int, default=6)
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--all-baselines", action="store_true")
ap.add_argument("--out", default=None)
a = ap.parse_args()

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(a.jobs)
tf.config.threading.set_inter_op_parallelism_threads(2)
from tensorflow import keras
from tensorflow.keras import layers
import lightgbm as lgb, xgboost as xgb
from sklearn import ensemble
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

UNDER, AE_CAP, BLOCK_S = 50, 800_000, 300.0
FPRS = (2e-5, 4e-5, 8e-5)
GRID = np.round(np.arange(0.001, 0.999, 0.001), 4)
EPOCHS = 3 if a.smoke else 100
OUT = a.out or f"experiments_v4/cicapt_{a.mode}_{a.benign}" + ("_smoke" if a.smoke else "")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- data
X = np.load("X_full.npy", mmap_mode="r"); N = X.shape[0]          # col 0 = OLD anomaly score (never used); cols 1..63 = raw features
ts = np.load("experiments_v4/ts_full.npy"); assert len(ts) == N
b = pd.read_csv("all_attack_bursts.csv")
att_rows = b["csv_row"].values.astype(np.int64); ts_att = b["ts"].values.astype(float); stage = b["subLabel"].values.astype(str)
assert len(att_rows) == 1004 and np.allclose(ts[att_rows], ts_att, rtol=0, atol=1e-3), "attack list does not align with timestamps"
y = np.zeros(N, np.uint8); y[att_rows] = 1
T0 = float(ts.min())
ben_all = np.where(y == 0)[0]
if a.smoke: ben_all = np.sort(np.random.default_rng(0).choice(ben_all, 400_000, replace=False))

names = json.load(open(a.feature_names)); names = names["feature_names"] if isinstance(names, dict) else names
ports = [i for i, nme in enumerate(names[:64]) if "port" in nme.lower()]
print("port columns:", [(i, names[i]) for i in ports], flush=True)
assert len(ports) == 2 and all(1 <= i <= 63 for i in ports), "port columns not found as expected -- send me this output"
keep_raw = [j for j in range(63) if (j + 1) not in ports]
assert len(keep_raw) == 61

base = joblib.load("cicapt_lgbm_v2.pkl").get_params()
tuned = json.load(open("experiments_v2/tuned_params.json"))
def lgb_base(seed):
    p = dict(base); p.update(scale_pos_weight=1, verbose=-1, n_jobs=a.jobs, random_state=seed)
    for k in [k for k in p if "early_stopping" in k]: p[k] = None
    if a.smoke: p["n_estimators"] = 50
    return p
def factory(name, seed):
    if name == "lgbm_tuned": return lgb.LGBMClassifier(**{**lgb_base(seed), **tuned["lgbm"]["params"]})
    if name == "xgb": return xgb.XGBClassifier(tree_method="hist", n_jobs=a.jobs, random_state=seed, **tuned["xgb"]["params"])
    if name == "rf": return ensemble.RandomForestClassifier(random_state=seed, n_jobs=a.jobs, **tuned["rf"]["params"])
    if name == "et": return ensemble.ExtraTreesClassifier(random_state=seed, n_jobs=a.jobs, **tuned["et"]["params"])
    if name == "gb": return ensemble.GradientBoostingClassifier(random_state=seed, **tuned["gb"]["params"])
    if name == "ada": return ensemble.AdaBoostClassifier(random_state=seed, **tuned["ada"]["params"])
BASELINES = ["ada", "rf"] + (["xgb", "et", "gb", "lgbm_tuned"] if a.all_baselines else [])

# ---------------------------------------------------------------- splits
def partition(ids, rng, props):
    nb = int(ids.max()) + 1; sizes = np.bincount(ids, minlength=nb)
    target, counts, assign = sizes.sum() * np.asarray(props, float), np.zeros(len(props)), np.zeros(nb, dtype=int)
    for bb in rng.permutation(nb):
        c = int(np.argmax(target - counts)); assign[bb] = c; counts[c] += sizes[bb]
    return assign[ids]

def bursts_from_gap(t, gap):
    order = np.argsort(t, kind="stable"); new = np.concatenate([[True], np.diff(t[order]) > gap])
    ids = np.empty(len(t), dtype=int); ids[order] = np.cumsum(new) - 1
    return ids

def chrono_attack_split():
    ids = bursts_from_gap(ts_att, 5.0); nb = int(ids.max()) + 1
    sizes = np.bincount(ids, minlength=nb); start = np.array([ts_att[ids == k].min() for k in range(nb)])
    order = np.argsort(start); mid = (np.cumsum(sizes[order]) - sizes[order] / 2) / sizes.sum()
    part = np.where(mid < 0.6, 0, np.where(mid < 0.8, 1, 2)); assign = np.empty(nb, dtype=int); assign[order] = part
    return assign[ids], (float(start[order][part == 1].min()), float(start[order][part == 2].min()))

def benign_split(rng, cuts=None):
    if cuts is not None:
        tb = ts[ben_all]; return ben_all[tb < cuts[0]], ben_all[(tb >= cuts[0]) & (tb < cuts[1])], ben_all[tb >= cuts[1]]
    if a.benign == "random":
        p = rng.permutation(ben_all); c1, c2 = int(.6 * len(p)), int(.8 * len(p))
        return np.sort(p[:c1]), np.sort(p[c1:c2]), np.sort(p[c2:])
    blk = ((ts[ben_all] - T0) // BLOCK_S).astype(int); sp = partition(blk, rng, (.6, .2, .2))
    return ben_all[sp == 0], ben_all[sp == 1], ben_all[sp == 2]

# ---------------------------------------------------------------- autoencoder / features
def raw(rows):
    rows = np.asarray(rows); out = np.empty((len(rows), 63), np.float32)
    for i in range(0, len(rows), 400_000): out[i:i + 400_000] = X[rows[i:i + 400_000], 1:64]
    return out

def fit_ae(Z, seed):
    keras.utils.set_random_seed(seed)
    Ztr, Zva = train_test_split(Z, test_size=0.1, random_state=seed)
    n = Z.shape[1]; inp = keras.Input(shape=(n,))
    x = layers.Dense(48, activation="relu")(inp); x = layers.BatchNormalization()(x); x = layers.Dropout(0.1)(x)
    x = layers.Dense(32, activation="relu")(x); x = layers.BatchNormalization()(x)
    x = layers.Dense(16, activation="relu")(x); x = layers.BatchNormalization()(x)
    x = layers.Dense(8, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x); x = layers.BatchNormalization()(x)
    x = layers.Dense(32, activation="relu")(x); x = layers.BatchNormalization()(x)
    x = layers.Dense(48, activation="relu")(x); x = layers.Dropout(0.1)(x)
    ae = keras.Model(inp, layers.Dense(n, activation="linear")(x)); ae.compile(optimizer="adam", loss="mse")
    h = ae.fit(Ztr, Ztr, validation_data=(Zva, Zva), epochs=EPOCHS, batch_size=256, verbose=0,
               callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)])
    return ae, len(h.history["loss"])

def err(ae, sc, R):
    out = np.empty(len(R))
    for i in range(0, len(R), 500_000):
        z = np.clip(sc.transform(R[i:i + 500_000]), -10, 10).astype(np.float32)
        out[i:i + 500_000] = ((z - ae.predict(z, batch_size=8192, verbose=0)) ** 2).mean(1)
    return out

def feats(R, E, var, clip):
    if var == "noscore": return R
    s = np.minimum(E, clip) if var == "full_clip" else E
    if var == "noports": return np.column_stack([R[:, keep_raw], s]).astype(np.float32)
    return np.column_stack([R, s]).astype(np.float32)

def predict(m, R, E, var, clip, ch=400_000):
    out = np.empty(len(R))
    for i in range(0, len(R), ch): out[i:i + ch] = m.predict_proba(feats(R[i:i + ch], E[i:i + ch], var, clip))[:, 1]
    return out

# ---------------------------------------------------------------- evaluation
def mcc(tp, fp, fn, tn):
    d = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)); return float((tp * tn - fp * fn) / d) if d > 0 else 0.0

def evaluate(vb, va, tb, ta, st, rule=True):
    out = {}
    for f in FPRS:
        t = float(np.quantile(vb, 1 - f)); tp = int((ta >= t).sum()); fp = int((tb >= t).sum()); fn = len(ta) - tp; tn = len(tb) - fp
        d = dict(mcc=mcc(tp, fp, fn, tn), recall=tp / len(ta), fp=fp, tp=tp, thr=t)
        if abs(f - 4e-5) < 1e-12: d["stage_recall"] = {str(s): float((ta[st == s] >= t).mean()) for s in np.unique(st)}
        out[f"fpr{f:g}"] = d
    if rule and len(va):
        sb, sa = np.sort(vb), np.sort(va)
        fpv = len(sb) - np.searchsorted(sb, GRID, "left"); tpv = len(sa) - np.searchsorted(sa, GRID, "left"); fnv = len(sa) - tpv; tnv = len(sb) - fpv
        den = np.sqrt((tpv + fpv).astype(float) * (tpv + fnv) * (tnv + fpv) * (tnv + fnv))
        m = np.where(den > 0, (tpv * tnv - fpv * fnv) / np.where(den > 0, den, 1.0), 0.0)
        t = float(GRID[int(np.argmax(m))]); tp = int((ta >= t).sum()); fp = int((tb >= t).sum())
        out["mccrule"] = dict(mcc=mcc(tp, fp, len(ta) - tp, len(tb) - fp), recall=tp / len(ta), fp=fp, tp=tp, thr=t)
    return out

# ---------------------------------------------------------------- one seed
def make_ctx(btr, bva, bte, seed, rng):
    t0 = time.time(); n = min(len(btr), 50_000 if a.smoke else AE_CAP)
    Rfit = raw(np.sort(rng.choice(btr, n, replace=False)))
    sc = RobustScaler().fit(Rfit); Z = np.clip(sc.transform(Rfit), -10, 10).astype(np.float32)
    ae, ep = fit_ae(Z, seed); clip = float(np.percentile(err(ae, sc, Rfit), 99)); del Z
    Rvb, Rtb = raw(bva), raw(bte)
    print(f"  AE fitted: {ep} epochs, {time.time() - t0:.0f}s | benign val/test {len(bva)}/{len(bte)}", flush=True)
    return dict(btr=btr, ae=ae, sc=sc, clip=clip, epochs=ep, Rvb=Rvb, Rtb=Rtb, Evb=err(ae, sc, Rvb), Etb=err(ae, sc, Rtb))

def run_one(tag, sp, ctx, seed):
    t0 = time.time(); i_tr, i_va, i_te = (np.where(sp == k)[0] for k in range(3))
    assert len(i_tr) > 0 and len(i_te) > 0
    rng = np.random.default_rng(1000 + seed); clip = ctx["clip"]; ae, sc = ctx["ae"], ctx["sc"]
    keep = np.sort(rng.choice(ctx["btr"], UNDER * len(i_tr), replace=False))
    Ra = raw(att_rows[i_tr]); Rtr = np.vstack([Ra, raw(keep)]); ytr = np.r_[np.ones(len(Ra), int), np.zeros(len(keep), int)]
    Etr = err(ae, sc, Rtr)
    Rva_a, Rte_a = raw(att_rows[i_va]), raw(att_rows[i_te]); Eva_a, Ete_a = err(ae, sc, Rva_a), err(ae, sc, Rte_a)
    st_te = stage[i_te]; res = {}; trees = {}
    res["ae_only"] = evaluate(ctx["Evb"], Eva_a, ctx["Etb"], Ete_a, st_te, rule=False)
    def fit_eval(name, model, var):
        model.fit(feats(Rtr, Etr, var, clip), ytr)
        vb = predict(model, ctx["Rvb"], ctx["Evb"], var, clip); tb = predict(model, ctx["Rtb"], ctx["Etb"], var, clip)
        va = predict(model, Rva_a, Eva_a, var, clip) if len(Rva_a) else np.array([]); ta = predict(model, Rte_a, Ete_a, var, clip)
        res[name] = evaluate(vb, va, tb, ta, st_te)
        if hasattr(model, "booster_"): trees[name] = int(model.booster_.num_trees())
        print(f"   {name:15s} {time.time() - t0:5.0f}s  FPR4e-5 MCC {res[name]['fpr4e-05']['mcc']:.3f} recall {res[name]['fpr4e-05']['recall']:.3f} FP {res[name]['fpr4e-05']['fp']}", flush=True)
    for name, var in (("lgbm_full", "full"), ("lgbm_full_clip", "full_clip"), ("lgbm_noscore", "noscore"), ("lgbm_noports", "noports")):
        fit_eval(name, lgb.LGBMClassifier(**lgb_base(seed)), var)
    for name in BASELINES: fit_eval(name, factory(name, seed), "full")
    info = dict(mode=a.mode, benign=a.benign, seed=seed, tag=tag, attacks=[int(len(i_tr)), int(len(i_va)), int(len(i_te))],
                test_stages={str(s): int((st_te == s).sum()) for s in np.unique(st_te)}, benign_sizes=[int(len(ctx["btr"])), int(len(ctx["Rvb"])), int(len(ctx["Rtb"]))],
                ae_epochs=int(ctx["epochs"]), clip=clip, trees=trees, ports=[names[i] for i in ports], secs=time.time() - t0)
    json.dump({"info": info, "res": res}, open(f"{OUT}/{tag}_seed{seed:02d}.json", "w"))
    print(f"[{tag} seed {seed}] done in {time.time() - t0:.0f}s | attacks tr/va/te {info['attacks']}", flush=True)

stage_order = sorted(set(stage), key=lambda s: ts_att[stage == s].min())
for seed in range(a.seeds):
    rng = np.random.default_rng(seed); print(f"== seed {seed} mode {a.mode} benign {a.benign}", flush=True)
    if a.mode in ("row", "random5", "random120"):
        sp = partition(np.arange(len(ts_att)) if a.mode == "row" else bursts_from_gap(ts_att, 5.0 if a.mode == "random5" else 120.0), rng, (.6, .2, .2))
        btr, bva, bte = benign_split(rng); jobs = [("split", sp)]
    elif a.mode == "chrono":
        sp, cuts = chrono_attack_split(); btr, bva, bte = benign_split(rng, cuts); jobs = [("split", sp)]
    else:
        btr, bva, bte = benign_split(rng); jobs = []
        for s in stage_order:
            te = stage == s; sp = np.full(len(stage), 2); sp[~te] = partition(bursts_from_gap(ts_att[~te], 5.0), rng, (.75, .25))
            jobs.append((f"loso_{s.replace(' ', '_')}", sp))
    todo = [(t, sp) for t, sp in jobs if not os.path.exists(f"{OUT}/{t}_seed{seed:02d}.json")]
    if not todo: continue
    ctx = make_ctx(btr, bva, bte, seed, rng)
    for t, sp in todo: run_one(t, sp, ctx, seed)
