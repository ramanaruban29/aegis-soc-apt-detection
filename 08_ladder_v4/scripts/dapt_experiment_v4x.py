import os, sys, json, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, pandas as pd, joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import lightgbm as lgb, xgboost as xgb
from sklearn import ensemble
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPClassifier

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 1
GAP = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
MODE = sys.argv[3] if len(sys.argv) > 3 else "random"   # random = redrawn burst split; chrono = time-ordered; row = per-flow groups
BENIGN = sys.argv[4] if len(sys.argv) > 4 else "random"  # random | block
assert MODE in ("random", "chrono", "row") and BENIGN in ("random", "block")
BLOCK_S = 300.0
OUT = f"/mnt/d/dapt_work/results_v4_{MODE}_gap{GAP:g}" + ("_block" if BENIGN == "block" else ""); os.makedirs(OUT, exist_ok=True)
FPRS = [0.002, 0.005, 0.01]
GRID = np.round(np.arange(0.001, 0.999, 0.001), 4)

d = pd.read_csv("/mnt/d/dapt_work/dapt_clean.csv", low_memory=False)
y = d["y"].values.astype(int); sec = d["sec"].values
skip = ("label", "activity", "stage")
feat = [c for c in d.select_dtypes(include=[np.number]).columns if c not in ("y", "sec") and not any(k in c.lower() for k in skip)]
ports = [i for i, c in enumerate(feat) if c.strip().lower() in ("src port", "dst port")]
print(len(feat), "features | port columns:", [feat[i] for i in ports]); print(feat)
assert len(ports) == 2
X = d[feat].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)

att = np.where(y == 1)[0]
o = att[np.argsort(sec[att], kind="stable")]
new = np.concatenate([[True], np.diff(sec[o]) > GAP])
bid = np.zeros(len(d), dtype=int); bid[o] = np.cumsum(new) - 1
if MODE == "row": bid[att] = np.arange(len(att))
print("gap", GAP, "| attack flows", len(att), "| bursts", int(new.sum()), "| mode", MODE, "| benign", BENIGN)

def partition(burst_id, rng, props=(0.6, 0.2, 0.2)):
    nb = int(burst_id.max()) + 1
    sizes = np.bincount(burst_id, minlength=nb)
    target, counts, assign = sizes.sum() * np.asarray(props), np.zeros(3), np.zeros(nb, dtype=int)
    for b in rng.permutation(nb):
        c = int(np.argmax(target - counts)); assign[b] = c; counts[c] += sizes[b]
    return assign[burst_id]

def chrono_split():
    a_bid = bid[att]; nb = int(a_bid.max()) + 1
    sizes = np.bincount(a_bid, minlength=nb)
    start = np.array([sec[att][a_bid == b].min() for b in range(nb)])
    order = np.argsort(start)
    mid = (np.cumsum(sizes[order]) - sizes[order] / 2) / sizes.sum()
    part = np.where(mid < 0.6, 0, np.where(mid < 0.8, 1, 2))
    assign = np.empty(nb, dtype=int); assign[order] = part
    sp = assign[a_bid]
    assert all((sp == k).any() for k in range(3)), 'a partition has no attacks: send me this output'
    t1 = start[order][part == 1].min(); t2 = start[order][part == 2].min()
    ben = np.where(y == 0)[0]; sb = sec[ben]
    btr, bva, bte = ben[sb < t1], ben[(sb >= t1) & (sb < t2)], ben[sb >= t2]
    assert min(len(btr), len(bva), len(bte)) > 100, 'too few benign flows in a partition: send me this output'
    tr = np.concatenate([att[sp == 0], btr]); va = np.concatenate([att[sp == 1], bva]); te = np.concatenate([att[sp == 2], bte])
    print('chrono cuts:', pd.to_datetime(t1, unit='s'), pd.to_datetime(t2, unit='s'))
    print('attacks train/val/test:', [int((sp == k).sum()) for k in range(3)], '| benign:', len(btr), len(bva), len(bte), flush=True)
    return sp, tr, va, te, len(btr), len(btr) + len(bva), np.concatenate([btr, bva, bte])

def mccf(tp, fp, fn, tn):
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / den) if den > 0 else 0.0

def at(p, yy, t):
    tp = int((p[yy == 1] >= t).sum()); fp = int((p[yy == 0] >= t).sum())
    fn = int((yy == 1).sum()) - tp; tn = int((yy == 0).sum()) - fp
    return dict(mcc=mccf(tp, fp, fn, tn), recall=tp / max(tp + fn, 1), fp=fp, tp=tp)

def evaluate(pv, yv, pt, yt):
    out = {}; bv = np.sort(pv[yv == 0]); av = np.sort(pv[yv == 1])
    for f in FPRS: out[f"fpr{f}"] = at(pt, yt, float(np.quantile(bv, 1 - f)))
    fp = len(bv) - np.searchsorted(bv, GRID, "left"); tp = len(av) - np.searchsorted(av, GRID, "left")
    fn = len(av) - tp; tn = len(bv) - fp
    den = np.sqrt((tp + fp).astype(float) * (tp + fn) * (tn + fp) * (tn + fn))
    m = np.where(den > 0, (tp * tn - fp * fn) / np.where(den > 0, den, 1.0), 0.0)
    t = float(GRID[int(np.argmax(m))]); out["mccrule"] = {**at(pt, yt, t), "thr": t}
    return out

def fit_ae(Zb, seed):
    keras.utils.set_random_seed(seed)
    Ztr, Zva = train_test_split(Zb, test_size=0.1, random_state=seed)
    n = Zb.shape[1]; inp = keras.Input(shape=(n,))
    x = layers.Dense(48, activation="relu")(inp); x = layers.BatchNormalization()(x); x = layers.Dropout(0.1)(x)
    x = layers.Dense(32, activation="relu")(x); x = layers.BatchNormalization()(x)
    x = layers.Dense(16, activation="relu")(x); x = layers.BatchNormalization()(x)
    x = layers.Dense(8, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x); x = layers.BatchNormalization()(x)
    x = layers.Dense(32, activation="relu")(x); x = layers.BatchNormalization()(x)
    x = layers.Dense(48, activation="relu")(x); x = layers.Dropout(0.1)(x)
    ae = keras.Model(inp, layers.Dense(n, activation="linear")(x)); ae.compile(optimizer="adam", loss="mse")
    ae.fit(Ztr, Ztr, validation_data=(Zva, Zva), epochs=100, batch_size=256, verbose=0,
           callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)])
    return ae

tuned = json.load(open("experiments_v2/tuned_params.json"))
base = joblib.load("cicapt_lgbm_v2.pkl").get_params(); base["scale_pos_weight"] = 1; base["verbose"] = -1
def factory(name, seed):
    p = tuned.get(name.split("_")[0], {}).get("params", {})
    if name == "lgbm_deployed": return lgb.LGBMClassifier(**base)
    if name == "lgbm_tuned": return lgb.LGBMClassifier(**{**base, **tuned["lgbm"]["params"]})
    if name == "xgb": return xgb.XGBClassifier(tree_method="hist", n_jobs=-1, random_state=seed, **tuned["xgb"]["params"])
    if name == "rf": return ensemble.RandomForestClassifier(random_state=seed, n_jobs=-1, **tuned["rf"]["params"])
    if name == "et": return ensemble.ExtraTreesClassifier(random_state=seed, n_jobs=-1, **tuned["et"]["params"])
    if name == "gb": return ensemble.GradientBoostingClassifier(random_state=seed, **tuned["gb"]["params"])
    if name == "ada": return ensemble.AdaBoostClassifier(random_state=seed, **tuned["ada"]["params"])
    if name == "mlp": return MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True, n_iter_no_change=10, max_iter=200, random_state=seed)

def unsup(model, Zb, Zva, Zte, cap=None, seed=0):
    if cap and len(Zb) > cap: Zb = Zb[np.random.default_rng(seed).choice(len(Zb), cap, replace=False)]
    model.fit(Zb); sb = -model.score_samples(Zb)
    f = lambda Z: 0.5 + np.arctan((-model.score_samples(Z) - sb.mean()) / (sb.std() + 1e-12)) / np.pi
    return f(Zva), f(Zte)

for seed in range(N_SEEDS):
    fo = f"{OUT}/seed{seed:02d}.json"
    if os.path.exists(fo): continue
    rng = np.random.default_rng(seed); t0 = time.time()
    sp = partition(bid[att], rng)
    ben = rng.permutation(np.where(y == 0)[0]); c1, c2 = int(0.6 * len(ben)), int(0.8 * len(ben))
    if BENIGN == "block":
        b0 = np.where(y == 0)[0]; blk = ((sec[b0] - sec.min()) // BLOCK_S).astype(int); _, blk = np.unique(blk, return_inverse=True)
        sb = partition(blk, rng); ben = np.concatenate([b0[sb == 0], b0[sb == 1], b0[sb == 2]])
        c1, c2 = int((sb == 0).sum()), int((sb == 0).sum() + (sb == 1).sum())
    tr = np.concatenate([att[sp == 0], ben[:c1]]); va = np.concatenate([att[sp == 1], ben[c1:c2]]); te = np.concatenate([att[sp == 2], ben[c2:]])
    if MODE == "chrono": sp, tr, va, te, c1, c2, ben = chrono_split()
    ytr, yva, yte = y[tr], y[va], y[te]
    sc = RobustScaler().fit(X[tr][ytr == 0]); Z = np.clip(sc.transform(X), -10, 10).astype(np.float32)
    ae = fit_ae(Z[tr][ytr == 0], seed)
    E = ((Z - ae.predict(Z, batch_size=8192, verbose=0)) ** 2).mean(1)
    clip = float(np.percentile(E[tr][ytr == 0], 99)); S = np.minimum(E, clip)
    F = np.column_stack([X, S]); nf = F.shape[1]
    sets = {"full": list(range(nf)), "noscore": list(range(nf - 1)), "noports": [i for i in range(nf) if i not in ports]}
    res = {}
    res["ae_only"] = evaluate(S[va] / clip, yva, S[te] / clip, yte)
    Eu = E / (E + np.median(E[tr][ytr == 0]))     # monotone map of the UNCLIPPED error into [0,1); use the FPR rows only
    res["ae_only_unclipped"] = evaluate(Eu[va], yva, Eu[te], yte)
    print("after ae+prep", int(time.time()-t0), flush=True)
    res["iforest"] = evaluate(yva, *[0] * 0, *[]) if False else None
    pv, pt = unsup(IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1), Z[tr][ytr == 0], Z[va], Z[te])
    res["iforest"] = evaluate(pv, yva, pt, yte)
    print("after iforest", int(time.time()-t0), flush=True)
    pv, pt = unsup(OneClassSVM(kernel="rbf", gamma="scale", nu=0.01), Z[tr][ytr == 0], Z[va], Z[te], cap=10000, seed=seed)
    res["ocsvm"] = evaluate(pv, yva, pt, yte)
    print("after ocsvm", int(time.time()-t0), flush=True)
    Fu = np.column_stack([X, Eu]); mu = factory("lgbm_deployed", seed); mu.fit(Fu[tr], ytr)
    res["lgbm_full_unclip"] = evaluate(mu.predict_proba(Fu[va])[:, 1], yva, mu.predict_proba(Fu[te])[:, 1], yte)
    prng = np.random.default_rng(2000 + seed); Fte_s = F[te].copy()
    for pc in ports: Fte_s[:, pc] = Fte_s[prng.permutation(len(te)), pc]      # port columns shuffled across all test flows
    for tag, name, key in [("lgbm_full", "lgbm_deployed", "full"), ("lgbm_noscore", "lgbm_deployed", "noscore"), ("lgbm_noports", "lgbm_deployed", "noports"),
                           ("lgbm_tuned", "lgbm_tuned", "full"), ("xgb", "xgb", "full"), ("rf", "rf", "full"), ("et", "et", "full"),
                           ("gb", "gb", "full"), ("ada", "ada", "full"), ("mlp", "mlp", "full")]:
        cols = sets[key]; m = factory(name, seed); m.fit(F[tr][:, cols], ytr)
        pv_ = m.predict_proba(F[va][:, cols])[:, 1]
        res[tag] = evaluate(pv_, yva, m.predict_proba(F[te][:, cols])[:, 1], yte)
        if key != "noports":      # thresholds from the unshuffled validation scores; only the test ports are shuffled
            res[tag + "_portshuf"] = evaluate(pv_, yva, m.predict_proba(Fte_s[:, cols])[:, 1], yte)
        print(tag, int(time.time()-t0), flush=True)
    info = {"gap": GAP, "attacks": [int((sp == k).sum()) for k in range(3)], "bursts": [int(len(set(bid[att][sp == k]))) for k in range(3)],
            "benign": [c1, c2 - c1, len(ben) - c2], "clip": clip}
    json.dump({"info": info, "res": res}, open(fo, "w"))
    print(f"seed {seed} {time.time()-t0:.0f}s attacks/split {info['attacks']} bursts {info['bursts']} | test MCC at FPR 0.5%: " +
          " ".join(f"{k} {v['fpr0.005']['mcc']:.3f}" for k, v in res.items()), flush=True)
