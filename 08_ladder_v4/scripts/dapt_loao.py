import sys, os, json, time
src = open("dapt_experiment_v4.py").read()
head = src.split("for seed in range(N_SEEDS):")[0]
assert head != src, "marker not found"
NS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
sys.argv = ["x", "1", "120", "random"]
g = {"__name__": "dapt_head"}
exec(compile(head, "dapt_experiment_v4.py", "exec"), g)
np, pd = g["np"], g["pd"]
X, y, d, att, bid = g["X"], g["y"], g["d"], g["att"], g["bid"]
evaluate, fit_ae, factory, ports, RobustScaler = g["evaluate"], g["fit_ae"], g["factory"], g["ports"], g["RobustScaler"]
act = d["Activity"].astype(str).str.strip().values
OUT2 = "/mnt/d/dapt_work/results_v4_loao_gap120"; os.makedirs(OUT2, exist_ok=True)
cnt = pd.Series(act[att]).value_counts()
BIG = [a for a in cnt.index if cnt[a] >= 2000]
print("held-out activities:", BIG, flush=True)

def part2(ids, rng, props=(0.75, 0.25)):
    u, inv = np.unique(ids, return_inverse=True)
    sizes = np.bincount(inv); target = sizes.sum() * np.asarray(props, float)
    counts = np.zeros(len(props)); assign = np.zeros(len(u), int)
    for b in rng.permutation(len(u)):
        c = int(np.argmax(target - counts)); assign[b] = c; counts[c] += sizes[b]
    return assign[inv]

MODELS = [("lgbm_full", "lgbm_deployed", "full"), ("lgbm_noscore", "lgbm_deployed", "noscore"),
          ("lgbm_noports", "lgbm_deployed", "noports"), ("lgbm_tuned", "lgbm_tuned", "full"),
          ("et", "et", "full"), ("rf", "rf", "full"), ("xgb", "xgb", "full"), ("gb", "gb", "full"), ("ada", "ada", "full")]

for seed in range(NS):
    for A in BIG:
        fo = f"{OUT2}/{A.replace(' ', '_')}_seed{seed:02d}.json"
        if os.path.exists(fo): continue
        t0 = time.time(); rng = np.random.default_rng(seed * 100 + 7)
        te_a = att[act[att] == A]; rest = att[act[att] != A]
        sp = part2(bid[rest], rng); tr_a, va_a = rest[sp == 0], rest[sp == 1]
        assert len(tr_a) > 0 and len(va_a) > 0 and len(te_a) > 0
        ben = rng.permutation(np.where(y == 0)[0]); c1, c2 = int(.6 * len(ben)), int(.8 * len(ben))
        tr = np.concatenate([tr_a, ben[:c1]]); va = np.concatenate([va_a, ben[c1:c2]]); te = np.concatenate([te_a, ben[c2:]])
        ytr, yva, yte = y[tr], y[va], y[te]
        sc = RobustScaler().fit(X[tr][ytr == 0]); Z = np.clip(sc.transform(X), -10, 10).astype(np.float32)
        ae = fit_ae(Z[tr][ytr == 0], seed)
        E = ((Z - ae.predict(Z, batch_size=8192, verbose=0)) ** 2).mean(1)
        Eu = E / (E + np.median(E[tr][ytr == 0]))          # monotone map of the UNCLIPPED error
        F = np.column_stack([X, Eu]); nf = F.shape[1]
        sets = {"full": list(range(nf)), "noscore": list(range(nf - 1)), "noports": [i for i in range(nf) if i not in ports]}
        res = {"ae_only_unclipped": evaluate(Eu[va], yva, Eu[te], yte)}
        for tag, name, key in MODELS:
            cols = sets[key]; m = factory(name, seed); m.fit(F[tr][:, cols], ytr)
            res[tag] = evaluate(m.predict_proba(F[va][:, cols])[:, 1], yva, m.predict_proba(F[te][:, cols])[:, 1], yte)
        info = dict(activity=A, seed=seed, attacks=[len(tr_a), len(va_a), len(te_a)], benign=[c1, c2 - c1, len(ben) - c2])
        json.dump({"info": info, "res": res}, open(fo, "w"))
        print(f"[{A} seed {seed}] {time.time() - t0:.0f}s attacks {info['attacks']} | MCC@0.5%: full {res['lgbm_full']['fpr0.005']['mcc']:.3f} noscore {res['lgbm_noscore']['fpr0.005']['mcc']:.3f} noports {res['lgbm_noports']['fpr0.005']['mcc']:.3f} ada {res['ada']['fpr0.005']['mcc']:.3f}", flush=True)
