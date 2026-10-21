src = open("dapt_experiment.py").read()

def sub(old, new):
    global src
    assert src.count(old) == 1, "pattern not found exactly once: " + old[:70]
    src = src.replace(old, new)

sub('OUT = f"/mnt/d/dapt_work/results_gap{GAP:g}"; os.makedirs(OUT, exist_ok=True)',
    'MODE = sys.argv[3] if len(sys.argv) > 3 else "random"   # random = redrawn burst split; chrono = time-ordered split\n'
    'OUT = f"/mnt/d/dapt_work/results_v4_{MODE}_gap{GAP:g}"; os.makedirs(OUT, exist_ok=True)')

CHRONO = """def chrono_split():
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

"""
sub("def mccf(tp, fp, fn, tn):", CHRONO + "def mccf(tp, fp, fn, tn):")

sub("te = np.concatenate([att[sp == 2], ben[c2:]])",
    "te = np.concatenate([att[sp == 2], ben[c2:]])\n"
    '    if MODE == "chrono": sp, tr, va, te, c1, c2, ben = chrono_split()')

sub('res["ae_only"] = evaluate(S[va] / clip, yva, S[te] / clip, yte)',
    'res["ae_only"] = evaluate(S[va] / clip, yva, S[te] / clip, yte)\n'
    '    Eu = E / (E + np.median(E[tr][ytr == 0]))     # monotone map of the UNCLIPPED error into [0,1); use the FPR rows only\n'
    '    res["ae_only_unclipped"] = evaluate(Eu[va], yva, Eu[te], yte)')

sub('print("after ocsvm", int(time.time()-t0), flush=True)',
    'print("after ocsvm", int(time.time()-t0), flush=True)\n'
    '    Fu = np.column_stack([X, Eu]); mu = factory("lgbm_deployed", seed); mu.fit(Fu[tr], ytr)\n'
    '    res["lgbm_full_unclip"] = evaluate(mu.predict_proba(Fu[va])[:, 1], yva, mu.predict_proba(Fu[te])[:, 1], yte)')

open("dapt_experiment_v4.py", "w").write(src)
print("wrote dapt_experiment_v4.py")
