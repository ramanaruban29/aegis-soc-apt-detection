import json, glob, sys, numpy as np
from scipy import stats
d = sys.argv[1]
R = [json.load(open(f)) for f in sorted(glob.glob(d + '/*.json'))]
J = len(R)
models = [m for m in R[0]['res'] if all(m in r['res'] for r in R) and m != 'ae_only']
F = ['fpr2e-05', 'fpr4e-05', 'fpr8e-05', 'mccrule']
col = lambda m, f, k='mcc': np.array([r['res'][m][f][k] for r in R])
ntr = np.mean([r['info']['attacks'][0] for r in R]); nte = np.mean([r['info']['attacks'][2] for r in R])
print(f"{d}: {J} runs | mean attacks train/test {ntr:.0f}/{nte:.0f}")
print("Median MCC [median FP]")
print(f"{'model':16s}" + "".join(f"{f:>22s}" for f in F))
for m in models:
    print(f"{m:16s}" + "".join(f"{np.median(col(m,f)):>12.3f} [{np.median(col(m,f,'fp')):>6.0f}]  " for f in F))
others = [m for m in models if m != 'lgbm_full']
rng = np.random.default_rng(0)
for f in F[:3]:
    print(f"\nDeployed minus other at {f} ({len(others)} comparisons)")
    for m in others:
        diff = col('lgbm_full', f) - col(m, f)
        bs = [rng.choice(diff, J).mean() for _ in range(5000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        t = diff.mean() / np.sqrt((1 / J + nte / ntr) * diff.var(ddof=1) + 1e-12)
        p = min(1, 2 * stats.t.sf(abs(t), J - 1) * len(others))
        print(f"  {m:14s} {diff.mean():+.3f} runs-CI [{lo:+.3f}, {hi:+.3f}] corrected p {p:.3f} | deployed better {int((diff>0).sum())}/{J}")
