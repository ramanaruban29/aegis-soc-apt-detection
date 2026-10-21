import json, glob, os, numpy as np, pandas as pd
from scipy import stats

CIC = [("row split (leaky)", "experiments_v4/cicapt_row_block"),
       ("random benign, 5 s bursts", "experiments_v4/cicapt_random5_random"),
       ("time-blocked benign, 5 s", "experiments_v4/cicapt_random5_block"),
       ("time-blocked benign, 120 s", "experiments_v4/cicapt_random120_block"),
       ("chronological (unseen stages)", "experiments_v4/cicapt_chrono_block")]
DAPT = [("DAPT random, 120 s gap", "/mnt/d/dapt_work/results_v4_random_gap120"),
        ("DAPT random, 5 s gap", "/mnt/d/dapt_work/results_v4_random_gap5"),
        ("DAPT chronological, 120 s", "/mnt/d/dapt_work/results_v4_chrono_gap120")]
LOSO = "experiments_v4/cicapt_loso_block"
LOAO = "/mnt/d/dapt_work/results_v4_loao_gap120"
DEPLOYED = "lgbm_full"
rng = np.random.default_rng(0)

def load(d):
    return [json.load(open(f)) for f in sorted(glob.glob(d + "/*.json"))]

def thr_keys(R, m):
    return [k for k, x in R[0]["res"][m].items() if isinstance(x, dict) and "mcc" in x]

def summary(R, setting):
    out = []
    for m in [m for m in R[0]["res"] if all(m in r["res"] for r in R)]:
        if not isinstance(R[0]["res"][m], dict):
            continue
        for k in thr_keys(R, m):
            mcc = np.array([r["res"][m][k]["mcc"] for r in R])
            rec = np.array([r["res"][m][k]["recall"] for r in R])
            fp = np.array([r["res"][m][k]["fp"] for r in R])
            out.append(dict(setting=setting, model=m, rule=k, runs=len(R),
                            mcc_median=np.median(mcc), mcc_mean=mcc.mean(), mcc_sd=mcc.std(ddof=1) if len(R) > 1 else np.nan,
                            recall_median=np.median(rec), fp_median=np.median(fp), fp_mean=fp.mean()))
    return out

def paired(R, setting):
    out = []
    J = len(R)
    try:
        ntr = np.mean([r["info"]["attacks"][0] for r in R]); nte = np.mean([r["info"]["attacks"][2] for r in R])
        ratio = nte / ntr
    except Exception:
        ratio = 1 / 3
    if DEPLOYED not in R[0]["res"]:
        return out
    others = [m for m in R[0]["res"] if m != DEPLOYED and isinstance(R[0]["res"][m], dict)
              and all(m in r["res"] for r in R) and thr_keys(R, m)]
    for k in thr_keys(R, DEPLOYED):
        for m in others:
            if k not in R[0]["res"][m]:
                continue
            d = np.array([r["res"][DEPLOYED][k]["mcc"] - r["res"][m][k]["mcc"] for r in R])
            bs = [rng.choice(d, J).mean() for _ in range(5000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            t = d.mean() / np.sqrt((1 / J + ratio) * d.var(ddof=1) + 1e-12) if J > 1 else np.nan
            p = min(1.0, 2 * stats.t.sf(abs(t), J - 1) * len(others)) if J > 1 else np.nan
            out.append(dict(setting=setting, rule=k, vs=m, runs=J, mean_diff=d.mean(), ci_low=lo, ci_high=hi,
                            corrected_p_bonferroni=p, deployed_better=int((d > 0).sum())))
    return out

sheets = {}
readme = ["Workbook of repeated-run results (AEGIS evaluation)",
          "Medians are over runs (seeds). MCC=Matthews correlation coefficient. FP=false positives on the test set.",
          "Rules: fprXe-05 = FPR-rule thresholds (0.002/0.004/0.008% on CICAPT-IIoT); fpr0.002/0.005/0.01 on DAPT2020 = 0.2/0.5/1%; mccrule = MCC-optimal threshold on validation.",
          "paired sheets: deployed (lgbm_full) minus other model; CI = bootstrap over runs; p = Nadeau-Bengio corrected t-test with Bonferroni multiplier.",
          "Run counts differ between settings; compare medians, not significance, across settings. Chronological runs have only 3 seeds: descriptive only.",
          "Gradient Boosting probabilities saturate near 1.0 (thresholds at/near 1.0): its low scores are a scoring artefact, not a model failure."]
sheets["README"] = pd.DataFrame({"notes": readme})

for tag, groups in (("CICAPT", CIC), ("DAPT", DAPT)):
    S, P = [], []
    for name, d in groups:
        if not os.path.isdir(d) or not glob.glob(d + "/*.json"):
            print("skip (no files):", d); continue
        R = load(d)
        try:
            S += summary(R, name); P += paired(R, name)
        except Exception as e:
            print("skip", d, repr(e))
    if S:
        df = pd.DataFrame(S); sheets[tag + "_summary"] = df
        lad = df[df.model == DEPLOYED].pivot_table(index="setting", columns="rule", values=["mcc_median", "fp_median"], sort=False)
        lad.columns = [f"{a}|{b}" for a, b in lad.columns]; sheets[tag + "_ladder"] = lad.reset_index()
    if P:
        sheets[tag + "_paired"] = pd.DataFrame(P)

# CICAPT leave-one-stage-out
try:
    G = {}
    for f in sorted(glob.glob(LOSO + "/*.json")):
        d = json.load(open(f)); G.setdefault(d["info"]["tag"].replace("loso_", ""), []).append(d)
    rows = []
    for s, L in sorted(G.items()):
        for m in [m for m in L[0]["res"] if all(m in x["res"] for x in L)]:
            for k in thr_keys(L, m):
                rows.append(dict(held_out_stage=s, model=m, rule=k, seeds=len(L),
                                 test_attacks=int(np.mean([x["info"]["attacks"][2] for x in L])),
                                 recall_median=np.median([x["res"][m][k]["recall"] for x in L]),
                                 mcc_median=np.median([x["res"][m][k]["mcc"] for x in L]),
                                 fp_median=np.median([x["res"][m][k]["fp"] for x in L])))
    sheets["CICAPT_loso"] = pd.DataFrame(rows)
except Exception as e:
    print("skip loso", repr(e))

# DAPT2020 leave-one-activity-out
try:
    G = {}
    for f in sorted(glob.glob(LOAO + "/*.json")):
        d = json.load(open(f)); G.setdefault(d["info"]["activity"], []).append(d)
    rows = []
    for a, L in sorted(G.items()):
        for m in [m for m in L[0]["res"] if all(m in x["res"] for x in L)]:
            for k in thr_keys(L, m):
                rows.append(dict(held_out_activity=a, model=m, rule=k, seeds=len(L),
                                 recall_median=np.median([x["res"][m][k]["recall"] for x in L]),
                                 mcc_median=np.median([x["res"][m][k]["mcc"] for x in L]),
                                 fp_median=np.median([x["res"][m][k]["fp"] for x in L])))
    sheets["DAPT_loao"] = pd.DataFrame(rows)
except Exception as e:
    print("skip loao", repr(e))

out = "results_workbook.xlsx"
with pd.ExcelWriter(out, engine="openpyxl") as w:
    for name, df in sheets.items():
        df.round(4).to_excel(w, sheet_name=name[:31], index=False)
        w.sheets[name[:31]].freeze_panes = "A2"
print("wrote", os.path.abspath(out))
for n, df in sheets.items():
    print(f"  {n:18s} {len(df):5d} rows")
