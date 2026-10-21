import os, pathlib, joblib, numpy as np
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score

OUT = pathlib.Path("experiments_v4"); OUT.mkdir(exist_ok=True)
ae = tf.keras.models.load_model("ae_corrected.keras", compile=False)
sc = joblib.load("ae_scaler_corrected.pkl")
CH = 200_000

def rescore(split):
    X = np.load(f"X_{split}_aegis_corrected.npy", mmap_mode="r")
    raw = np.empty(len(X)); stored = np.asarray(X[:, 63], dtype=np.float64)
    for i in range(0, len(X), CH):
        xs = np.clip(sc.transform(np.asarray(X[i:i+CH, :63])), -10, 10).astype(np.float32)
        raw[i:i+CH] = ((xs - ae(xs, training=False).numpy()) ** 2).mean(1)
    return raw, stored

def mcc(tp, fp, fn, tn):
    d = np.sqrt(float(tp+fp) * (tp+fn) * (tn+fp) * (tn+fn))
    return (tp*tn - fp*fn) / d if d else 0.0

S, Y = {}, {}
for sp in ("val", "test"):
    S[sp], stored = rescore(sp)
    Y[sp] = np.load(f"y_{sp}_aegis_corrected.npy")
    clip = stored.max()
    diff = np.abs(np.minimum(S[sp], clip) - stored).max()
    print(f"[{sp}] clip={clip:.6f}  max|recomputed-stored|={diff:.2e}")
    if diff >= 1e-4:
        print("MISMATCH with stored column 63 -- stop and send me this output")
        raise SystemExit
    np.save(OUT / f"ae_score_unclipped_{sp}.npy", S[sp])

yv, yt = Y["val"], Y["test"]
print("Test AUC: %.4f  AP: %.4f" % (roc_auc_score(yt, S["test"]), average_precision_score(yt, S["test"])))
bv = S["val"][yv == 0]
print("Enrichment on val (attacks vs benign above a benign quantile):")
for q in (.99, .999, .9999):
    t = np.quantile(bv, q)
    print(f"  q={q}: attacks {(S['val'][yv==1] > t).mean():.4f} | benign {(S['val'][yv==0] > t).mean():.4f}")
print("AE alone, unclipped, FPR rule (threshold from val benign, applied to test):")
for f in (2e-5, 4e-5, 8e-5):
    t = np.quantile(bv, 1 - f); p = S["test"] > t
    tp = int((p & (yt == 1)).sum()); fp = int((p & (yt == 0)).sum())
    fn = int(((~p) & (yt == 1)).sum()); tn = int(((~p) & (yt == 0)).sum())
    print(f"  FPR {f*100:.3f}%: recall {tp/(tp+fn):.3f}  FP {fp}  MCC {mcc(tp, fp, fn, tn):.3f}")
