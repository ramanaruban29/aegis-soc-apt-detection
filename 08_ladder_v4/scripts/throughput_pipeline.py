import time, numpy as np, joblib
from tensorflow import keras
sc = joblib.load("ae_scaler_corrected.pkl")
ae = keras.models.load_model("ae_corrected.keras")
gbm = joblib.load("cicapt_lgbm_v2_aegis_corrected.pkl")
clip = float(np.load("X_train_aegis_corrected.npy", mmap_mode="r")[:, 63].max())
X = np.load("X_full.npy", mmap_mode="r"); N = X.shape[0]
def stage1(raw):
    xs = np.clip(sc.transform(raw), -10, 10).astype(np.float32)
    s = np.minimum(((xs - ae(xs, training=False).numpy()) ** 2).mean(1), clip)
    return np.concatenate([raw, s[:, None]], axis=1)
idx = np.load("idx_test_corrected.npy"); Xt = np.load("X_test_aegis_corrected.npy", mmap_mode="r")
pos = np.sort(np.random.default_rng(0).choice(len(idx), 50_000, replace=False))
f = stage1(np.asarray(X[idx[pos], 1:64], dtype=np.float32))
print("sanity: max |score difference| vs stored:", float(np.abs(f[:, 63] - np.asarray(Xt[pos, 63])).max()),
      "| max |probability difference|:", float(np.abs(gbm.predict_proba(f)[:, 1] - gbm.predict_proba(np.asarray(Xt[pos]))[:, 1]).max()))
t1 = t2 = 0.0
for s in range(0, N, 250_000):
    raw = np.asarray(X[s:s + 250_000, 1:64], dtype=np.float32)
    a = time.perf_counter(); f = stage1(raw); b = time.perf_counter(); gbm.predict_proba(f); c = time.perf_counter()
    t1 += b - a; t2 += c - b
print(f"full stream {N:,} flows: scaler+autoencoder {t1:.1f}s, LightGBM {t2:.1f}s, total {t1+t2:.1f}s = {N/(t1+t2):,.0f} flows/s")
rows = np.asarray(X[np.sort(np.random.default_rng(1).choice(N, 1000, replace=False)), 1:64], dtype=np.float32)
for i in range(20): gbm.predict_proba(stage1(rows[i:i+1]))
d_all, d_gbm = [], []
for i in range(1000):
    a = time.perf_counter(); f = stage1(rows[i:i+1]); b = time.perf_counter(); gbm.predict_proba(f); c = time.perf_counter()
    d_all.append((c - a) * 1e3); d_gbm.append((c - b) * 1e3)
for n, d in [("whole pipeline", d_all), ("LightGBM only", d_gbm)]:
    print(f"single-flow latency, {n}: mean {np.mean(d):.2f} ms, median {np.median(d):.2f} ms, P99 {np.percentile(d, 99):.2f} ms")
