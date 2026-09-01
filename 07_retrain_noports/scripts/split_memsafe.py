"""
Memory-safe stratified 60/20/20 split from the temp feature files.
Reads _tmp_X.npy memory-mapped (never loads full 2.3GB into RAM),
splits by stratified indices (seed 42, identical methodology), and
writes X_train/val/test + y_train/val/test. Use ONLY if script 02's
in-memory split OOMs — the temp files are complete by then.
"""
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

MODEL_DIR = Path("/home/ramanaruban.online/soc.ramanaruban.online/Soc/07_retrain_noports/models")
RANDOM_SEED = 42

# y is small — load normally. Need shape of X to know row count.
y = np.load(MODEL_DIR / "_tmp_y.npy")
n = len(y)
print(f"total rows: {n:,} | attacks: {int(y.sum()):,}")

# X memory-mapped — stays on disk
X = np.load(MODEL_DIR / "_tmp_X.npy", mmap_mode="r")
assert X.shape[0] == n, f"row mismatch X{X.shape} y{y.shape}"
assert X.shape[1] == 62, f"expected 62 cols, got {X.shape[1]}"
print(f"X shape (mmap): {X.shape}")

idx = np.arange(n)
# same split as thesis: 20% test, then 25% of remainder = val -> 60/20/20
idx_tv, idx_test = train_test_split(idx, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
idx_train, idx_val = train_test_split(idx_tv, test_size=0.25, random_state=RANDOM_SEED, stratify=y[idx_tv])

for name, ix in [("train", idx_train), ("val", idx_val), ("test", idx_test)]:
    ix_sorted = np.sort(ix)                    # sorted for efficient mmap gather
    Xs = np.asarray(X[ix_sorted])              # materialize only this split
    ys = y[ix_sorted]
    np.save(MODEL_DIR / f"X_{name}.npy", Xs)
    np.save(MODEL_DIR / f"y_{name}.npy", ys)
    print(f"  {name}: X{Xs.shape} attacks={int(ys.sum()):,}")
    del Xs, ys

print("MEMORY-SAFE SPLIT DONE")
