import json, numpy as np
z16 = np.load("data/pilot_latents/latents.npy")
z32 = z16.astype(np.float32)
idx = np.array(json.load(open("data/pilot_latents/index.json", encoding="utf-8"))["std"], float)
print("dtype", z16.dtype, "shape", z16.shape)
for i in (1, 4, 8, 9, 13, 20, 28):
    print(f"dim{i:>3}  f16={float(z16[:, i].std()):.6f}  f32={float(z32[:, i].std()):.6f}  index={float(idx[i]):.6f}")
d16 = np.diff(z16, axis=0)
d64 = np.diff(z32.astype(np.float64), axis=0)
print("mae  f16=%.6f  f64=%.6f" % (np.abs(d16).mean(), np.abs(d64).mean()))
print("mse  f16=%.8f  f64=%.8f" % ((d16**2).mean(), (d64**2).mean()))
print("norm-mse f16=%.4f" % ((d16/np.maximum(z16.std(0),1e-8))**2).mean())
