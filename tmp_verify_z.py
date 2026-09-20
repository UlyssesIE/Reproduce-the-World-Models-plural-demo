import json, numpy as np, torch, sys
from pathlib import Path
sys.path.insert(0, ".")
from src.data.preprocess import preprocess_frame, to_float
from src.models.vae import VAE, VAEConfig

info = json.loads(Path("data/pilot_latents/index.json").read_text())
names = [f[0] for f in info["files"]]
print("index says:", len(names), "files, first =", names[0])

print("\n--- what is actually on disk ---")
for pat in ("data/**/*.npz", "data/*.npy", "data/**/*.npy", "**/rollout_*.npz"):
    hits = sorted(Path(".").glob(pat))
    print(f"{pat:22} -> {len(hits)} hits", [str(h) for h in hits[:3]])
print("data/ listing:", [str(p) for p in sorted(Path("data").iterdir())])

# find the first index-listed rollout anywhere under the repo
target = names[0]
cand = [p for p in Path(".").rglob(target) if ".git" not in str(p)]
print("\nlocated target file:", cand[:3])
if not cand:
    raise SystemExit("!! target rollout not found on disk -- tell me where data/pilot*.npz live")

src = cand[0]
ck = torch.load("runs/vae_pilot2/best.pt", map_location="cpu", weights_only=False)
vae = VAE(VAEConfig(**ck["cfg"])).eval(); vae.load_state_dict(ck["model"])
by = {n: (s, c) for n, s, c in info["files"]}
s0, cnt = by[target]
lat = np.load("data/pilot_latents/latents.npy", mmap_mode="r")
with np.load(src) as d:
    print("npz keys:", list(d.keys()), "obs", d["obs"].shape)
    obs = d["obs"]
for k in (0, 1, 137, cnt - 1):
    x = to_float(preprocess_frame(obs[k], 12, 64, False))
    t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).unsqueeze(0)
    with torch.no_grad():
        mu, _ = vae.encode(t)
    print(k, "max|dz| =", float(np.abs(mu[0].numpy() - lat[s0 + k].astype(np.float32)).max()))
