import torch
for p in ["runs/vae_pilot2/best.pt", "runs/mdn_pilot/best.pt"]:
    d = torch.load(p, map_location="cpu", weights_only=False)
    print("=" * 72); print(p, "->", type(d).__name__)
    if isinstance(d, dict):
        for k, v in d.items():
            t = type(v).__name__
            if hasattr(v, "shape"):     t += f" shape={tuple(v.shape)}"
            elif isinstance(v, dict):   t += f" n={len(v)} keys={list(v.keys())[:6]}"
            elif hasattr(v, "__dict__"):t += f" attrs={[a for a in vars(v)][:12]}"
            print(f"   {k!r:22} {t}")
