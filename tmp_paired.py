import json, numpy as np
A = "runs/controller_vonly/eval.json"   # 99 params
B = "runs/controller_867/eval.json"     # 867 params
a, b = (json.load(open(p, encoding="utf-8")) for p in (A, B))
sa, sb = a["seed_returns"], b["seed_returns"]
common = sorted(set(sa) & set(sb), key=int)
ra = np.array([sa[k] for k in common]); rb = np.array([sb[k] for k in common])
d = rb - ra; n = d.size
m, sd = d.mean(), d.std(ddof=1); se = sd / np.sqrt(n)
print(f"common seeds: {n}")
print(f"A {a['params']:>4}p  mean {ra.mean():7.2f}  sd {ra.std(ddof=1):6.2f}")
print(f"B {b['params']:>4}p  mean {rb.mean():7.2f}  sd {rb.std(ddof=1):6.2f}")
print(f"paired diff (B-A): {m:+.2f}   sd {sd:.1f}  se {se:.2f}   t = {m/se:.2f}  (df {n-1})")
rng = np.random.default_rng(0)
bs = np.array([rng.choice(d, n, replace=True).mean() for _ in range(20000)])
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f"bootstrap 95% CI: [{lo:+.2f}, {hi:+.2f}]")
print(f"B > A on {int((d>0).sum())}/{n} rollouts ({100*(d>0).mean():.0f}%)")
