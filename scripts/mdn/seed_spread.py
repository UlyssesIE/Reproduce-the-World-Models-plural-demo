"""Seed-level spread of D/B (the axis Henderson et al. 2017 / Colas et al. 2018 audit).

Fixes over the first version:
  * D/B parse corrected: D/B = g8 / g5   (was g7/g4 = A/B / A_sd -- wrong)
  * family() ordered so mdn_pilot_noact is NOT classed as 'real'
  * self-checks use a 5e-7 tolerance (references are recorded to 6 dp)

Families with trained-seed replicas on disk:
    real  : runs/mdn_pilot{,_s1,_s2,_s3,_s4}/best.pt          (5)
    shuf  : runs/mdn_pilot_shuf{,_s1,_s2,_s3}/best.pt        (4)
    noact : runs/mdn_pilot_noact{,_s1,_s2}/best.pt           (3)

Protocol: dream_controllability.py --rollout r --steps 200 --warmup 16 --reps 50 --taus 1.0
Probe dir: ckpt.args.latents, EXCEPT the noact family -> data/pilot_latents
           (their own dir is ALL-ZERO actions; declared override, as in the table).
"""
import csv, glob, os, re, statistics as st, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import torch

NR, WORKERS, TOL = 20, 6, 5e-7
FAM = [("real", r"runs\mdn_pilot\best.pt"),        ("real", r"runs\mdn_pilot_s1\best.pt"),
       ("real", r"runs\mdn_pilot_s2\best.pt"),     ("real", r"runs\mdn_pilot_s3\best.pt"),
       ("real", r"runs\mdn_pilot_s4\best.pt"),
       ("shuf", r"runs\mdn_pilot_shuf\best.pt"),   ("shuf", r"runs\mdn_pilot_shuf_s1\best.pt"),
       ("shuf", r"runs\mdn_pilot_shuf_s2\best.pt"),("shuf", r"runs\mdn_pilot_shuf_s3\best.pt"),
       ("noact", r"runs\mdn_pilot_noact\best.pt"), ("noact", r"runs\mdn_pilot_noact_s1\best.pt"),
       ("noact", r"runs\mdn_pilot_noact_s2\best.pt")]
OVR = {"noact": r"data\pilot_latents"}
PROTO = ["--steps","200","--warmup","16","--reps","50","--taus","1.0"]
OUT = Path("tmp_seed"); OUT.mkdir(exist_ok=True)
ENV = dict(os.environ); ENV.update({"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","PYTHONIOENCODING":"utf-8"})
RE = re.compile(r"^\s*([\d.]+)\s+(\d+)\s+([\d.]+)\s*\u00b1\s*([\d.]+)\s+([\d.]+)\s*\u00b1\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)")

present = [(f, c) for f, c in FAM if os.path.exists(c)]
missing = [(f, c) for f, c in FAM if not os.path.exists(c)]
print("ckpts present = %d ; missing = %s" % (len(present), [m[1] for m in missing] or "none"))
lat = {}
for f, c in present:
    own = torch.load(c, map_location="cpu").get("args",{}).get("latents")
    lat[c] = OVR.get(f, own)
    print("   %-6s %-46s -> %s%s" % (f, c, lat[c], "   <- OVERRIDE" if f in OVR else ""))

def run(fam, ck, ro):
    tag = Path(ck).parent.name
    f = OUT / ("%s_r%02d.txt" % (tag, ro))
    if not (f.exists() and f.stat().st_size > 0):
        p = subprocess.run([sys.executable,"scripts/mdn/dream_controllability.py","--mdn",ck,
                            "--latents",lat[ck],"--rollout",str(ro)]+PROTO,
                           capture_output=True,text=True,encoding="utf-8",errors="replace",env=ENV)
        f.write_text((p.stdout or "")+(p.stderr or ""),encoding="utf-8")
    t = f.read_text(encoding="utf-8",errors="replace")
    for ln in t.splitlines():
        m = RE.match(ln)
        if m and abs(float(m.group(1))-1.0) < 1e-9 and int(m.group(2)) == 1:
            B = float(m.group(5))
            return fam, ck, ro, (float(m.group(8))/B if B else None)
    return fam, ck, ro, None

tasks = [(f, c, r) for f, c in present for r in range(NR)]
print("\nrunning %d invocations (%d ckpts x %d rollouts) ...\n" % (len(tasks), len(present), NR))
got = {}; done = 0; t0 = time.time()
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    for fu in as_completed([ex.submit(run, *t) for t in tasks]):
        fam, ck, ro, v = fu.result(); got.setdefault(ck, {})[ro] = v; done += 1
        if done % 40 == 0 or done == len(tasks):
            el = time.time()-t0
            print("  %3d/%d  %5.1f min  ETA %5.1f min" % (done, len(tasks), el/60, (el/done)*(len(tasks)-done)/60), flush=True)

print("\n=== self-checks (tolerance %.0e, references stored to 6 dp) ===")
m0 = st.mean([got["runs\\mdn_pilot\\best.pt"][r] for r in range(NR)])
print("  A) seed-0 mean = %.7f  vs v2 real = 0.089631  -> %s" % (m0, "EXACT" if abs(m0-0.089631)<TOL else "DIFF"))
prov = {(r["row"], int(r["ro"])): float(r["DoverB"]) for r in csv.DictReader(open("tmp_final_provenance_v2.csv", encoding="utf-8-sig"))}
ok = sum(1 for r in range(NR) if ("real", r) in prov and abs(got["runs\\mdn_pilot\\best.pt"][r]-prov[("real", r)]) < TOL)
print("  B) seed-0 vs provenance_v2: %d/%d within %.0e" % (ok, NR, TOL))

print("\n=== per-checkpoint D/B (t=1, tau=1) ===")
per = {}
for f, c in present:
    v = [got[c][r] for r in range(NR) if got.get(c, {}).get(r) is not None]
    tm = re.search(r"mdn_pilot_shuf_s\d", Path(c).parent.name)
    per[c] = st.mean(v)
    print("  %-6s %-24s mean=%.6f  sd=%.6f  se=%.6f  n=%d" % (f, Path(c).parent.name, st.mean(v), st.stdev(v), st.stdev(v)/len(v)**0.5, len(v)))

print("\n=== SEED AXIS by family ===")
summary = {}
for fam in ("real", "shuf", "noact"):
    cks = [c for (ff, c) in present if ff == fam]
    ms = [per[c] for c in cks if c in per]
    if len(ms) < 2:
        print("  %-6s replicas=%d -> not measurable" % (fam, len(ms))); continue
    sm, ss = st.mean(ms), st.stdev(ms)
    summary[fam] = (len(ms), sm, ss)
    print("  %-6s n_seeds=%d  seed-mean=%.6f  seed-sd=%.6f  SE=%.6f  cv=%.1f%%" % (
        fam, len(ms), sm, ss, ss/len(ms)**0.5, 100*ss/sm))
    print("        per-seed: " + "  ".join("%.6f" % m for m in ms))

print("\n=== family separation on the seed axis ===")
if "real" in summary and "shuf" in summary:
    (n1,m1,s1),(n2,m2,s2) = summary["real"], summary["shuf"]
    sd = (s1**2/n1 + s2**2/n2)**0.5
    print("  real %.6f +- %.6f  vs  shuf %.6f +- %.6f   diff=%.6f +- %.6f => %.1f sigma" % (
        m1, s1/n1**0.5, m2, s2/n2**0.5, m1-m2, sd, (m1-m2)/sd))
if "real" in summary and "noact" in summary:
    (n1,m1,s1),(n2,m2,s2) = summary["real"], summary["noact"]
    sd = (s1**2/n1 + s2**2/n2)**0.5
    print("  real %.6f +- %.6f  vs  noact %.6f +- %.6f  diff=%.6f +- %.6f => %.1f sigma  (tests: does killing the channel RAISE D/B?)" % (
        m1, s1/n1**0.5, m2, s2/n2**0.5, m1-m2, sd, (m1-m2)/sd))

with open("tmp_seed_summary.csv","w",encoding="utf-8",newline="") as fh:
    w = csv.writer(fh); w.writerow(["family","ckpt","rollout_mean_DB","rollout_sd","rollout_se"])
    for f, c in present:
        v = [got[c][r] for r in range(NR) if got.get(c, {}).get(r) is not None]
        w.writerow([f, Path(c).parent.name, "%.6f"%st.mean(v), "%.6f"%st.stdev(v), "%.6f"%(st.stdev(v)/len(v)**0.5)])
print("\nwrote tmp_seed_summary.csv")
