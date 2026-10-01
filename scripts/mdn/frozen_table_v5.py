"""Frozen table v5 -- ONE cycle, ALL 15 rows x 20 rollouts x 3 taus.

OPERATOR DECISIONS
  * NR = 20 episodes (dataset holds 100; 20 is the reporting level)
  * x = 'new-x' := NLL(lam=0)-NLL(lam=1) of THAT ROW's seed-0 checkpoint,
    metric_validation.py --split val --seed 0.   The five real training seeds
    appear only as the error bar on x(real).  'brief-x' is superseded:
    external, not reproducible, never quoted.

DECLARED PROTOCOL (uniform, except two declared overrides)
  scripts/mdn/dream_controllability.py --mdn <ckpt> --latents <PROBE_DIR(row)>
      --rollout r --steps 200 --warmup 16 --reps 50 --taus 0.1,0.5,1.0      r=0..19
  D/B = D(det)/B(noise)        A/B = printed A/B        (per (tau,t))
  PROBE_DIR = ckpt.args.latents, EXCEPT:
      real-struct, synth3-struct -> family base dir (own dir actions are ALL ZERO,
      which forces A=D=0; declared here with reason).

OUTPUTS  tmp_v5_protocol.csv (declaration), tmp_v5_cells.csv (row x rollout x tau x t),
         tmp_v5_table.csv (surfaces), tmp_v5_headline.csv, tmp_v5_log.txt, raw in tmp_v5/
RESUMABLE: raw files are keyed (ckpt, rollout); rerunning reuses them ('cached').
"""
import csv, os, re, statistics as st, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import torch

NR, TS, TAUS = 20, [1,2,4,8,16,32,64,128,200], "0.1,0.5,1.0"
PROTO = ["--steps","200","--warmup","16","--reps","50","--taus",TAUS]
ROWS = [
 ("shuf",          r"runs\mdn_pilot_shuf\best.pt",            True,  None, ""),
 ("p25",           r"runs\mdn_pilot_p25\best.pt",             True,  None, ""),
 ("p50",           r"runs\mdn_pilot_p50\best.pt",             True,  None, ""),
 ("p75",           r"runs\mdn_pilot_p75\best.pt",             True,  None, ""),
 ("real",          r"runs\mdn_pilot\best.pt",                 True,  None, ""),
 ("real-struct",   r"runs\mdn_pilot_noact\best.pt",           False, r"data\pilot_latents",
   "own dir actions ALL ZERO -> A=D=0 degenerate; family base used so the channel is exercised"),
 ("s3p10",         r"runs\mdn_pilot_synth3p10\best.pt",       True,  None, ""),
 ("s3p15",         r"runs\mdn_pilot_synth3p15\best.pt",       True,  None, ""),
 ("s3p20",         r"runs\mdn_pilot_synth3p20\best.pt",       True,  None, ""),
 ("s3p25",         r"runs\mdn_pilot_synth3p25\best.pt",       True,  None, ""),
 ("s3p50",         r"runs\mdn_pilot_synth3p50\best.pt",       True,  None, ""),
 ("s3p75",         r"runs\mdn_pilot_synth3p75\best.pt",       True,  None, ""),
 ("s3p90",         r"runs\mdn_pilot_synth3p90\best.pt",       True,  None, ""),
 ("PC2",           r"runs\mdn_pilot_pc2\best.pt",             False, None, ""),
 ("synth3-struct", r"runs\mdn_pilot_synth3noact\best.pt",     False, r"data\pilot_latents_synth3",
   "own dir actions ALL ZERO -> A=D=0 degenerate; family base used so the channel is exercised"),
]
V2_T1 = {"shuf":0.023117,"p25":0.015406,"p50":0.024856,"p75":0.081848,"real":0.089631,
         "real-struct":0.156493,"s3p10":0.142133,"s3p15":0.145003,"s3p20":0.184936,
         "s3p25":0.270098,"s3p50":0.464836,"s3p75":0.667618,"s3p90":1.059601,
         "PC2":1.281279,"synth3-struct":0.731066}
WORKERS = max(2, min(6, (os.cpu_count() or 4) - 2))
OUT  = Path("tmp_v5")
ENV  = dict(os.environ); ENV.update({"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","PYTHONIOENCODING":"utf-8"})
RE_CELL  = re.compile(r"^\s*([\d.]+)\s+(\d+)\s+([\d.]+)\s*\u00b1\s*([\d.]+)\s+([\d.]+)\s*\u00b1\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)")
RE_X     = re.compile(r"lam=([\d.]+).*?NLL\s+(-?[\d.]+)")
RE_MOVED = re.compile(r"permutation moved ([\d.]+)%")

OUT.mkdir(exist_ok=True)
spec = OUT/".spec"; want = "v5 NR=%d taus=%s reps=50 steps=200 warmup=16" % (NR,TAUS)
if spec.exists():
    have = spec.read_text(encoding="utf-8").strip()
    if have != want:
        print("SPEC MISMATCH\n  cached: %s\n  wanted: %s\n  fix: Remove-Item -Recurse tmp_v5" % (have,want)); sys.exit(1)
else:
    spec.write_text(want, encoding="utf-8")

lat = {t: torch.load(c, map_location="cpu").get("args",{}).get("latents") for t,c,hx,o,w in ROWS}
plan = [dict(row=t, ckpt=c, args_latents=lat[t], probe_dir=(o if o else lat[t]),
             override=bool(o), reason=w, has_x=hx) for t,c,hx,o,w in ROWS]
with open("tmp_v5_protocol.csv","w",encoding="utf-8",newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["row","ckpt","args_latents","probe_dir","override","reason","NR","x_definition"])
    for p in plan:
        wr.writerow([p["row"],p["ckpt"],p["args_latents"],p["probe_dir"],int(p["override"]),p["reason"],
                     NR,"NLL(lam=0)-NLL(lam=1), seed-0 ckpt, --split val --seed 0"])
print("=== declared protocol (NR=%d) ===" % NR)
for p in plan:
    print("  %-14s probe=%-30s %s" % (p["row"], p["probe_dir"], "<- OVERRIDE (declared)" if p["override"] else ""))

scan = [(p["row"],p["ckpt"],p["probe_dir"],r) for p in plan for r in range(NR)]
xs   = [(p["row"],p["ckpt"],p["probe_dir"]) for p in plan if p["has_x"]]
est  = len(scan)*129.0/WORKERS
print("\n[preflight] workers=%d grid=%d x=%d ETA ~ %.0f min (%.1f h)" % (WORKERS,len(scan),len(xs),est/60,est/3600))

LOG = open("tmp_v5_log.txt","w",encoding="utf-8")
def call(cmd,path):
    LOG.write(":: "+" ".join(cmd)+"\n"); LOG.flush()
    if path.exists() and path.stat().st_size>0: return path.read_text(encoding="utf-8",errors="replace"),"cached"
    p = subprocess.run(cmd,capture_output=True,text=True,encoding="utf-8",errors="replace",env=ENV)
    t = (p.stdout or "")+(p.stderr or ""); path.write_text(t,encoding="utf-8")
    return t,("ok" if p.returncode==0 else "exit%d"%p.returncode)

def task_db(tag,ck,probe,ro):
    f = OUT/("%s_r%03d.txt"%(Path(ck).parent.name,ro))
    cmd = [sys.executable,"scripts/mdn/dream_controllability.py","--mdn",ck,"--latents",probe,"--rollout",str(ro)]+PROTO
    txt,status = call(cmd,f)
    mv = RE_MOVED.search(txt); moved = float(mv.group(1)) if mv else -1.0
    cells = []
    for ln in txt.splitlines():
        m = RE_CELL.match(ln)
        if m:
            tau = float(m.group(1))
            if abs(tau-0.1)<1e-9 or abs(tau-0.5)<1e-9 or abs(tau-1.0)<1e-9:
                A=float(m.group(3)); B=float(m.group(5)); AB=float(m.group(7)); D=float(m.group(8))
                cells.append((tau,int(m.group(2)),A,B,AB,D,(D/B if B else None)))
    return (tag,ro,cells,status,moved)

def task_x(tag,ck,probe):
    f = OUT/("x_%s.txt"%Path(ck).parent.name)
    cmd = [sys.executable,"scripts/mdn/metric_validation.py","--mdn",ck,"--latents",probe,"--split","val","--seed","0"]
    txt,status = call(cmd,f); d = {}
    for ln in txt.splitlines():
        m = RE_X.search(ln)
        if m: d[float(m.group(1))] = float(m.group(2))
    return tag, ((d[0.0]-d[1.0]) if (0.0 in d and 1.0 in d) else None), status

print("\n=== phase 1: grid, %d invocations x 3 taus ===" % len(scan))
raw = {}; done = 0; t0 = time.time()
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    for fu in as_completed([ex.submit(task_db,*t) for t in scan]):
        tag,ro,cells,status,moved = fu.result(); raw[(tag,ro)] = (cells,status,moved); done += 1
        if done % 20 == 0 or done == len(scan):
            el = time.time()-t0
            print("  %3d/%d  %5.1f min  ETA %5.1f min" % (done,len(scan),el/60,(el/done)*(len(scan)-done)/60), flush=True)
emp = [k for k,(c,s,m) in raw.items() if not c]
bad = [(k,len(c),s) for k,(c,s,m) in raw.items() if c and len(c) != 3*len(TS)]
zm  = [k for k,(c,s,m) in raw.items() if m == 0.0]
print("  empty=%d  wrong-cell-count=%d  %s" % (len(emp),len(bad),bad[:5]))
print("  zero-action invocations = %d -> %s" % (len(zm), "OK (none)" if not zm else zm[:6]))

print("\n=== phase 2: x (new-x), %d invocations ===" % len(xs))
xnew = {}
with ThreadPoolExecutor(max_workers=4) as ex:
    for fu in as_completed([ex.submit(task_x,*t) for t in xs]):
        tag,x,status = fu.result(); xnew[tag] = x
        print("  %-8s x=%-9s %s" % (tag,("%.3f"%x) if x is not None else "-",status), flush=True)

def allv(row,tau,t,idx):
    v = []
    for ro in range(NR):
        for c in raw.get((row,ro),([],))[0]:
            if abs(c[0]-tau)<1e-9 and c[1]==t: v.append(c[idx]); break
    return v
def mn(row,tau,t):
    v = [x for x in allv(row,tau,t,6) if x is not None]
    return (st.mean(v), st.stdev(v)/len(v)**0.5) if v else (None,None)

print("\n=== CHECK 1: mean(tau=1,t=1) vs v2 (expect 15/15 EXACT within 5e-07) ===")
ex = 0
for p in plan:
    m,_ = mn(p["row"],1.0,1); ref = V2_T1[p["row"]]
    good = (m is not None and abs(m-ref) < 5e-7); ex += good
    print("  %-14s v5=%.6f  v2=%.6f  %s" % (p["row"], m, ref, "EXACT" if good else "DIFF %+.2e" % (m-ref)))
print("  => exact %d / 15" % ex)

print("\n=== CHECK 2: D(det) tau-invariance (printed at 2 dp, so this is a 0.01-level check) ===")
viol = 0
for p in plan:
    for t in TS:
        d = {}
        for tau in (0.1,0.5,1.0):
            v = allv(p["row"],tau,t,5)
            if v: d[tau] = (round(min(v),9), round(max(v),9), round(st.mean(v),9))
        if len(d) == 3 and len({d[k] for k in d}) != 1:
            viol += 1
            if viol <= 5: print("   VIOLATION %-14s t=%-3d %s" % (p["row"],t,d))
print("  tau-variant (row,t) cells = %d of %d -> %s" % (
    viol, len(plan)*len(TS), "TAU-INVARIANT CONFIRMED" if viol==0 else "VIOLATED"))

with open("tmp_v5_cells.csv","w",encoding="utf-8",newline="") as fh:
    wr = csv.writer(fh); wr.writerow(["row","rollout","tau","t","A","B","A_over_B","D","D_over_B"])
    for (tag,ro),(cells,status,moved) in sorted(raw.items()):
        for (tau,t,A,B,AB,D,DB) in sorted(cells):
            wr.writerow([tag,ro,"%.2f"%tau,t,"%.4f"%A,"%.4f"%B,"%.4f"%AB,"%.4f"%D,"" if DB is None else "%.6f"%DB])

with open("tmp_v5_table.csv","w",encoding="utf-8",newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["row","probe_dir","override","tau","t","n","DB_mean","DB_se","AB_mean","AB_se","D_mean","x"])
    for p in plan:
        for tau in (0.1,0.5,1.0):
            for t in TS:
                db = [z for z in allv(p["row"],tau,t,6) if z is not None]
                ab = allv(p["row"],tau,t,4); dd = allv(p["row"],tau,t,5)
                if not db: continue
                n = len(db)
                wr.writerow([p["row"],p["probe_dir"],int(p["override"]),"%.2f"%tau,t,n,
                             "%.6f"%st.mean(db),"%.6f"%((st.stdev(db)/n**0.5) if n>1 else 0.0),
                             "%.4f"%st.mean(ab),"%.4f"%((st.stdev(ab)/len(ab)**0.5) if len(ab)>1 else 0.0),
                             "%.4f"%st.mean(dd), "" if xnew.get(p["row"]) is None else "%.6f"%xnew[p["row"]]])

FR = {"s3p10":0.10,"s3p15":0.15,"s3p20":0.20,"s3p25":0.25,"s3p50":0.50,"s3p75":0.75,"s3p90":0.90}
DOSE = sorted(FR, key=lambda z: FR[z])
SEEDS = {"real_s0":0.910,"real_s1":1.431,"real_s2":1.290,"real_s3":1.883,"real_s4":1.848}

print("\n=== HEADLINE A (x-axis FREE): synth3 dose response at tau=1, t=1 ===")
print("  %-8s %11s %12s %10s" % ("row","drive-frac","D/B","se"))
for k in DOSE:
    m,s = mn(k,1.0,1); print("  %-8s %11.2f %12.4f %10.4f" % (k,FR[k],m,s))
ms = [mn(k,1.0,1)[0] for k in DOSE]
cr = next(((DOSE[i],DOSE[i+1]) for i in range(len(DOSE)-1) if ms[i] <= 1.0 <= ms[i+1]), None)
if cr: print("  => D/B crosses 1.0 between drive-frac %.2f and %.2f (%s -> %s)" % (FR[cr[0]],FR[cr[1]],cr[0],cr[1]))
else:  print("  => NO crossing in the measured range")
print("  => real family max D/B = %.4f" % max(mn(k,1.0,1)[0] for k in ("shuf","p25","p50","p75","real")))

print("\n=== HEADLINE B (x = new-x) ===")
XN = {k: xnew.get(k) for k in ("s3p50","s3p75","s3p90")}
XB = {"s3p50":12.705,"s3p75":19.567,"s3p90":24.321}
for tau in (0.1,0.5,1.0):
    for lbl,X in (("new-x",XN),("brief-x[SUPERSEDED]",XB)):
        pts = sorted((X[k],mn(k,tau,1)[0],k) for k in ("s3p50","s3p75","s3p90") if X.get(k) is not None)
        pr = next(((pts[i],pts[i+1]) for i in range(len(pts)-1) if pts[i][1] <= 1.0 <= pts[i+1][1]), None)
        if not pr:
            print("   tau=%.1f %-22s <no bracket> pts=%s" % (tau,lbl,[(p[2],round(p[1],3)) for p in pts])); continue
        (x1,r1,k1),(x2,r2,k2) = pr
        xc = x1 + (1-r1)/(r2-r1)*(x2-x1)
        zs = [xc/v for v in SEEDS.values()]
        print("   tau=%.1f %-22s %s->%s x_cross=%6.2f mean=%.2f median=%.2f min=%.2f max=%.2f >=15x %d/5 point(real s0)=%.2fx" % (
            tau,lbl,k1,k2,xc,st.mean(zs),st.median(zs),min(zs),max(zs),sum(1 for z in zs if z>=15),xc/0.910))

with open("tmp_v5_headline.csv","w",encoding="utf-8",newline="") as fh:
    wr = csv.writer(fh); wr.writerow(["kind","row","tau","drive_frac","x_new","DB_mean","DB_se"])
    for k in DOSE:
        m,s = mn(k,1.0,1); wr.writerow(["dose",k,1.0,FR[k],"" if xnew.get(k) is None else "%.6f"%xnew[k],"%.6f"%m,"%.6f"%s])
    for k in ("shuf","p25","p50","p75","real"):
        m,s = mn(k,1.0,1); wr.writerow(["real-family",k,1.0,"","" if xnew.get(k) is None else "%.6f"%xnew[k],"%.6f"%m,"%.6f"%s])
LOG.close()
print("\nwrote tmp_v5_protocol.csv, tmp_v5_cells.csv, tmp_v5_table.csv, tmp_v5_headline.csv, tmp_v5_log.txt ; raw in tmp_v5/")
