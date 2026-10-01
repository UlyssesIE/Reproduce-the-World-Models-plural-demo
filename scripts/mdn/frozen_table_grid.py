"""One-cycle frozen table v4 -- FULL (t, tau) grid; D/B and A/B both reported.

DECLARED definitions (not inferred).  For every rollout r = 0..19 the instrument
is run ONCE with all three taus:

  scripts/mdn/dream_controllability.py --mdn <ckpt>
      --latents <ckpt.args.latents> --rollout r
      --steps 200 --warmup 16 --reps 50 --taus 0.1,0.5,1.0

It prints, for each (tau, t) with t in {1,2,4,8,16,32,64,128,200}:
      tau  t  A(stoch)+-sd  B(noise)+-sd  A/B  D(det) | medA medB medRatio

  D/B := D(det) / B(noise)      (the table's 'D/B' column)
  A/B := the printed A/B column

x(row) is tau-independent (teacher-forced one-step NLL, no temperature):
  x = NLL(lam=0) - NLL(lam=1)   metric_validation.py --split val --seed 0

latents paths are read from the checkpoints themselves.
Raw: tmp_v4/*.txt  |  commands: tmp_v4_log.txt  |  cells: tmp_v4_cells.csv
Aggregates: tmp_v4_table.csv
"""
import csv, os, re, statistics as st, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import torch

ROWS = [
 ("shuf",          r"runs\mdn_pilot_shuf\best.pt",            True),
 ("p25",           r"runs\mdn_pilot_p25\best.pt",             True),
 ("p50",           r"runs\mdn_pilot_p50\best.pt",             True),
 ("p75",           r"runs\mdn_pilot_p75\best.pt",             True),
 ("real",          r"runs\mdn_pilot\best.pt",                 True),
 ("real-struct",   r"runs\mdn_pilot_noact\best.pt",           False),
 ("s3p10",         r"runs\mdn_pilot_synth3p10\best.pt",       True),
 ("s3p15",         r"runs\mdn_pilot_synth3p15\best.pt",       True),
 ("s3p20",         r"runs\mdn_pilot_synth3p20\best.pt",       True),
 ("s3p25",         r"runs\mdn_pilot_synth3p25\best.pt",       True),
 ("s3p50",         r"runs\mdn_pilot_synth3p50\best.pt",       True),
 ("s3p75",         r"runs\mdn_pilot_synth3p75\best.pt",       True),
 ("s3p90",         r"runs\mdn_pilot_synth3p90\best.pt",       True),
 ("PC2",           r"runs\mdn_pilot_pc2\best.pt",             False),
 ("synth3-struct", r"runs\mdn_pilot_synth3noact\best.pt",     False),
]
V2_T1 = {"shuf":0.023117,"p25":0.015406,"p50":0.024856,"p75":0.081848,
         "real":0.089631,"real-struct":0.156493,"s3p10":0.142133,"s3p15":0.145003,
         "s3p20":0.184936,"s3p25":0.270098,"s3p50":0.464836,"s3p75":0.667618,
         "s3p90":1.059601,"PC2":1.281279,"synth3-struct":0.731066}
TAUS, TS, NR, WORKERS = "0.1,0.5,1.0", [1,2,4,8,16,32,64,128,200], 20, 6
PROTO = ["--steps","200","--warmup","16","--reps","50","--taus",TAUS]
OUT = Path("tmp_v4"); OUT.mkdir(exist_ok=True)
ENV = dict(os.environ); ENV.update({"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","PYTHONIOENCODING":"utf-8"})
RE_CELL = re.compile(r"^\s*([\d.]+)\s+(\d+)\s+([\d.]+)\s*\u00b1\s*([\d.]+)\s+([\d.]+)\s*\u00b1\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)")
RE_X = re.compile(r"lam=([\d.]+).*?NLL\s+(-?[\d.]+)")
LOG = open("tmp_v4_log.txt","w",encoding="utf-8")

def latents_of(ck):
    return torch.load(ck, map_location="cpu").get("args",{}).get("latents")

def call(cmd, path):
    LOG.write(":: "+" ".join(cmd)+"\n"); LOG.flush()
    if path.exists() and path.stat().st_size>0:
        return path.read_text(encoding="utf-8",errors="replace"),"cached"
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=ENV)
    txt=(p.stdout or "")+(p.stderr or ""); path.write_text(txt,encoding="utf-8")
    return txt, ("ok" if p.returncode==0 else "exit%d"%p.returncode)

def task_db(tag, ck, lat, ro):
    f = OUT/("%s_r%02d.txt"%(Path(ck).parent.name, ro))
    cmd=[sys.executable,"scripts/mdn/dream_controllability.py","--mdn",ck,
         "--latents",lat,"--rollout",str(ro)]+PROTO
    txt,status = call(cmd,f); cells=[]
    for ln in txt.splitlines():
        m=RE_CELL.match(ln)
        if m:
            tau=float(m.group(1)); t=int(m.group(2))
            if abs(tau-0.1)<1e-9 or abs(tau-0.5)<1e-9 or abs(tau-1.0)<1e-9:
                A,B,AB,D=float(m.group(3)),float(m.group(5)),float(m.group(7)),float(m.group(8))
                cells.append((tau,t,A,B,AB,D,D/B if B else None))
    return (tag,ro,cells,status)

def task_x(tag, ck, lat):
    f = OUT/("x_%s.txt"%Path(ck).parent.name)
    cmd=[sys.executable,"scripts/mdn/metric_validation.py","--mdn",ck,
         "--latents",lat,"--split","val","--seed","0"]
    txt,status = call(cmd,f); d={}
    for ln in txt.splitlines():
        m=RE_X.search(ln)
        if m: d[float(m.group(1))]=float(m.group(2))
    return (tag, (d[0.0]-d[1.0]) if (0.0 in d and 1.0 in d) else None, status)

lat={}
for tag,ck,hx in ROWS:
    lat[tag]=latents_of(ck); print("latents %-14s %s"%(tag,lat[tag]))

tasks=[(t,c,lat[t],r) for t,c,h in ROWS if lat[t] for r in range(NR)]
print("\n=== phase 1: D/B + A/B, %d invocations x %s, %d parallel ==="%(len(tasks),TAUS,WORKERS))
raw={}; done=0
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    for fu in as_completed([ex.submit(task_db,*t) for t in tasks]):
        tag,ro,cells,status=fu.result(); raw[(tag,ro)]=(cells,status); done+=1
        if done%20==0 or done==len(tasks): print("  %3d/%d"%(done,len(tasks)),flush=True)
bad=[]; empty=[]
for k,(cells,status) in raw.items():
    if not cells: empty.append((k,status))
    if len(cells)!=3*len(TS): bad.append((k,len(cells),status))
print("  empty=%d  wrong-cell-count=%d  %s"%(len(empty),len(bad),bad[:6]))

print("\n=== phase 2: x (%d) ==="%sum(1 for t,c,h in ROWS if h))
xnew={}
with ThreadPoolExecutor(max_workers=4) as ex:
    for fu in as_completed([ex.submit(task_x,t,c,lat[t]) for t,c,h in ROWS if h and lat[t]]):
        tag,x,status=fu.result(); xnew[tag]=x
        print("  %-8s x=%-9s %s"%(tag,("%.3f"%x) if x is not None else "ERR",status),flush=True)

with open("tmp_v4_cells.csv","w",encoding="utf-8",newline="") as f:
    w=csv.writer(f); w.writerow(["row","rollout","tau","t","A","B","A_over_B","D","D_over_B"])
    for (tag,ro),(cells,status) in sorted(raw.items()):
        for (tau,t,A,B,AB,D,DB) in sorted(cells):
            w.writerow([tag,ro,"%.2f"%tau,t,"%.4f"%A,"%.4f"%B,"%.4f"%AB,"%.4f"%D,
                        "" if DB is None else "%.6f"%DB])

print("\n=== check: tau=1.0, t=1 vs v2 table ===")
ok=diff=0
for tag,ck,hx in ROWS:
    cells=raw.get((tag,0),(None,))[0]
    if not cells: continue
    hit=[c for c in cells if abs(c[0]-1.0)<1e-9 and c[1]==1]
    if not hit: continue
    v=hit[0][6]
    if abs(v-V2_T1[tag])<1e-9: ok+=1
    else: diff+=1; print("  DIFF %-14s v4=%.6f v2=%.6f"%(tag,v,V2_T1[tag]))
print("  matched=%d  DIFF=%d"%(ok,diff))

print("\n=== aggregates: D/B and A/B surfaces (mean over 20 rollouts) ===")
with open("tmp_v4_table.csv","w",encoding="utf-8",newline="") as f:
    w=csv.writer(f); w.writerow(["row","tau","t","n","DB_mean","DB_se","AB_mean","AB_se","D_mean","x"])
    for tag,ck,hx in ROWS:
        for tau in (0.1,0.5,1.0):
            for t in TS:
                vs=[]; ab=[]; dd=[]
                for r in range(NR):
                    cells=raw.get((tag,r),(None,))[0]
                    if not cells: continue
                    hit=[c for c in cells if abs(c[0]-tau)<1e-9 and c[1]==t]
                    if hit and hit[0][6] is not None:
                        vs.append(hit[0][6]); ab.append(hit[0][4]); dd.append(hit[0][5])
                if not vs: continue
                n=len(vs); mu=st.mean(vs); sd=st.stdev(vs) if n>1 else 0.0
                se=sd/n**0.5 if n>1 else 0.0
                am=st.mean(ab); asd=st.stdev(ab) if n>1 else 0.0
                w.writerow([tag,"%.2f"%tau,t,n,"%.6f"%mu,"%.6f"%se,"%.4f"%am,
                            "%.4f"%(asd/n**0.5 if n>1 else 0.0),"%.4f"%st.mean(dd),
                            "" if xnew.get(tag) is None else "%.6f"%xnew[tag]])

print("\n=== tau-dependence spotlight (D/B at t=1 and A/B at t=200) ===")
print("%-14s %10s %10s %10s | %10s %10s %10s"%("row","DB(t1,0.1)","DB(t1,0.5)","DB(t1,1.0)","AB(t200,0.1)","AB(t200,0.5)","AB(t200,1.0)"))
for tag,ck,hx in ROWS:
    def val(tau,t,idx):
        xs=[]
        for r in range(NR):
            cells=raw.get((tag,r),(None,))[0]
            if not cells: continue
            hit=[c for c in cells if abs(c[0]-tau)<1e-9 and c[1]==t]
            if hit: xs.append(hit[0][idx] if idx==6 else hit[0][4])
        return st.mean(xs) if xs else float("nan")
    print("%-14s %10.3f %10.3f %10.3f | %10.3f %10.3f %10.3f"%(
        tag,val(0.1,1,6),val(0.5,1,6),val(1.0,1,6),val(0.1,200,4),val(0.5,200,4),val(1.0,200,4)))
LOG.close()
print("\nwrote tmp_v4_cells.csv, tmp_v4_table.csv, tmp_v4_log.txt ; raw in tmp_v4/")
