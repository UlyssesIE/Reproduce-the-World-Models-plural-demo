import os, csv, hashlib, statistics as st

def load(p):
    with open(p, newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))
def fn(s):
    try: return float(s)
    except Exception: return None

table = load("tmp_v5_table.csv")
cells = load("tmp_v5_cells.csv")
prov  = load("tmp_final_provenance_v2.csv")
T = {(r["row"], fn(r["tau"]), int(r["t"])): r for r in table}
C = {}
for r in cells:
    C.setdefault((r["row"], fn(r["tau"]), int(r["t"])), []).append(r)

REAL = ["shuf","p25","p50","p75","real","real-struct"]
SYN  = ["s3p10","s3p15","s3p20","s3p25","s3p50","s3p75","s3p90","PC2","synth3-struct"]
DOSE = ["s3p10","s3p15","s3p20","s3p25","s3p50","s3p75","s3p90"]
ALL  = REAL + SYN
TAUS = [0.1, 0.5, 1.0]
T200 = 200

H = {
 "shuf":(0.019,0.023117,0.024,0.45), "p25":(-0.045,0.015406,0.02,0.24),
 "p50":(-0.131,0.024856,0.03,0.32), "p75":(0.163,0.081848,0.09,0.66),
 "real":(0.910,0.089631,0.093,0.72), "real-struct":(None,0.156493,None,None),
 "s3p10":(0.685,0.142133,0.15,2.60), "s3p15":(2.161,0.145003,0.17,3.63),
 "s3p20":(2.435,0.184936,0.17,5.64), "s3p25":(6.813,0.270098,0.23,7.53),
 "s3p50":(13.993,0.464836,0.45,7.03), "s3p75":(22.053,0.667618,0.69,6.42),
 "s3p90":(26.837,1.059601,1.081,6.89), "PC2":(None,1.281279,1.283,33.06),
 "synth3-struct":(None,0.731066,None,None),
}

out = []
def w(s=""):
    out.append(s); print(s)

w("="*96)
w("N18 AUDIT v2 (COMPLETE)  fixes: #103 'tt' unbound, #106 tuple index, #107 delivery")
w("="*96)
w("table rows=%d  cells rows=%d  taus=%s" % (len(table), len(cells), sorted({fn(r['tau']) for r in table})))

w(""); w("-"*96); w("A) A/B(t=200) monotonicity in tau"); w("-"*96)
c = {}
for row in ALL:
    v = [fn(T[(row,tau,T200)]["AB_mean"]) if (row,tau,T200) in T else None for tau in TAUS]
    if any(x is None for x in v):
        w("  %-14s MISSING" % row); continue
    lab = "mono_rise" if v[0]<=v[1]<=v[2] else ("mono_fall" if v[0]>=v[1]>=v[2] else "non_mono")
    c[lab] = c.get(lab,0)+1
    w("  %-14s %-7s %7.3f %7.3f %7.3f  %s" % (row, "real" if row in REAL else "synth3", v[0],v[1],v[2], lab))
w("  mono_rise=%d mono_fall=%d non_mono=%d" % (c.get("mono_rise",0),c.get("mono_fall",0),c.get("non_mono",0)))

w(""); w("-"*96); w("B) D/B vs A/B at t=1 (table columns, rollout means)"); w("-"*96)
lst = []
for row in ALL:
    for tau in TAUS:
        r = T.get((row,tau,1))
        if not r: continue
        db, ab = fn(r["DB_mean"]), fn(r["AB_mean"])
        if db is None or not ab: continue
        lst.append((abs(db-ab)/ab*100.0, row, tau, db, ab))
lst.sort(reverse=True)
for rel,row,tau,db,ab in lst[:10]:
    w("  %-14s tau=%.1f D/B=%.6f A/B=%.4f rel=%5.2f%%" % (row,tau,db,ab,rel))
w("  max rel = %.2f%%   criterion <6%% : %s" % (lst[0][0], "PASS" if lst[0][0]<6 else "FAIL"))
w("  (row,tau) with rel<6%%: %d / %d" % (sum(1 for x in lst if x[0]<6), len(lst)))

w(""); w("-"*96); w("C) x_cross recompute (crossing 1.0, linear in x, t=1)"); w("-"*96)
EXP = {0.1:17.93, 0.5:23.36, 1.0:26.11}
def cross(tau, key):
    pts = []
    for row in DOSE:
        r = T.get((row,tau,1))
        if not r: continue
        y = fn(r[key]); x = fn(r["x"])
        if y is None or x is None: continue
        pts.append((x, y, row))
    pts.sort()
    for i in range(len(pts)-1):
        x0,y0,r0 = pts[i]; x1,y1,r1 = pts[i+1]
        if (y0-1.0)*(y1-1.0) <= 0 and y1 != y0:
            return (x0 + (1.0-y0)/(y1-y0)*(x1-x0), r0,x0,y0, r1,x1,y1)
    return None
for tau in TAUS:
    for key,lab in (("DB_mean","D/B"), ("AB_mean","A/B")):
        c1 = cross(tau,key)
        if c1 is None:
            w("  tau=%.1f  %-3s  NO crossing in dose grid" % (tau,lab)); continue
        xc,r0,x0,y0,r1,x1,y1 = c1
        tail = ("   [brief N17=%.2f  delta=%+.2f]" % (EXP[tau], xc-EXP[tau])) if key=="DB_mean" else ""
        w("  tau=%.1f  %-3s  x_cross=%7.2f   %s x=%.3f val=%.4f  ->  %s x=%.3f val=%.4f%s"
          % (tau,lab,xc, r0,x0,y0, r1,x1,y1, tail))

w(""); w("-"*96); w("D) credential md5 (11 items)"); w("-"*96)
CRED = ["tmp_v5_protocol.csv","tmp_v5_cells.csv","tmp_v5_table.csv","tmp_v5_headline.csv",
        "tmp_v5_notnote11.csv","tmp_v5_log.txt","tmp_seed_summary.csv","tmp_seed_dbread.csv",
        "tmp_final_table_v2.csv","tmp_final_provenance_v2.csv","manifest_ckpt.txt"]
def md5(p):
    h = hashlib.md5()
    with open(p,"rb") as f:
        for b in iter(lambda: f.read(1<<20), b""): h.update(b)
    return h.hexdigest().upper()
for p in CRED:
    if os.path.exists(p):
        w("  OK       %9d B  %s  %s" % (os.path.getsize(p), md5(p)[:12], p))
    else:
        w("  MISSING  %9s    %s  %s" % ("-","-"*12,p))

w(""); w("-"*96); w("E) provenance sources + tmp inventory"); w("-"*96)
srcs = set(r["source"] for r in prov if r.get("source"))
miss = sorted(s for s in srcs if not os.path.exists(s))
w("  provenance_v2: %d distinct source files, %d missing on disk" % (len(srcs), len(miss)))
for s in miss[:12]: w("     MISSING SRC: %s" % s)
top = [f for f in sorted(os.listdir(".")) if f.startswith("tmp_")]
w("  tmp_* top-level = %d entries ; tmp_*.py = %d" % (len(top), sum(1 for f in top if f.endswith(".py"))))

w(""); w("-"*96); w("R) reconcile Section 2.5 columns vs tmp_v5_table.csv (tau=1,t=1)"); w("-"*96)
w("  %-14s| %-9s %-9s | %-9s %-9s | %-8s %-8s | flags"
  % ("row","DB tbl","DB brief","AB tbl","AB brief","D tbl","D brief"))
for row in ALL:
    r = T.get((row,1.0,1))
    if not r:
        w("  %-14s MISSING" % row); continue
    db,ab,dm = fn(r["DB_mean"]), fn(r["AB_mean"]), fn(r["D_mean"])
    hx,hdb,hab,hd = H[row]
    fl = []
    if hdb is not None and db is not None:
        fl.append("DB " + ("ok" if abs(db-hdb)<=5e-5 else "MISMATCH"))
    if hab is not None and ab is not None:
        e = (ab-hab)/hab*100.0
        fl.append("AB %+.1f%%%s" % (e, "" if abs(e)<=5 else " <<<"))
    if hd is not None and dm is not None:
        e = (dm-hd)/hd*100.0
        fl.append("D %+.1f%%%s" % (e, "" if abs(e)<=5 else " <<<"))
    w("  %-14s| %9.6f %9.6f | %9.4f %9.4f | %8.3f %8.3f | %s"
      % (row, db if db is not None else -1, hdb if hdb is not None else -1,
         ab if ab is not None else -1, hab if hab is not None else -1,
         dm if dm is not None else -1, hd if hd is not None else -1, "  ".join(fl)))

w(""); w("-"*96); w("S) 'A/B' column statistic + PAIRED sigma test (tau=1,t=1)"); w("-"*96)
for row in ["s3p10","s3p15","s3p20","s3p25","s3p50","shuf","real","s3p90","PC2"]:
    rs = C.get((row,1.0,1), [])
    if not rs:
        w("  %-14s MISSING cells" % row); continue
    ab = [fn(x["A_over_B"]) for x in rs]
    A  = [fn(x["A"]) for x in rs]; B = [fn(x["B"]) for x in rs]
    d  = [a-b for a,b in zip(ab,[fn(x["D_over_B"]) for x in rs])]
    m  = st.mean(d); sd = st.stdev(d) if len(d)>1 else 0.0; se = sd/len(d)**0.5
    w("  %-14s briefA/B=%-7s mean(A/B)=%.4f med(A/B)=%.4f "
      "mean(A)/mean(B)=%.4f | paired(A/B-D/B)=%+.5f+-%.5f  z=%+.1f sigma"
      % (row, H[row][2], st.mean(ab), st.median(ab), st.mean(A)/st.mean(B), m, se, (m/se if se else 0.0)))
w("  s3p10 A_over_B per rollout: %s" % ", ".join("%.3f"%v for v in sorted(fn(x["A_over_B"]) for x in C[("s3p10",1.0,1)])))
w("  s3p10 D_over_B per rollout: %s" % ", ".join("%.3f"%v for v in sorted(fn(x["D_over_B"]) for x in C[("s3p10",1.0,1)])))

w(""); w("-"*96); w("T) tmp_v5/ raw naming (to locate the medRatio tail)"); w("-"*96)
if os.path.isdir("tmp_v5"):
    fs = sorted(os.listdir("tmp_v5"))
    w("  n=%d" % len(fs))
    w("  first 24: %s" % fs[:24])
    w("  last   6: %s" % fs[-6:])
else:
    w("  tmp_v5/ absent -- raw per-rollout dumps are not shipped (only the derived")
    w("  tmp_v5_*.csv surfaces are, under notes/artifacts/). Re-run frozen_table_v5.py")
    w("  against the checkpoints to regenerate them.")

w(""); w("END")
with open("tmp_n18_audit.txt","w",encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
print("written -> tmp_n18_audit.txt  (%d lines)" % (len(out)+1))

