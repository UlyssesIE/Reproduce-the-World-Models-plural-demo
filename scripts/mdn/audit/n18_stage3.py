import os, csv
def load(p):
    with open(p, newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))
def fn(s):
    try: return float(s)
    except Exception: return None

T = {(r["row"], fn(r["tau"]), int(r["t"])): r for r in load("tmp_v5_table.csv")}
TAU = [0.1, 0.5, 1.0]
ROWS = ["shuf","p25","p50","p75","real","s3p10","s3p15","s3p20","s3p25","s3p50","s3p75","s3p90","PC2"]
BAB = {"shuf":0.024,"p25":0.02,"p50":0.03,"p75":0.09,"real":0.093,"s3p10":0.15,"s3p15":0.17,
       "s3p20":0.17,"s3p25":0.23,"s3p50":0.45,"s3p75":0.69,"s3p90":1.081,"PC2":1.283}
BD  = {"shuf":0.45,"p25":0.24,"p50":0.32,"p75":0.66,"real":0.72,"s3p10":2.60,"s3p15":3.63,
       "s3p20":5.64,"s3p25":7.53,"s3p50":7.03,"s3p75":6.42,"s3p90":6.89,"PC2":33.06}

o = []
def w(s=""):
    o.append(s); print(s)

w("Q1) full (tau,t) search: closest table cell to Brief 2.5 A/B / D(det)")
for row in ROWS:
    bA = min(((abs(fn(T[k]["AB_mean"]) - BAB[row]), k) for k in T if k[0] == row), key=lambda z: z[0])
    bD = min(((abs(fn(T[k]["D_mean"])  - BD[row]),  k) for k in T if k[0] == row), key=lambda z: z[0])
    w("  %-6s A/B brief %6.3f | closest tau=%.1f t=%3d val %7.4f d %+.4f | D brief %7.2f | closest tau=%.1f t=%3d val %8.3f d %+.3f"
      % (row, BAB[row], bA[1][1], bA[1][2], fn(T[bA[1]]["AB_mean"]), bA[0],
         BD[row], bD[1][1], bD[1][2], fn(T[bD[1]]["D_mean"]), bD[0]))

w(""); w("Q2) table at t=200, tau=1  (test: is brief D from t=200?)")
for row in ROWS:
    r = T[(row, 1.0, 200)]
    w("  %-6s t200 D=%8.3f AB=%.4f   ||  brief D=%6.2f AB=%.3f" % (row, fn(r["D_mean"]), fn(r["AB_mean"]), BD[row], BAB[row]))

w(""); w("Q3) bounded repo scan for distinctive Brief tokens")
TOKS = ["33.0600","5.6400","7.5300","3.6300","2.6000","6.8900","6.4200","1.0810","0.0930","0.156493"]
SKIP = {".git","data","runs",".venv","__pycache__"}
hits = {t: [] for t in TOKS}; n = 0
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in SKIP]
    for f in files:
        if os.path.splitext(f)[1].lower() in (".npy",".pt",".pth",".png",".jpg",".zip",".bundle"):
            continue
        p = os.path.join(root, f)
        try:
            if os.path.getsize(p) > 5000000: continue
            s = open(p, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        n += 1
        for t in TOKS:
            if t in s: hits[t].append(p)
w("  scanned %d text files" % n)
for t in TOKS:
    w("  %-10s %4d files   e.g. %s" % (t, len(hits[t]), hits[t][:3]))

w(""); w("Q4) DELIVERABLE table -> tmp_n18_deliverable.csv")
hdr = ["row","family","tau","t","DB_mean","DB_se","AB_mean","AB_se","D_mean","x"]
lines = [",".join(hdr)]
for row in ROWS + ["real-struct", "synth3-struct"]:
    fam = "real" if row in ("shuf","p25","p50","p75","real","real-struct") else "synth3"
    for tau in TAU:
        for t in (1, 200):
            if (row, tau, t) not in T: continue
            r = T[(row, tau, t)]
            lines.append(",".join([row, fam, "%.1f" % tau, str(t), r["DB_mean"], r["DB_se"],
                                   r["AB_mean"], r["AB_se"], r["D_mean"], r["x"]]))
open("tmp_n18_deliverable.csv", "w", encoding="utf-8").write("\n".join(lines) + "\n")
w("  wrote tmp_n18_deliverable.csv (%d data lines)" % (len(lines) - 1))
for l in lines[:5]: w("    " + l)

w(""); w("END")
open("tmp_n18_audit_p3.txt", "w", encoding="utf-8").write("\n".join(o) + "\n")
print("written -> tmp_n18_audit_p3.txt (%d lines)" % (len(o) + 1))

