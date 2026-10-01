import os, csv, hashlib
def load(p):
    with open(p, newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))
def fn(s):
    try: return float(s)
    except Exception: return None
def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest().upper()

T = {(r["row"], fn(r["tau"]), int(r["t"])): r for r in load("tmp_v5_table.csv")}
TAU = [0.1, 0.5, 1.0]
REAL = ["shuf","p25","p50","p75","real","real-struct"]
SYN  = ["s3p10","s3p15","s3p20","s3p25","s3p50","s3p75","s3p90","PC2","synth3-struct"]
ALL  = REAL + SYN

o = []
def w(s=""):
    o.append(s); print(s)

w("TABLE A   t=1, tau=1.0   (from tmp_v5_table.csv)")
w("  %-14s %-7s %10s %11s %11s %11s %11s %10s" %
  ("row","family","x","DB_mean","DB_se","AB_mean","AB_se","D_mean"))
for row in ALL:
    r = T.get((row, 1.0, 1))
    if not r:
        w("  %-14s MISSING" % row); continue
    xv = fn(r["x"])
    w("  %-14s %-7s %10.3f %11.6f %11.6f %11.4f %11.4f %10.3f" %
      (row, "real" if row in REAL else "synth3", xv if xv is not None else float("nan"),
       fn(r["DB_mean"]), fn(r["DB_se"]), fn(r["AB_mean"]), fn(r["AB_se"]), fn(r["D_mean"])))

w("")
w("TABLE B   A/B at t=200 by tau  (the family separator)")
for row in ALL:
    v = [fn(T[(row, tau, 200)]["AB_mean"]) if (row, tau, 200) in T else None for tau in TAU]
    if any(z is None for z in v):
        w("  %-14s MISSING" % row); continue
    lab = "mono_rise" if v[0] <= v[1] <= v[2] else ("mono_fall" if v[0] >= v[1] >= v[2] else "non_mono")
    w("  %-14s %7.4f %7.4f %7.4f   %s" % (row, v[0], v[1], v[2], lab))

w("")
w("CREDENTIALS  (KEEP set)")
KEEP = ["tmp_v5_protocol.csv","tmp_v5_cells.csv","tmp_v5_table.csv","tmp_v5_headline.csv",
        "tmp_v5_notnote11.csv","tmp_v5_log.txt","tmp_seed_summary.csv","tmp_seed_dbread.csv",
        "tmp_final_table_v2.csv","tmp_final_provenance_v2.csv","manifest_ckpt.txt",
        "tmp_n18_audit.txt","tmp_n18_audit_p2.txt","tmp_n18_audit_p3.txt","tmp_n18_deliverable.csv",
        "tmp_n18_audit.py","tmp_n18_audit_p2.py","tmp_n18_audit_p3.py"]
lines = ["file,bytes,md5_12"]
for p in KEEP:
    if os.path.exists(p):
        m = md5(p); sz = os.path.getsize(p)
        w("  OK       %9d B  %s  %s" % (sz, m[:12], p))
        lines.append("%s,%d,%s" % (p, sz, m[:12]))
    else:
        w("  MISSING  %9s    %s  %s" % ("-", "-" * 12, p))
        lines.append("%s,0,MISSING" % p)
open("tmp_n18_credentials.csv", "w", encoding="utf-8").write("\n".join(lines) + "\n")
w("  wrote tmp_n18_credentials.csv  (%d items)" % len(KEEP))

top = [f for f in os.listdir(".") if f.startswith("tmp_")]
w("")
w("tmp_* top-level = %d ; tmp_*.py = %d" % (len(top), sum(1 for f in top if f.endswith(".py"))))
w("END")
open("tmp_n18_audit_p4.txt", "w", encoding="utf-8").write("\n".join(o) + "\n")
print("written -> tmp_n18_audit_p4.txt (%d lines)" % (len(o) + 1))

