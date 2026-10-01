import csv, statistics as st
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
PR = {}
for r in prov:
    PR.setdefault(r["row"], []).append(r)

ROWS = ["shuf","p25","p50","p75","real","s3p10","s3p15","s3p20","s3p25","s3p50","s3p75","s3p90","PC2"]
B = {"shuf":(0.45,0.024),"p25":(0.24,0.02),"p50":(0.32,0.03),"p75":(0.66,0.09),"real":(0.72,0.093),
     "s3p10":(2.60,0.15),"s3p15":(3.63,0.17),"s3p20":(5.64,0.17),"s3p25":(7.53,0.23),
     "s3p50":(7.03,0.45),"s3p75":(6.42,0.69),"s3p90":(6.89,1.081),"PC2":(33.06,1.283)}

o = []
def w(s=""):
    o.append(s); print(s)

w("P1) table aggregate vs recomputed from cells (tau=1, t=1)")
for row in ROWS:
    rs = C[(row, 1.0, 1)]; r = T[(row, 1.0, 1)]
    mab = st.mean(fn(x["A_over_B"]) for x in rs)
    mdb = st.mean(fn(x["D_over_B"]) for x in rs)
    md  = st.mean(fn(x["D"]) for x in rs)
    ma  = st.mean(fn(x["A"]) for x in rs)
    mb  = st.mean(fn(x["B"]) for x in rs)
    w("  %-6s AB %8.4f/%-8.4f %s | DB %.6f/%.6f %s | D %8.3f/%-8.3f %s | meanA %8.3f meanB %8.3f"
      % (row, fn(r["AB_mean"]), mab, "ok" if abs(fn(r["AB_mean"])-mab) < 5e-4 else "XX",
         fn(r["DB_mean"]), mdb, "ok" if abs(fn(r["DB_mean"])-mdb) < 5e-6 else "XX",
         fn(r["D_mean"]), md, "ok" if abs(fn(r["D_mean"])-md) < 5e-3 else "XX", ma, mb))

w(""); w("P2) is table D_mean tau-independent at t=1 ?")
for row in ROWS:
    v = [fn(T[(row, tau, 1)]["D_mean"]) for tau in (0.1, 0.5, 1.0)]
    w("  %-6s tau .1/.5/1.0 = %8.3f %8.3f %8.3f   spread %+.4f" % (row, v[0], v[1], v[2], max(v)-min(v)))

w(""); w("P3) Brief 2.5 D(det) vs table D_mean(t=1,tau=1), and closest table cell")
for row in ROWS:
    bd = B[row][0]; td = fn(T[(row, 1.0, 1)]["D_mean"])
    best = min(((abs(fn(T[k]["D_mean"]) - bd), k) for k in T if k[0] == row), key=lambda z: z[0])
    w("  %-6s brief %7.2f | table(t1) %7.3f (%+6.1f%%) | closest: tau=%.1f t=%3d val %7.3f d %+7.3f"
      % (row, bd, td, (td-bd)/bd*100.0, best[1][1], best[1][2], fn(T[best[1]]["D_mean"]), best[0]))

w(""); w("P4) Brief 2.5 D(det) vs v2-provenance D_det (20 rollouts)")
for row in ROWS:
    ds = [fn(p["D_det"]) for p in PR.get(row, [])]
    bs = [fn(p["B_noise"]) for p in PR.get(row, [])]
    if not ds:
        w("  %-6s (no provenance rows)" % row); continue
    db = [fn(p["DoverB"]) for p in PR[row]]
    w("  %-6s brief %7.2f | prov D_det mean %7.3f min %6.3f max %7.3f | B_noise mean %8.2f | D/B mean %.6f"
      % (row, B[row][0], st.mean(ds), min(ds), max(ds), st.mean(bs), st.mean(db)))

w(""); w("END")
open("tmp_n18_audit_p2.txt", "w", encoding="utf-8").write("\n".join(o) + "\n")
print("written -> tmp_n18_audit_p2.txt (%d lines)" % (len(o)+1))

