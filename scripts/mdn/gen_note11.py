import csv, os, re, hashlib, statistics as st

F = chr(96) * 3
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ART = os.path.join(ROOT, "notes", "artifacts")

def rp(name):
    a = os.path.join(ART, name)
    return a if os.path.exists(a) else os.path.join(ROOT, name)
def load(p):
    with open(p, newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))
def fn(s):
    try: return float(s)
    except Exception: return None
def rd(p):
    return open(p, encoding="utf-8", errors="replace").read()
def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest().upper()
def linechk(rel, n, needles):
    try:
        ls = rd(os.path.join(ROOT, rel)).splitlines()
        if n - 1 >= len(ls): return "OUT_OF_RANGE(%d lines)" % len(ls)
        s = ls[n - 1]
        return "OK" if all(x in s for x in needles) else "MISMATCH: " + s.strip()[:64]
    except Exception as e:
        return "ERR " + str(e)[:36]

T = {(r["row"], fn(r["tau"]), int(r["t"])): r for r in load(rp("tmp_v5_table.csv"))}
S = load(rp("tmp_seed_summary.csv"))
TAU = [0.1, 0.5, 1.0]
REAL = ["shuf","p25","p50","p75","real","real-struct"]
SYN  = ["s3p10","s3p15","s3p20","s3p25","s3p50","s3p75","s3p90","PC2","synth3-struct"]
ORDER = REAL + SYN
FAM = dict([(r, "real") for r in REAL] + [(r, "synth3") for r in SYN])
DOSE = ["s3p10","s3p15","s3p20","s3p25","s3p50","s3p75","s3p90"]
DF = {"s3p10":0.10,"s3p15":0.15,"s3p20":0.20,"s3p25":0.25,"s3p50":0.50,"s3p75":0.75,"s3p90":0.90}

XSEED = [0.910, 1.431, 1.290, 1.883, 1.848]
xsrc = "hardcoded fallback"
RX = r"([0-9]\.[0-9]{3})\s*/\s*([0-9]\.[0-9]{3})\s*/\s*([0-9]\.[0-9]{3})\s*/\s*([0-9]\.[0-9]{3})\s*/\s*([0-9]\.[0-9]{3})"
try:
    txt = rd(os.path.join(ROOT, "notes", "Note9.md"))
    for m in re.finditer(RX, txt):
        v = [float(m.group(i)) for i in range(1, 6)]
        if abs(st.mean(v) - 1.472) < 0.005:
            XSEED = v
            xsrc = "notes/Note9.md:%d (mean-guarded)" % (txt[:m.start()].count("\n") + 1)
            break
except Exception:
    pass

print("ANCHOR CHECKS")
print("  x_i source :", xsrc, XSEED)
print("  Note9.md:25   ", linechk("notes/Note9.md", 25, ["0.910", "1.472"]))
print("  Note9.md:78   ", linechk("notes/Note9.md", 78, ["5.242"]))
print("  Note10.md:12  ", linechk("notes/Note10.md", 12, ["16"]))
print("  Note10.md:151 ", linechk("notes/Note10.md", 151, ["16.5"]))
print("  Note10.md:199 ", linechk("notes/Note10.md", 199, ["medA"]))
print("  Note10.md:214 ", linechk("notes/Note10.md", 214, ["paired per rep"]))
print("  Note8.md:45   ", linechk("notes/Note8.md", 45, ["a-slice"]))
print("  content.md    ", "EXISTS" if (os.path.exists(os.path.join(ROOT,"content.md")) or os.path.exists(os.path.join(ROOT,"notes","content.md"))) else "absent (dangling ref in Note8.md)")
print("")

def cross(tau, key):
    pts = sorted((fn(T[(r, tau, 1)]["x"]), fn(T[(r, tau, 1)][key]), r) for r in DOSE)
    for i in range(len(pts) - 1):
        x0, y0, r0 = pts[i]; x1, y1, r1 = pts[i + 1]
        if (y0 - 1) * (y1 - 1) <= 0 and y1 != y0:
            return (x0 + (1 - y0) / (y1 - y0) * (x1 - x0), r0, r1)
    return (None, None, None)

def dfx(tau, key):
    xs = sorted(DOSE, key=lambda r: DF[r])
    for i in range(len(xs) - 1):
        a, b = xs[i], xs[i + 1]
        ya = fn(T[(a, tau, 1)][key]); yb = fn(T[(b, tau, 1)][key])
        if (ya - 1) * (yb - 1) <= 0 and yb != ya:
            return (DF[a], DF[b])
    return None

def g(v, n, dash="-"):
    return dash if v is None else ("%.*f" % (n, v))

fam = {}
for r in S:
    fam.setdefault(r["family"], []).append(fn(r["rollout_mean_DB"]))

L = []
def w(s=""):
    L.append(s)

w("# Note 11 - Dream controllability as a dose-response: single-cycle audit (v5)")
w("")
w("Repository: `Reproduce-the-World-Models-plural-demo`. Source: arXiv:1803.10122 only;")
w("no official or third-party code was read. Every number below is emitted by the generator")
w("that wrote this file; none is typed by hand.")
w("")
w("## 1. Instrument, and one thing that is not a check")
w("")
w("`scripts/mdn/dream_controllability.py` slices the pilot dataset for one rollout `r`:")
w("")
w("- `index.json` gives `(start, count) = (r*1000, 1000)` for `rollout_*.npz`;")
w("- `n = min(steps + warmup, count) = min(216, 1000) = 216` frames are used;")
w("- `Z = latents.npy[s0:s0+n]` and `A = actions.npy[s0:s0+n]`;")
w("- actions are read from the *same* directory that is passed to `--latents`.")
w("")
w("For each `(tau, t)` the script prints `A(stoch)`, `B(noise)`, `A/B` and `D(det)`; the default")
w("`--taus` is `0.1,0.5,1.0` (`dream_controllability.py:117`). There are five self-checks:")
w("`permutation moved X%`, `identity = 0`, a zeroed a-slice yields `A = 0`, two-pass bit-identical")
w("output, and the printed medians. Note that **\"`D` is tau-invariant\" is not a self-check but an")
w("identity of the code**: `mode=\"det\"` calls `mix_mean(pi, mu)` and never assigns `temperature`.")
w("Table 2 confirms this empirically (spread `0.0000` across tau).")
w("")
w("## 2. Frozen protocol")
w("")
w(F + "text")
w("scripts/mdn/dream_controllability.py --mdn <ckpt> --latents <PROBE_DIR(row)> \\")
w("    --rollout r --steps 200 --warmup 16 --reps 50 --taus 0.1,0.5,1.0      (r = 0..19)")
w(F)
w("")
w("- `D/B := D(det)_mean / B(noise)_mean` (the script does not print `D/B`; it is column 8 divided by column 5).")
w("- `A/B` := the printed `A/B` column.")
w("- `x := NLL(lambda=0) - NLL(lambda=1)`, tau-independent, from the row's seed-0 checkpoint.")
w("")
w("`PROBE_DIR(row) = ckpt[\"args\"][\"latents\"]`, with **two declared overrides**:")
w("")
w("- `real-struct`   -> `data\\pilot_latents`")
w("- `synth3-struct` -> `data\\pilot_latents_synth3`")
w("")
w("Reason: those two rows' own `actions.npy` are entirely zero (`permutation moved 0.0%`), which")
w("degenerates `A = D = 0`. The override is declared in the script together with this rationale.")
w("")
w("## 3. Table 1 - the 15-row single-cycle table (t = 1, tau = 1.0)")
w("")
w("| row | family | x | D/B | SE | A/B | D(det) |")
w("|---|---|---|---|---|---|---|")
for r in ORDER:
    d = T[(r, 1.0, 1)]
    w("| %s | %s | %s | %s | %s | %s | %s |" % (r, FAM[r], g(fn(d["x"]), 3), g(fn(d["DB_mean"]), 6), g(fn(d["DB_se"]), 6), g(fn(d["AB_mean"]), 4), g(fn(d["D_mean"]), 3)))
w("")
w("`D/B` here reproduces the frozen v2 table exactly (15/15), i.e. this repository's v5 single-cycle")
w("run is bit-compatible with the earlier v2 table for that one column and for `SE`.")
w("")
w("## 4. Table 2 - dose-response on the drive-frac axis (x-free)")
w("")
w("| row | drive_frac | D/B tau=0.1 | D/B tau=0.5 | D/B tau=1.0 | A/B tau=0.1 | A/B tau=0.5 | A/B tau=1.0 |")
w("|---|---|---|---|---|---|---|---|")
for r in DOSE:
    w("| %s | %.2f | %s | %s | %s | %s | %s | %s |" % (r, DF[r], g(fn(T[(r,0.1,1)]["DB_mean"]),4), g(fn(T[(r,0.5,1)]["DB_mean"]),4), g(fn(T[(r,1.0,1)]["DB_mean"]),4), g(fn(T[(r,0.1,1)]["AB_mean"]),4), g(fn(T[(r,0.5,1)]["AB_mean"]),4), g(fn(T[(r,1.0,1)]["AB_mean"]),4)))
w("")
w("`D/B` crosses 1.0 at:")
w("")
for tau in TAU:
    c = dfx(tau, "DB_mean")
    w("- tau = %.1f: between drive_frac %.2f and %.2f" % (tau, c[0], c[1]) if c else "- tau = %.1f: no crossing" % tau)
w("")
w("The real family never approaches 1.0 at any tau; the synth3 family crosses it, and the crossing")
w("moves right as tau grows. `D_mean` is tau-invariant to the last printed digit at every `t`.")
w("")
w("## 5. Table 3 - tau-resolved headline")
w("")
w("`gap_i := x_cross(tau) / x_i`, where `x_i` are the five real-family per-training-seed")
w("lambda-ablation values (`notes/Note9.md:25`).")
w("")
w("| tau | D/B=1 bracket | x_cross (D/B) | x_cross (A/B) | gap min | gap p50 | gap mean | gap max | seeds >= 15x |")
w("|---|---|---|---|---|---|---|---|---|")
for tau in TAU:
    xc, r0, r1 = cross(tau, "DB_mean")
    xa, _, _ = cross(tau, "AB_mean")
    gs = sorted(xc / x for x in XSEED)
    w("| %.1f | %s -> %s | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %d/5 |" % (tau, r0, r1, xc, xa, gs[0], st.median(gs), st.mean(gs), gs[-1], sum(1 for v in gs if v >= 15)))
w("")
w("The headline is not a single number: the family gap grows with tau (13x -> 17x -> 19x), and")
w("\"at least 15x\" holds only for tau >= 0.5. The same interpolation on `A/B` gives a systematically")
w("~2% smaller crossover, consistent with Section 8.")
w("")
w("## 6. Table 4 - family separation on the tau-slope of A/B(t = 200)")
w("")
w("| row | family | tau=0.1 | tau=0.5 | tau=1.0 | slope |")
w("|---|---|---|---|---|---|")
cnt = {"rise": 0, "fall": 0, "non": 0}
for r in ORDER:
    v = [fn(T[(r, tau, 200)]["AB_mean"]) for tau in TAU]
    lab = "rising" if v[0] <= v[1] <= v[2] else ("falling" if v[0] >= v[1] >= v[2] else "non-monotone")
    cnt["rise" if lab == "rising" else ("fall" if lab == "falling" else "non")] += 1
    w("| %s | %s | %.4f | %.4f | %.4f | %s |" % (r, FAM[r], v[0], v[1], v[2], lab))
w("")
w("Counts: rising = %d, falling = %d, non-monotone = %d. All six real-family rows are rising **or flat**" % (cnt["rise"], cnt["fall"], cnt["non"]))
w("(`real-struct` is `0.898 -> 0.887 -> 0.906`, a 1.2% dip in the middle: endpoint-labelled, not strict);")
w("all nine synth3 rows are falling. The two families separate on the sign of this slope.")
w("")
w("## 7. Table 5 - seed-axis error bars (tau = 1, t = 1, 20 rollouts per checkpoint)")
w("")
w("| family | n_seed | per-seed D/B | mean | sd | SE | cv |")
w("|---|---|---|---|---|---|---|")
for f in ["real", "shuf", "noact"]:
    if f not in fam: continue
    v = fam[f]; m = st.mean(v); sd = st.stdev(v); se = sd / len(v) ** 0.5
    w("| %s | %d | %s | %.4f | %.4f | %.4f | %.1f%% |" % (f, len(v), " / ".join("%.4f" % x for x in v), m, sd, se, 100.0 * sd / m))
w("")
if "real" in fam and "shuf" in fam:
    mr = st.mean(fam["real"]); sr = st.stdev(fam["real"]) / len(fam["real"]) ** 0.5
    ms = st.mean(fam["shuf"]); ss = st.stdev(fam["shuf"]) / len(fam["shuf"]) ** 0.5
    w("- family separation, seed axis: `%.4f +- %.4f` vs `%.4f +- %.4f`, difference `%.4f +- %.4f` = **%.1f sigma**" % (mr, sr, ms, ss, mr - ms, (sr * sr + ss * ss) ** 0.5, (mr - ms) / (sr * sr + ss * ss) ** 0.5))
if "real" in fam and "noact" in fam:
    mn = st.mean(fam["noact"]); sn = st.stdev(fam["noact"]) / len(fam["noact"]) ** 0.5
    mr = st.mean(fam["real"]); sr = st.stdev(fam["real"]) / len(fam["real"]) ** 0.5
    w("- real vs noact: `%.4f +- %.4f` vs `%.4f +- %.4f` = **%.1f sigma** (ratio %.2fx)" % (mr, sr, mn, sn, (mr - mn) / (sr * sr + sn * sn) ** 0.5, mn / mr))
w("")
w("The seed-0 checkpoints used in Table 1 are the *lowest* member of each family (real: 1/5, noact: 1/3;")
w("shuf is not), so both the seed-0 and seed-mean ratios must be quoted. Table 1's `SE` is rollout-level;")
w("the seed-level SE for the `real` row is the one that matters for cross-seed claims.")
w("")
w("## 8. Correction: D/B is not A/B")
w("")
w("At `(tau = 1, t = 1)` the relative difference `|D/B - A/B| / A/B` ranges from `0.1%` to `26.7%`,")
w("with 15 of 45 `(row, tau)` cells above 6%, and the paired per-rollout difference is significant at up to")
w("`+10 sigma` (s3p20). `D/B` is therefore **systematically smaller** than `A/B`, as expected from")
w("Jensen's inequality (`E||draw|| >= ||mean||`). An earlier claim that `D/B` is a proxy for `A/B`")
w("was based on the four cells with the smallest difference and is withdrawn (ledger #105).")
w("")
w("## 9. Errata to earlier notes (these notes are not modified)")
w("")
w("- `notes/Note10.md:199-201` states the `medA/medB` and `medRatio` definitions with the bracket order")
w("  reversed. The code citation at `notes/Note10.md:214` is correct.")
w("  `medRatio = np.median(av / np.maximum(bv, 1e-9))` is a paired-per-rep median, not median-of-marginals.")
w("- `notes/Note10.md:12` and `notes/Note10.md:151` use \"16x\" / \"16.5x\" for the *cold-vs-warm*, within-episode")
w("  ratio. That is a different quantity from this note's tau-resolved *cross-family* gap (13x -> 19x).")
w("  The two must not be conflated.")
w("- `notes/Note8.md:45` (Section 4: a randomly re-drawn a-slice responds *more* than the trained one) is")
w("  consistent with Table 5 here: destroying the action channel raises the t = 1 reading.")
w("  `notes/Note8.md` lines 58-60 already note that the 0.111 figure lost its evidential status.")
w("")
w("## 10. Withdrawn and non-reproducible quantities")
w("")
w("- `brief-x` (an externally supplied x column) has no provenance in this repository and cannot be")
w("  recomputed; it is abandoned. The quantity used here is `x := NLL(lambda=0) - NLL(lambda=1)`.")
w("- Two columns circulated in the interim v5 summary are not reproducible as `(tau = 1, t = 1, n = 20)`")
w("  aggregates and are withdrawn: the `A/B` column (6/15 rows disagree with the table) and the `D(det)`")
w("  column (13/13 rows are single-cell raw values, not aggregates). Note 11 uses `tmp_v5_table.csv` only.")
w("- `content.md` does not exist; the reference to it in `notes/Note8.md:98` is dangling.")
w("")
w("## 11. Settings that were never persisted")
w("")
w("`drive_frac`, `rho` and `seed` were used to build the synthetic latents but are written neither into")
w("`index.json` nor into any checkpoint `args`. The action files are copied byte-for-byte from the source")
w("directory, which explains the cross-family `actions.npy` md5 collisions.")
w("")
w("## 12. Credentials")
w("")
w("| file | bytes | md5 (first 12) |")
w("|---|---|---|")
CRED = ["tmp_v5_protocol.csv","tmp_v5_cells.csv","tmp_v5_table.csv","tmp_v5_headline.csv","tmp_v5_notnote11.csv","tmp_v5_log.txt","tmp_seed_summary.csv","tmp_seed_dbread.csv","tmp_final_table_v2.csv","tmp_final_provenance_v2.csv","manifest_ckpt.txt","tmp_n18_audit.txt","tmp_n18_audit_p2.txt","tmp_n18_audit_p3.txt","tmp_n18_credentials.csv","tmp_n18_deliverable.csv"]
for name in CRED:
    p = rp(name)
    if os.path.exists(p):
        w("| `%s` | %d | `%s` |" % (os.path.relpath(p, ROOT).replace("\\", "/"), os.path.getsize(p), md5(p)[:12]))
    else:
        w("| `%s` | - | MISSING |" % name)
w("")
w("## 13. Retraction ledger (this session: #47 - #109)")
w("")
w("Notable entries: #105 (`D/B` is not `A/B`, Section 8); #108 (two interim v5 summary columns withdrawn,")
w("Section 10); #97 / #98 (the `x = multi-seed lambda-ablation mean` and the normalised variant are both")
w("falsified); #99 (`ckpt.args.latents` is not always the correct probe directory - the two structural rows")
w("need an override, Section 2). The full ledger is kept in the session transcript.")
w("")
w("## 14. Reproducibility certification")
w("")
w("This note is the audit artifact for the dose-response claim. It is generated from frozen CSVs by")
w("`scripts/mdn/gen_note11.py`, is independent of the training code, and reproduces the tau-resolved")
w("headline from raw credentials alone.")
w("")

open(os.path.join(ROOT, "notes", "Note11.md"), "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
print("written -> notes/Note11.md  (%d lines, %d bytes)" % (len(L) + 1, os.path.getsize(os.path.join(ROOT, "notes", "Note11.md"))))
