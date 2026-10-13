"""GATE-1 freeze: untrained-channel + norm-pairing evidence for Note11 sec. 7.

Recomputes, from the raw per-rollout stdout already on disk, every number the paper
quotes in par. 4 / sec. 6 that previously lived only in a working session, and writes
ONE durable artifact. No training, no probe re-run needed (caches exist for all cells);
--allow-run regenerates a missing cache via scripts/mdn/dream_controllability.py.

Cells   family     probe latents              ||Wa||     role
        real       data/pilot_latents          ~5.242     info+
        shuf       data/pilot_latents_shuf     ~1.118     info-
        shufup     data/pilot_latents_shuf     5.242      info-  (norm-pinned up)
        realdown   data/pilot_latents          1.118      info+  (norm-pinned down)
        noact      data/pilot_latents (base)   1.732      never trained (artefact)

Metric (t=1, tau=1): D/B = mean over rollouts of the per-rollout ratio D_i / B_i.
Cross-seed SE = sd(per-ckpt means) / sqrt(n_ckpt).

Reads  <repo>/tmp/repo_root_backup/{tmp_ctrl,tmp_seed}/*.txt   (first hit wins;
       also checks <repo>/tmp_ctrl, <repo>/tmp_seed)
Writes <repo>/notes/artifacts/tmp_n24_gate1_2x2.csv
       <repo>/notes/artifacts/tmp_n24_gate1_values.csv

Run    python scripts/mdn/n24_gate1_2x2.py
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import statistics as st
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "notes" / "artifacts"
TMP = ROOT.parent / "tmp"                     # tmp/ is a SIBLING of the repo, not inside it
CACHE_DIRS = [TMP / "repo_root_backup" / "tmp_ctrl",
              TMP / "repo_root_backup" / "tmp_seed",
              TMP / "tmp_ctrl", TMP / "tmp_seed",
              ROOT / "tmp_ctrl", ROOT / "tmp_seed"]
SUMMARY = TMP / "repo_root_backup" / "tmp_ctrl_train" / "summary.csv"
MANIFEST = ART / "manifest_ckpt.txt"


REAL = "data/pilot_latents"
SHUF = "data/pilot_latents_shuf"
FAM = {
    "real":     (REAL, [("real", r"runs\mdn_pilot\best.pt")] +
                 [(f"mdn_pilot_s{i}", rf"runs\mdn_pilot_s{i}\best.pt") for i in (1, 2, 3, 4)]),
    "shuf":     (SHUF, [("shuf", r"runs\mdn_pilot_shuf\best.pt")] +
                 [(f"mdn_pilot_shuf_s{i}", rf"runs\mdn_pilot_shuf_s{i}\best.pt") for i in (1, 2, 3)]),
    "shufup":   (SHUF, [("shufup", r"runs\mdn_pilot_shufup\best.pt")] +
                 [(f"mdn_pilot_shufup_s{i}", rf"runs\mdn_pilot_shufup_s{i}\best.pt") for i in (1, 2, 3, 4, 5)]),
    "realdown": (REAL, [("realdown", r"runs\mdn_pilot_realdown\best.pt")] +
                 [(f"mdn_pilot_realdown_s{i}", rf"runs\mdn_pilot_realdown_s{i}\best.pt") for i in (1, 2, 3, 4, 5)]),
    "noact":    (REAL, [("mdn_pilot_noact", r"runs\mdn_pilot_noact\best.pt")] +
                 [(f"mdn_pilot_noact_s{i}", rf"runs\mdn_pilot_noact_s{i}\best.pt") for i in (1, 2)]),
}
ORDER = ["real", "shuf", "shufup", "realdown", "noact"]
NR = 20
PROTO = ["--steps", "200", "--warmup", "16", "--reps", "50", "--taus", "1.0"]
RE = re.compile(r"^\s*([\d.]+)\s+(\d+)\s+([\d.]+)\s*±\s*([\d.]+)\s+([\d.]+)"
                r"\s*±\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)")

REF = {"real": 0.089631, "shuf": 0.023117}      # frozen seed-0 means (Note11 Table 1 / provenance v2)
TOL = 5e-7


def find_cache(tag, ro):
    for d in CACHE_DIRS:
        f = d / ("%s_r%02d.txt" % (tag, ro))
        if f.exists() and f.stat().st_size > 0:
            return f
    return None


def parse_row(f):
    """(D/B, A, B, D) at (tau=1, t=1) from one raw stdout file; Nones if absent."""
    for ln in f.read_text(encoding="utf-8", errors="replace").splitlines():
        m = RE.match(ln)
        if m and abs(float(m.group(1)) - 1.0) < 1e-9 and int(m.group(2)) == 1:
            A, B, D = float(m.group(3)), float(m.group(5)), float(m.group(8))
            return (D / B if B else None), A, B, D
    return None, None, None, None


def run_one(tag, ck, lat, ro):
    f = CACHE_DIRS[0] / ("%s_r%02d.txt" % (tag, ro))
    CACHE_DIRS[0].mkdir(parents=True, exist_ok=True)
    env = dict(os.environ); env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                                        "PYTHONIOENCODING": "utf-8"})
    p = subprocess.run([sys.executable, str(ROOT / "scripts/mdn/dream_controllability.py"),
                        "--mdn", ck, "--latents", lat, "--rollout", str(ro)] + PROTO,
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env, cwd=str(ROOT))
    f.write_text((p.stdout or "") + (p.stderr or ""), encoding="utf-8")
    return f


def manifest_norm(name):
    if not MANIFEST.exists():
        return None
    for ln in MANIFEST.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"^(\S+)/best\.pt\s+\S+\s+Wa\s+([\d.]+)", ln)
        if m and m.group(1) == name:
            return float(m.group(2))
    return None


def pinned_norms():
    out = {}
    if SUMMARY.exists():
        for r in csv.DictReader(SUMMARY.open(encoding="utf-8")):
            out.setdefault(r["cell"], []).append(float(r["wa_norm"]))
    return {k: st.mean(v) for k, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-run", action="store_true",
                    help="regenerate a missing cache via dream_controllability.py")
    a = ap.parse_args()

    # ---- gather per-ckpt aggregates --------------------------------------
    per_ck, missing = {}, []
    for fam in ORDER:
        lat, cks = FAM[fam]
        for tag, ck in cks:
            rows = []
            for ro in range(NR):
                f = find_cache(tag, ro)
                if f is None:
                    if a.allow_run and os.path.exists(ROOT / ck):
                        f = run_one(tag, ck, lat, ro)
                    else:
                        break
                db, A, B, D = parse_row(f)
                if db is not None:
                    rows.append((db, A, B, D))
            if len(rows) == NR:
                per_ck[tag] = (fam, rows)
            else:
                missing.append((fam, tag, ck, len(rows)))

    print("cache coverage: %d checkpoints complete, %d incomplete" % (len(per_ck), len(missing)))
    for fam, tag, ck, n in missing:
        print("   INCOMPLETE %-9s %-22s %2d/%d  %s" % (fam, tag, n, NR, ck))

    # ---- cells ------------------------------------------------------------
    cells, tag2fam = {}, {}
    for tag, (fam, rows) in per_ck.items():
        tag2fam[tag] = fam
    for fam in ORDER:
        ms = [st.mean([r[0] for r in rows]) for t, (f2, rows) in per_ck.items() if f2 == fam]
        if not ms:
            continue
        mu = st.mean(ms); sd = st.stdev(ms) if len(ms) > 1 else 0.0
        se = sd / len(ms) ** 0.5 if len(ms) > 1 else 0.0
        cells[fam] = (mu, sd, se, len(ms))

    def ratio(x, y):
        return cells[x][0] / cells[y][0] if (x in cells and y in cells and cells[y][0]) else None

    def zz(x, y):
        if x in cells and y in cells:
            return abs(cells[x][0] - cells[y][0]) / (cells[x][2] ** 2 + cells[y][2] ** 2) ** 0.5
        return None

    contrasts = [("INFO@norm1.118", "realdown", "shuf"),
                 ("INFO@norm5.242", "real", "shufup"),
                 ("NORM@info-", "shufup", "shuf"),
                 ("NORM@info+", "real", "realdown"),
                 ("NOACT/real", "noact", "real")]

    # ---- self-check vs frozen seed-0 means --------------------------------
    print("\nself-check (seed-0 D/B vs frozen means):")
    for tag, ref in REF.items():
        if tag in per_ck:
            m = st.mean([r[0] for r in per_ck[tag][1]])
            print("   %-9s %.7f vs %.6f -> %s" % (tag, m, ref, "EXACT" if abs(m - ref) < TOL else "DIFF"))

    # ---- console table ----------------------------------------------------
    print("\n=== cells (t=1, tau=1; D/B mean of per-ckpt means) ===")
    for fam in ORDER:
        if fam in cells:
            mu, sd, se, n = cells[fam]
            print("   %-9s n=%d  D/B = %.4f +- %.4f (sd %.4f, se %.4f)" % (fam, n, mu, se, sd, se))

    print("\n=== 2x2 / contrasts ===")
    for name, x, y in contrasts:
        r, z = ratio(x, y), zz(x, y)
        print("   %-15s %-9s vs %-9s ratio %s  z %s"
              % (name, x, y, ("%.2fx" % r) if r else "NA", ("%.2f sigma" % z) if z else "NA"))

    # ---- scalar evidence table -------------------------------------------
    mB = {t: st.mean([r[2] for r in rows]) for t, (f2, rows) in per_ck.items()}
    mD = {t: st.mean([r[3] for r in rows]) for t, (f2, rows) in per_ck.items()}
    pin = pinned_norms()
    ev = []
    if "real" in cells:   ev.append(("EVID-3a", "real D/B (seed axis)", "%.4f" % cells["real"][0], "this artifact"))
    if "shuf" in cells:   ev.append(("EVID-3a", "shuf D/B (seed axis)", "%.4f" % cells["shuf"][0], "this artifact"))
    if "noact" in cells:  ev.append(("EVID-3a", "noact D/B (seed axis)", "%.4f" % cells["noact"][0], "this artifact"))
    ev += [("EVID-3b", "||Wa|| noact", "%.3f" % (manifest_norm("mdn_pilot_noact") or -1), "notes/artifacts/manifest_ckpt.txt:5"),
           ("EVID-3b", "||Wa|| real", "%.3f" % (manifest_norm("mdn_pilot") or -1), "notes/artifacts/manifest_ckpt.txt:1")]
    if "real" in mB:  ev.append(("EVID-3c", "mean B real (seed-0)", "%.3f" % mB["real"], "cache tmp_seed/mdn_pilot_r*.txt"))
    if "mdn_pilot_noact" in mB: ev.append(("EVID-3c", "mean B noact (seed-0)", "%.3f" % mB["mdn_pilot_noact"], "cache tmp_seed/mdn_pilot_noact_r*.txt"))
    if "real" in mD:  ev.append(("EVID-3d", "mean D real (seed-0)", "%.3f" % mD["real"], "tmp_v5_table.csv real (tau=1,t=1)"))
    if "mdn_pilot_noact" in mD: ev.append(("EVID-3d", "mean D noact (seed-0)", "%.3f" % mD["mdn_pilot_noact"], "tmp_v5_table.csv real-struct (tau=1,t=1)"))
    for cell, v in pin.items():
        ev.append(("EVID-3e", "pinned ||Wa|| %s" % cell, "%.4f" % v, "tmp_ctrl_train/summary.csv"))
    for name, x, y in contrasts:
        r, z = ratio(x, y), zz(x, y)
        ev.append(("EVID-3f", name, "%s / %s" % (("%.2fx" % r) if r else "NA", ("%.2f sigma" % z) if z else "NA"), "this artifact"))

    # ---- write artifacts --------------------------------------------------
    ART.mkdir(parents=True, exist_ok=True)
    f1 = ART / "tmp_n24_gate1_2x2.csv"
    with f1.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["kind", "label", "n_ckpt", "db_mean", "db_sd", "db_se", "ratio", "z"])
        for fam in ORDER:
            if fam in cells:
                mu, sd, se, n = cells[fam]
                w.writerow(["cell", fam, n, "%.6f" % mu, "%.6f" % sd, "%.6f" % se, "", ""])
        for name, x, y in contrasts:
            r, z = ratio(x, y), zz(x, y)
            w.writerow(["contrast", "%s = %s/%s" % (name, x, y), "", "", "", "",
                        "" if r is None else "%.4f" % r, "" if z is None else "%.4f" % z])
    f2 = ART / "tmp_n24_gate1_values.csv"
    with f2.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["evid", "quantity", "value", "source"]); w.writerows(ev)

    print("\nwrote %s" % f1.relative_to(ROOT))
    print("wrote %s" % f2.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
