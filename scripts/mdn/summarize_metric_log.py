"""Summarise repeated metric_validation runs.

    python scripts/mdn/summarize_metric_log.py runs/mv_areinit5.log
"""
import statistics as st
import sys
from collections import defaultdict

rows = defaultdict(list)
for line in open(sys.argv[1], encoding="utf-16", errors="ignore"):
    if "|D|bar" not in line or "NEW" not in line or "OLD" not in line:
        continue                                # skip verdict/sanity/etc.
    tag = line.split("|D|bar")[0].strip()       # 'trained' / 'a-reinit' / 'random-init'
    t = line.split()
    try:
        new = float(t[t.index("NEW") + 1])
        old = float(t[t.index("OLD") + 1])
        wa = float(t[t.index("||Wa||") + 1]) if "||Wa||" in t else float("nan")
    except (ValueError, IndexError):
        continue
    rows[tag].append((new, old, wa))

print(f"{'condition':<13}{'n':>3}  {'NEW':>15}  {'NEW/||Wa||':>17}  {'OLD':>21}  {'cv(OLD)':>8}")
for tag, r in rows.items():
    new = [x[0] for x in r]; old = [x[1] for x in r]; wa = [x[2] for x in r]
    g = [n / w for n, w in zip(new, wa) if w > 0]
    print(f"{tag:<13}{len(r):>3}  {st.mean(new):7.3f}±{st.pstdev(new):5.3f}  "
          f"{st.mean(g):7.3f}±{st.pstdev(g):5.3f}  "
          f"{st.mean(old):.6e}±{st.pstdev(old):.1e}  "
          f"{st.pstdev(old)/max(st.mean(old),1e-12):7.1%}")
