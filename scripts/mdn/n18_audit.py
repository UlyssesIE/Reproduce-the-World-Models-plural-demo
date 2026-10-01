"""N18 merged audit runner: executes the four frozen stages from the repository root."""
import os, runpy
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
for i in (1, 2, 3, 4):
    s = "scripts/mdn/audit/n18_stage%d.py" % i
    print("=" * 96); print("STAGE", s); print("=" * 96)
    runpy.run_path(s, run_name="__main__")
