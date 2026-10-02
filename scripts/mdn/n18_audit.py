"""N18 merged audit runner: hydrates frozen credentials from notes/artifacts/ then runs the four stages."""
import os, shutil, runpy
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
ART = os.path.join(ROOT, "notes", "artifacts")
if os.path.isdir(ART):
    for _f in sorted(os.listdir(ART)):
        _src = os.path.join(ART, _f); _dst = os.path.join(ROOT, _f)
        if os.path.isfile(_src) and not os.path.exists(_dst):
            shutil.copy2(_src, _dst)
for i in (1, 2, 3, 4):
    s = "scripts/mdn/audit/n18_stage%d.py" % i
    print("=" * 96); print("STAGE", s); print("=" * 96)
    runpy.run_path(s, run_name="__main__")
