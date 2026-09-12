#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
check_env.py — environment sanity check for the World Models (2018) reproduction.

Usage:
    conda activate worldmodels
    python check_env.py                 # full check (creates CarRacing + VizDoom)
    python check_env.py --skip-env      # skip creating environments
    python check_env.py --no-color      # plain output (for logs / CI)

Exit code: 0 if no FAIL, 1 otherwise.
"""

import argparse
import glob
import importlib
import os
import platform
import shutil
import subprocess
import sys

try:
    import importlib.metadata as md
except ImportError:                                    # py<3.8
    md = None

OK, WARN, FAIL, SKIP = "OK", "WARN", "FAIL", "SKIP"

COLOR = {
    OK:   "\033[92m",
    WARN: "\033[93m",
    FAIL: "\033[91m",
    SKIP: "\033[90m",
    "BOLD": "\033[1m",
    "END":  "\033[0m",
}

ROWS = []          # (section, status, name, detail)
USE_COLOR = True


def paint(status, text):
    if not USE_COLOR:
        return text
    return "{}{}{}".format(COLOR.get(status, ""), text, COLOR["END"])


def rec(section, status, name, detail=""):
    ROWS.append((section, status, name, str(detail)))
    return status


# --------------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------------- #
def check_runtime():
    v = sys.version_info
    detail = "{} ({} {})".format(platform.python_version(), platform.system(), platform.machine())
    if v[:2] == (3, 11):
        rec("Runtime", OK, "Python", detail)
    elif v[:2] == (3, 10):
        rec("Runtime", OK, "Python", detail + " -- 3.10/3.11 both fine for vizdoom")
    elif v[:2] >= (3, 12):
        rec("Runtime", FAIL, "Python",
            detail + " -- vizdoom has NO 3.12 wheels; rebuild the env with python=3.11")
    else:
        rec("Runtime", WARN, "Python", detail + " -- older than the recommended 3.11")

    rec("Runtime", OK, "Executable", sys.executable)

    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    prefix = os.environ.get("CONDA_PREFIX")
    if conda_env:
        rec("Runtime", OK, "Conda env", "{} ({})".format(conda_env, prefix))
    else:
        rec("Runtime", WARN, "Conda env", "no active env detected (CONDA_DEFAULT_ENV unset)")

    cwd = os.getcwd()
    if " " in cwd:
        rec("Runtime", WARN, "Working dir",
            "'{}' -- spaces break CMake/SWIG when building C++ ext. Now looks fine.".format(cwd))
    else:
        rec("Runtime", OK, "Working dir", cwd)


# --------------------------------------------------------------------------- #
# Packages
# --------------------------------------------------------------------------- #
# (import name, distribution name, required?)
PACKAGES = [
    ("torch",           "torch",                True),
    ("torchvision",     "torchvision",          True),
    ("numpy",           "numpy",                True),
    ("scipy",           "scipy",                True),
    ("gymnasium",       "gymnasium",            True),
    ("Box2D",           "box2d-py",             True),
    ("pygame",          "pygame",               True),
    ("vizdoom",         "vizdoom",              True),
    ("cv2",             "opencv-python-headless", True),
    ("PIL",             "Pillow",               True),
    ("cma",             "cma",                  True),
    ("yaml",            "PyYAML",               True),
    ("matplotlib",      "matplotlib",           True),
    ("pandas",          "pandas",               True),
    ("tqdm",            "tqdm",                 True),
    ("tensorboard",     "tensorboard",          True),
    ("imageio",         "imageio",              True),
    ("imageio_ffmpeg",  "imageio-ffmpeg",       True),
    ("pytest",          "pytest",               True),
    ("psutil",          "psutil",               False),
]


def check_packages():
    for import_name, dist_name, required in PACKAGES:
        try:
            mod = importlib.import_module(import_name)
        except Exception as exc:
            rec("Packages", FAIL if required else WARN, dist_name,
                "import '{}' failed: {}: {}".format(import_name, type(exc).__name__, exc))
            continue

        ver = None
        if md is not None:
            try:
                ver = md.version(dist_name)
            except Exception:
                ver = None
        if ver is None:
            ver = getattr(mod, "__version__", None)

        detail = "version {}".format(ver) if ver else "installed (version unknown)"
        if not required:
            detail += "  [optional]"
        rec("Packages", OK, dist_name, detail)


# --------------------------------------------------------------------------- #
# GPU / CUDA
# --------------------------------------------------------------------------- #
def sep(s):
    return s.replace("\n", " | ").strip()


def check_gpu():
    try:
        import torch
    except Exception:
        rec("GPU / CUDA", FAIL, "torch CUDA", "torch not importable -- see Packages")
        return

    if torch.version.cuda is None:
        rec("GPU / CUDA", FAIL, "torch build",
            "CPU-only wheel (torch.version.cuda is None). Reinstall the +cu121 wheel.")
        return

    rec("GPU / CUDA", OK, "torch build",
        "torch {} + cu{}".format(torch.__version__, torch.version.cuda))

    if not torch.cuda.is_available():
        rec("GPU / CUDA", FAIL, "cuda.is_available()",
            "False -- GPU wheel installed but CUDA not visible (driver? headless?).")
        return

    rec("GPU / CUDA", OK, "cuda.is_available()", "True")
    rec("GPU / CUDA", OK, "device count", torch.cuda.device_count())

    try:
        props = torch.cuda.get_device_properties(0)
        cap = torch.cuda.get_device_capability(0)
        total_gb = props.total_memory / (1024 ** 3)
        rec("GPU / CUDA", OK, "device 0",
            "{} | {:.1f} GB | sm_{}{}".format(props.name, total_gb, cap[0], cap[1]))
        if "3060" in props.name and total_gb < 6.5:
            rec("GPU / CUDA", OK, "VRAM budget",
                "{:.1f} GB usable -- plenty for VAE + 256-d GRU".format(total_gb))
    except Exception as exc:
        rec("GPU / CUDA", WARN, "device properties", str(exc))

    try:
        cudnn = torch.backends.cudnn
        rec("GPU / CUDA", OK, "cudnn",
            "available={} version={}".format(cudnn.is_available(),
                                              getattr(cudnn, "version", lambda: None)()))
    except Exception as exc:
        rec("GPU / CUDA", WARN, "cudnn", str(exc))

    # bf16 support matters for the 6 GB memory tricks
    try:
        rec("GPU / CUDA", OK, "bf16 supported", torch.cuda.is_bf16_supported())
    except Exception:
        pass

    exe = shutil.which("nvidia-smi")
    if not exe:
        rec("GPU / CUDA", WARN, "nvidia-smi", "not on PATH (not fatal)")
    else:
        try:
            out = subprocess.run(
                [exe, "--query-gpu=name,driver_version,memory.total",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=20)
            rec("GPU / CUDA", OK, "nvidia-smi", sep(out.stdout))
        except Exception as exc:
            rec("GPU / CUDA", WARN, "nvidia-smi", str(exc))


# --------------------------------------------------------------------------- #
# Build toolchain (only needed if box2d-py must be compiled from source)
# --------------------------------------------------------------------------- #
def check_toolchain():
    exe = shutil.which("swig")
    if not exe:
        rec("Build toolchain", WARN, "swig",
            "not on PATH -- run `pip install swig` inside the env")
    else:
        try:
            out = subprocess.run([exe, "-version"], capture_output=True,
                                 text=True, timeout=20)
            first = (out.stdout or out.stderr).splitlines()[0] if (out.stdout or out.stderr) else "?"
            rec("Build toolchain", OK, "swig", "{}  ({})".format(first, exe))
        except Exception as exc:
            rec("Build toolchain", WARN, "swig", str(exc))

    # MSVC: only needed for source builds of box2d-py
    cl = shutil.which("cl.exe")
    if cl:
        rec("Build toolchain", OK, "MSVC cl.exe", cl)
        return

    pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = os.path.join(pf86, "Microsoft Visual Studio", "Installer", "vswhere.exe")
    if os.path.isfile(vswhere):
        try:
            out = subprocess.run(
                [vswhere, "-latest", "-products", "*",
                 "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                 "-property", "displayName"],
                capture_output=True, text=True, timeout=30)
            name = out.stdout.strip()
            if name:
                rec("Build toolchain", OK, "MSVC (vswhere)",
                    "{} -- cl.exe just isn't on PATH (that's normal)".format(sep(name)))
                return
        except Exception:
            pass
    rec("Build toolchain", WARN, "MSVC / VS Build Tools",
        "not found -- only required IF `pip install box2d-py` tries to build from source")

    ffi = shutil.which("ffmpeg")
    if ffi:
        rec("Build toolchain", OK, "ffmpeg (system)", ffi)
    else:
        rec("Build toolchain", OK, "ffmpeg", "not on PATH -- imageio-ffmpeg bundles its own")


# --------------------------------------------------------------------------- #
# Environments
# --------------------------------------------------------------------------- #
def check_carracing():
    try:
        import gymnasium as gym
    except Exception as exc:
        rec("Environments", FAIL, "CarRacing-v3", "gymnasium import failed: {}".format(exc))
        return
    env = None
    try:
        env = gym.make("CarRacing-v3", continuous=True)
        obs, _ = env.reset(seed=0)
        for _ in range(3):
            obs, r, term, trunc, info = env.step(env.action_space.sample())
        rec("Environments", OK, "CarRacing-v3",
            "obs={} {} range=[{}, {}] | action_space={}".format(
                obs.shape, obs.dtype, obs.min(), obs.max(), env.action_space))
    except Exception as exc:
        rec("Environments", FAIL, "CarRacing-v3",
            "make/step failed: {}: {}".format(type(exc).__name__, exc))
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def check_vizdoom():
    try:
        import vizdoom
    except Exception as exc:
        rec("Environments", FAIL, "vizdoom",
            "import failed: {}: {}".format(type(exc).__name__, exc))
        return

    d = os.path.dirname(os.path.abspath(vizdoom.__file__))
    rec("Environments", OK, "vizdoom package dir", d)

    scen = os.path.join(d, "scenarios")
    if os.path.isdir(scen):
        cfgs = sorted(f for f in os.listdir(scen) if f.endswith(".cfg"))
        rec("Environments", OK if cfgs else WARN, "vizdoom scenarios",
            "{} .cfg file(s): {}".format(len(cfgs), ", ".join(cfgs[:4]) or "(none)"))
        if not any("takecover" in c.lower() for c in cfgs):
            rec("Environments", WARN, "DoomTakeCover.cfg",
                "not shipped -- you must write it yourself for the paper's 2nd experiment")
    else:
        rec("Environments", WARN, "vizdoom scenarios", "no scenarios/ dir under package")

    wads = glob.glob(os.path.join(d, "**", "*.wad"), recursive=True)
    if wads:
        rec("Environments", OK, "vizdoom .wad",
            "{} found, e.g. {}".format(len(wads), os.path.basename(wads[0])))
    else:
        rec("Environments", WARN, "vizdoom .wad",
            "none found -- supply freedoom2.wad yourself")


def check_sdl():
    need = {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"}
    vals = {k: os.environ.get(k) for k in need}
    if platform.system() == "Windows":
        if any(vals.values()):
            rec("Environments", OK, "SDL env vars", str(vals))
        else:
            rec("Environments", OK, "SDL env vars",
                "unset -- fine on a Windows desktop with a display")
    else:
        missing = [k for k, v in vals.items() if not v]
        if missing:
            rec("Environments", WARN, "SDL env vars",
                "missing {} -- headless rendering may crash".format(", ".join(missing)))
        else:
            rec("Environments", OK, "SDL env vars", str(vals))


# --------------------------------------------------------------------------- #
# System
# --------------------------------------------------------------------------- #
def check_system():
    try:
        free = shutil.disk_usage(os.getcwd()).free / (1024 ** 3)
        rec("System", OK if free > 20 else WARN, "Disk free",
            "{:.1f} GB on current drive".format(free))
    except Exception as exc:
        rec("System", WARN, "Disk free", str(exc))

    try:
        import psutil
        vm = psutil.virtual_memory()
        total_gb = vm.total / (1024 ** 3)
        avail_gb = vm.available / (1024 ** 3)
        status = OK if avail_gb > 16 else WARN
        rec("System", status, "RAM",
            "{:.1f} GB total / {:.1f} GB available -- replay buffer lives here".format(
                total_gb, avail_gb))
    except Exception:
        rec("System", SKIP, "RAM", "install psutil for this check")


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
SECTION_ORDER = ["Runtime", "Packages", "GPU / CUDA", "Build toolchain",
                 "Environments", "System"]


def print_report():
    print()
    print("=" * 78)
    print("  World Models (2018) reproduction -- environment check")
    print("  python {} | {}".format(platform.python_version(), platform.platform()))
    print("=" * 78)

    for sec in SECTION_ORDER:
        rows = [r for r in ROWS if r[0] == sec]
        if not rows:
            continue
        print("\n{}[{}]{}".format(COLOR["BOLD"], sec, COLOR["END"]) if USE_COLOR
              else "\n[{}]".format(sec))
        width = max(len(r[2]) for r in rows)
        for _, status, name, detail in rows:
            print("  {} {:<{w}}  {}".format(paint(status, "[{:<4}]".format(status)),
                                            name, detail, w=width + 6))

    counts = {s: sum(1 for r in ROWS if r[1] == s) for s in (OK, WARN, FAIL, SKIP)}
    print("\n" + "-" * 78)
    print("  Summary: {} OK | {} WARN | {} FAIL | {} SKIP".format(
        counts[OK], counts[WARN], counts[FAIL], counts[SKIP]))

    fails = [r for r in ROWS if r[1] == FAIL]
    if fails:
        print("\n  Blocking issues:")
        for _, _, name, detail in fails:
            print("    - {}: {}".format(name, detail))
    else:
        print("\n  No blocking issues. You are good to go.")

    warns = [r for r in ROWS if r[1] == WARN]
    if warns:
        print("\n  Non-blocking warnings:")
        for _, _, name, detail in warns:
            print("    - {}: {}".format(name, detail))

    print("-" * 78 + "\n")
    return 1 if fails else 0


def main():
    global USE_COLOR
    p = argparse.ArgumentParser(description="World Models env checker")
    p.add_argument("--skip-env", action="store_true",
                   help="skip creating CarRacing / VizDoom environments")
    p.add_argument("--no-color", action="store_true", help="plain output")
    args = p.parse_args()

    USE_COLOR = not args.no_color and sys.stdout.isatty()
    if os.name == "nt" and USE_COLOR:
        os.system("")          # enable ANSI escapes in old conhost

    check_runtime()
    check_packages()
    check_gpu()
    check_toolchain()
    if not args.skip_env:
        check_carracing()
        check_vizdoom()
        check_sdl()
    else:
        rec("Environments", SKIP, "CarRacing-v3", "--skip-env")
        rec("Environments", SKIP, "vizdoom", "--skip-env")
        rec("Environments", SKIP, "SDL env vars", "--skip-env")
    check_system()

    sys.exit(print_report())


if __name__ == "__main__":
    main()
