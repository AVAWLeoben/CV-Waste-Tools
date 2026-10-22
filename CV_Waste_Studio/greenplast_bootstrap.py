# -*- coding: utf-8 -*-
"""Linux CLI preflight: select/create a working Conda environment, then open GUI.

Intentionally imports no non-standard modules or tkinter from Conda base.
"""
from __future__ import annotations
import os
from pathlib import Path
import subprocess
import sys
import traceback
from greenplast_environment import EnvironmentManager, env_python, process_environment

APP_DIR = Path(__file__).resolve().parent


def report(kind, value):
    if kind == "detail":
        print(value, flush=True)
    elif kind in ("log", "status"):
        print("[GreenPlast] " + str(value), flush=True)


def main():
    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        print("WARNING: No DISPLAY/WAYLAND_DISPLAY detected. A graphical Linux "
              "desktop (X11/XWayland) is normally needed.", flush=True)
    conda, env, check = EnvironmentManager(report).setup()
    print(f"[GreenPlast] Ready: {env}", flush=True)
    print(f"[GreenPlast] CUDA accessible: {check.get('cuda', False)}", flush=True)
    args = [str(env_python(env)), "-u", str(APP_DIR / "greenplast_launcher.py"),
            "--env-prefix", str(env), "--conda-exe", conda]
    return subprocess.call(args, cwd=str(APP_DIR), env=process_environment(env))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
