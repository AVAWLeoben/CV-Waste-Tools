"""Windows-spawn-safe, headless native Ultralytics model export worker."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("settings_json", type=Path)
    args = parser.parse_args(argv)
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ['YOLO_AUTOINSTALL'] = 'False'  # no pip/apt changes to JetPack or SenSoRTC
    settings = json.loads(args.settings_json.read_text(encoding="utf-8"))
    if sys.platform == "win32" and settings.get("format") == "litert":
        from wsl_litert import run_wsl_litert_export
        run_wsl_litert_export(settings)
    else:
        from model_export import run_export_job
        run_export_job(settings)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
