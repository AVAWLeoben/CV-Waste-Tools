"""Windows-safe native YOLO tuner entry point. No GUI or CUDA at import time."""
from __future__ import annotations
import json
import multiprocessing
from pathlib import Path
import sys
import traceback


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        raise SystemExit('Usage: run_hyperparameter_tuning.py settings.json')
    from hyperparameter_tuning import validate_settings, run_native_tuning
    config = json.loads(Path(argv[0]).read_text(encoding='utf-8'))
    run_native_tuning(validate_settings(config))


if __name__ == '__main__':
    multiprocessing.freeze_support()
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
