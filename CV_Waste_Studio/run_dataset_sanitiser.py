#!/usr/bin/env python3
"""Headless dataset-sanitisation worker (no Tk imports)."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import traceback


def main(args=None):
    args = sys.argv[1:] if args is None else args
    if len(args) != 1:
        raise ValueError("Usage: python run_dataset_sanitiser.py <job.json>")
    job = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    if job.get("phase") == "scan":
        from dataset_sanitiser import scan_dataset
        scan_dataset(job["settings"], job["plan"])
    elif job.get("phase") == "apply":
        from dataset_sanitiser import apply_plan
        apply_plan(job["plan"])
    else:
        raise ValueError("Unknown sanitisation phase; expected scan or apply.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
