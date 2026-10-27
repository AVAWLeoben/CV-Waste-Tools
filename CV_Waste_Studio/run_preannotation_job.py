# -*- coding: utf-8 -*-
"""Run pre-annotation in the verified training environment, outside Tkinter."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import traceback


def main(argv=None):
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        raise SystemExit("Usage: python run_preannotation_job.py <settings.json>")
    settings = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise ValueError("Pre-annotation settings must be a JSON object")
    from preannotate_dataset import preannotate_dataset
    print("CV Waste Studio — pre-annotation job", flush=True)
    preannotate_dataset(**settings)
    print("JOB COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.exit(1)
