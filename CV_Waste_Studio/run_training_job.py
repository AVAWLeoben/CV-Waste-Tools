# -*- coding: utf-8 -*-
"""Headless split/train/existing-dataset worker for the unified desktop UI.

Running YOLO outside the Tk event loop keeps the same backend implementation,
retains the entire output in a file and keeps the interface responsive.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import traceback


def main(argv=None):
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        raise SystemExit("Usage: python run_training_job.py <settings.json>")
    config = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Training job settings must be a JSON object")
    from train_and_split_tool_GUI import autosplit_and_train_model
    print("CV Waste Studio — dataset/training job", flush=True)
    autosplit_and_train_model(**config)
    print("JOB COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.exit(1)
