#!/usr/bin/env python3
"""Headless Recall@Precision worker for TensorRT engines in an external Python.

The selected Python is the same one used for export and SenSoRTC deployment.
No package installation, environment activation, or GUI setup takes place here.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback

os.environ["YOLO_AUTOINSTALL"] = "False"
os.environ.setdefault("MPLBACKEND", "Agg")


def _event(kind, value):
    print("STUDIO_RECALL_EVENT=" + json.dumps({"kind": kind, "value": value},
                                             ensure_ascii=False, default=str), flush=True)


class FileCancellation:
    def __init__(self, marker: Path):
        self.marker = marker

    def is_set(self):
        return self.marker.is_file()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a TensorRT engine in the selected runtime")
    parser.add_argument("settings", type=Path)
    parser.add_argument("--cancel-file", type=Path, required=True)
    opts = parser.parse_args(argv)
    settings = json.loads(opts.settings.read_text(encoding="utf-8"))
    from test_model_recall_at_precision import evaluate, EvaluationCancelled
    if Path(settings["model"]).suffix.lower() != ".engine":
        raise ValueError("The external inference worker only accepts .engine files")
    _event("log", "Inference worker: " + sys.executable)
    try:
        evaluate(settings, _event, cancel_event=FileCancellation(opts.cancel_file))
    except EvaluationCancelled as exc:
        _event("cancelled", str(exc))
        return 2
    except Exception as exc:
        _event("error", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
