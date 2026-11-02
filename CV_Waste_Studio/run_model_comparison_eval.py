"""Isolated test-split evaluation: exit the process to release CUDA allocations."""
from __future__ import annotations
import gc
import json
import multiprocessing
import os
from pathlib import Path
import sys
import traceback


def evaluate(config):
    # Scope the existing studio's OpenMP workaround to this inference worker.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    from ultralytics import YOLO
    import torch
    model = None
    try:
        model = YOLO(config["weights"])
        device = config.get("device", "auto")
        if device in ("gpu", "auto"):
            device = 0 if torch.cuda.is_available() else "cpu"
        split = config.get("split", "test")
        if split not in ("val", "test"):
            raise ValueError("Only val or test splits are permitted in model comparison.")
        print(f"Evaluating best.pt on {split.upper()} split: {config['dataset_yaml']}", flush=True)
        results = model.val(data=config["dataset_yaml"], split=split, imgsz=config["imgsz"],
                            batch=config["batch"], device=device, workers=config["workers"],
                            project=config["project"], name=config["name"], plots=True)
        box = results.box
        metrics = {"precision": float(box.mp), "recall": float(box.mr),
                   "map50": float(box.map50), "map50_95": float(box.map),
                   "save_dir": str(results.save_dir)}
        destination = Path(config["result_json"])
        destination.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print("Test-set metrics: " + json.dumps(metrics), flush=True)
        return metrics
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except RuntimeError:
                pass  # Process exit still frees the CUDA context.


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: run_model_comparison_eval.py settings.json")
    evaluate(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.exit(1)
