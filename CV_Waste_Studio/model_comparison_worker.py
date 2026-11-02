"""Headless sequential YOLO model comparison.

This coordinator never imports PyTorch or Ultralytics. Every training and test
validation runs in its own Python child process, so CUDA allocations and image
caches die with the child before the next model is loaded. This is particularly
important for Windows spawn-based dataloaders.
"""
from __future__ import annotations

import csv
from datetime import datetime
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

from augmentation_options import validate_augmentation
from train_and_split_tool_GUI import _build_model_filename, _normalize_device_choice, resolve_existing_dataset_yaml
from training_log import parse_iteration_progress, parse_epoch_metrics, parse_training_run_dir

EVENT_PREFIX = "STUDIO_COMPARE_EVENT "
VALID_CACHE = ("off", "disk", "ram")
METRICS = ("precision", "recall", "map50", "map50_95")


def emit(kind, **data):
    print(EVENT_PREFIX + json.dumps({"kind": kind, **data}, allow_nan=False), flush=True)


def parse_event(line):
    if not line.startswith(EVENT_PREFIX):
        return None
    try:
        result = json.loads(line[len(EVENT_PREFIX):])
    except (TypeError, ValueError):
        return None
    return result if isinstance(result, dict) and "kind" in result else None


def _positive_int(value, label, minimum=1):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer >= {minimum}.")
    try:
        number = int(str(value))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label} must be an integer >= {minimum}.") from exc
    if number < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}.")
    return number


def validate_settings(settings):
    """Validate JSON before launching any GPU work; require explicit test split."""
    if not isinstance(settings, dict):
        raise ValueError("Comparison settings must be an object.")
    details = resolve_existing_dataset_yaml(settings.get("existing_data_yaml", ""))
    yaml_path = Path(details["yaml"])
    import yaml
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8-sig"))
    screening_mode = settings.get("screening_mode", False)
    if not isinstance(screening_mode, bool):
        raise ValueError("Screening mode must be true or false.")
    if not screening_mode:
        if not isinstance(config, dict) or not config.get("test"):
            raise ValueError("The selected data.yaml must define a non-empty 'test' split. "
                             "Full comparison never substitutes the validation split for test.")
        tests = config["test"]
        if not isinstance(tests, (str, list)) or (isinstance(tests, list) and not tests):
            raise ValueError("data.yaml 'test' must be a path or a non-empty list of paths.")
    models = settings.get("models", [])
    if not isinstance(models, list) or len(models) < 2 or len(models) > 20:
        raise ValueError("Select between 2 and 20 models for comparison.")
    normalized = []
    for model in models:
        if not isinstance(model, dict):
            raise ValueError("Each queued model must be an object.")
        if model.get("model_path"):
            path = Path(model["model_path"]).expanduser().resolve()
            if not path.is_file() or path.suffix.lower() != ".pt":
                raise ValueError(f"Weights file does not exist: {path}")
            normalized.append({"model_path": str(path), "label": path.stem})
        else:
            version, size = str(model.get("version", "11")), str(model.get("size", "n"))
            weights = _build_model_filename(version, size)
            normalized.append({"version": version, "size": size, "label": weights.removesuffix(".pt")})
    cache = settings.get("cache", "disk")
    if cache not in VALID_CACHE:
        raise ValueError("Cache must be off, disk, or ram.")
    workers = _positive_int(settings.get("workers", 8), "Workers", 0)
    batch = _positive_int(settings.get("batch", 4), "Training batch")
    imgsz = _positive_int(settings.get("imgsz", 640), "Image size")
    eval_batch = _positive_int(settings.get("eval_batch", 4), "Test batch")
    epochs = settings.get("epochs", 100)
    duration = settings.get("time")
    if duration is not None:
        duration = float(duration)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Training duration must be a positive number of hours.")
        epochs = None
    else:
        epochs = _positive_int(epochs, "Epochs")
    cls = float(settings.get("cls_weight", .5))
    if not math.isfinite(cls) or cls < 0:
        raise ValueError("Classification loss weight must be finite and non-negative.")
    device = _normalize_device_choice(settings.get("device", "auto"))
    root = Path(settings.get("output_dir") or yaml_path.parent / "model_comparisons").expanduser().resolve()
    if root.exists() and not root.is_dir():
        raise ValueError("The comparison output location is not a directory.")
    normalized_settings = {
        "existing_data_yaml": str(yaml_path), "models": normalized,
        "screening_mode": screening_mode,
        "output_dir": str(root), "cache": cache, "cleanup_disk_cache": bool(settings.get("cleanup_disk_cache", True)),
        "workers": workers, "batch": batch, "eval_batch": eval_batch,
        "imgsz": imgsz, "epochs": epochs, "time": duration, "cls_weight": cls,
        "device": device, "auto_install_cuda": bool(settings.get("auto_install_cuda", False)),
        "augmentation": validate_augmentation(settings.get("augmentation")),
        "compact_progress": bool(settings.get("compact_progress", True)),
        "retry_oom": bool(settings.get("retry_oom", True)),
    }
    return normalized_settings


def _unique_run_folder(root):
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("Comparison_%Y%m%d_%H%M%S_%f")
    for i in range(100):
        folder = root / (stamp if i == 0 else f"{stamp}_{i}")
        try:
            folder.mkdir()
            return folder
        except FileExistsError:
            continue
    raise FileExistsError("Could not allocate a unique comparison folder.")


def _save_summary(run_root, results, status, settings):
    payload = {"status": status, "dataset_yaml": settings["existing_data_yaml"],
               "created": run_root.name, "screening_mode": settings.get("screening_mode", False),
               "evaluated_split": "val" if settings.get("screening_mode", False) else "test",
               "models": results}
    temp = run_root / "comparison.json.tmp"
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(run_root / "comparison.json")
    with (run_root / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        columns = ["model", "status", "precision", "recall", "map50", "map50_95",
                   "evaluated_split", "train_seconds", "weights_mb", "train_batch",
                   "best_epoch", "weights", "training_run", "test_run", "error"]
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def _run_child(command, log_path, update=None):
    """Stream subprocess logs safely without importing torch in this process."""
    env = os.environ.copy()
    # Match existing training script's opt-in OpenMP workaround in workers only.
    # Never mutate the GUI process's environment or enable it globally.
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with log_path.open("w", encoding="utf-8") as logfile:
        child = subprocess.Popen(command, cwd=str(Path(__file__).parent), env=env,
                                 stdout=logfile, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, creationflags=flags)
    offset = 0
    pending = ""
    last_progress = 0
    outcome = {"run_dir": None, "oom": False}
    while True:
        with log_path.open("r", encoding="utf-8", errors="replace") as stream:
            stream.seek(offset)
            chunk = stream.read(256 * 1024)
            offset = stream.tell()
        if chunk:
            lines = (pending + chunk).replace("\r", "\n").split("\n")
            pending = lines.pop()
            for line in lines:
                if "out of memory" in line.lower() or "cuda error: out of memory" in line.lower():
                    outcome["oom"] = True
                run_dir = parse_training_run_dir(line)
                if run_dir:
                    outcome["run_dir"] = str(run_dir)
                progress = parse_iteration_progress(line)
                if progress is not None and update is not None:
                    now = time.monotonic()
                    if now - last_progress > .45 or progress[2] == progress[3]:
                        update("batch", epoch=progress[0], epochs=progress[1],
                               iteration=progress[2], iterations=progress[3])
                        last_progress = now
                metrics = parse_epoch_metrics(line)
                if metrics is not None and update is not None:
                    update("epoch", **metrics)
        code = child.poll()
        if code is not None:
            if log_path.stat().st_size > offset:
                continue
            if pending:
                if "out of memory" in pending.lower():
                    outcome["oom"] = True
                maybe_dir = parse_training_run_dir(pending)
                if maybe_dir:
                    outcome["run_dir"] = str(maybe_dir)
            return code, outcome
        time.sleep(.17)


def _train_config(settings, model, project, name, batch):
    config = {
        "existing_data_yaml": settings["existing_data_yaml"],
        "dataset_paths": None, "train_after_split": True,
        "batch": batch, "imgsz": settings["imgsz"], "workers": settings["workers"],
        "epochs": settings["epochs"], "time": settings["time"],
        "cache": settings["cache"], "cleanup_disk_cache": settings["cleanup_disk_cache"],
        "cls_weight": settings["cls_weight"], "augmentation": settings["augmentation"],
        "device": settings["device"], "auto_install_cuda": settings["auto_install_cuda"],
        "project": str(project), "run_name": name, "compact_progress": settings["compact_progress"],
    }
    if model.get("model_path"):
        config["model_path"] = model["model_path"]
    else:
        config.update(version=model["version"], size=model["size"])
    return config


def run_comparison(raw):
    settings = validate_settings(raw)
    run_root = _unique_run_folder(Path(settings["output_dir"]))
    (run_root / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    results = []
    total = len(settings["models"])
    emit("started", output_dir=str(run_root), total=total)
    for index, model in enumerate(settings["models"], 1):
        label = model["label"]
        safe_label = re.sub(r"[^a-zA-Z0-9_.-]", "_", label)[:42]
        stem = f"{index:02d}_{safe_label}"
        record = {"model": label, "status": "training", "precision": None,
                  "recall": None, "map50": None, "map50_95": None,
                  "evaluated_split": "val" if settings["screening_mode"] else "test",
                  "train_seconds": None, "weights_mb": None,
                  "train_batch": settings["batch"], "best_epoch": None,
                  "weights": "", "training_run": "", "test_run": "", "error": ""}
        results.append(record)
        emit("model", index=index, total=total, model=label, phase="training")
        batch = settings["batch"]
        attempts = 3 if settings["retry_oom"] else 1
        successful = False
        training_started = time.monotonic()
        for attempt in range(attempts):
            run_name = stem if attempt == 0 else f"{stem}_retry_batch{batch}"
            cfg = _train_config(settings, model, run_root / "training", run_name, batch)
            config_file = run_root / f"{run_name}_settings.json"
            config_file.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            if cfg["cache"] == "disk" and cfg["cleanup_disk_cache"]:
                cfg["cache_cleanup_manifest"] = str(run_root / f"{run_name}_disk_cache_manifest.json")
                config_file.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            logfile = run_root / f"{run_name}_train.log"
            emit("message", text=f"Training {label} (batch={batch}), worker subprocess {attempt+1}/{attempts}")
            code, details = _run_child([sys.executable, "-u", str(Path(__file__).parent / "run_training_job.py"),
                                        str(config_file)], logfile, update=lambda k, **d: emit(k, **d))
            # Worker exit frees CUDA caches and any large Windows dataloader objects.
            if cfg.get("cache_cleanup_manifest") and Path(cfg["cache_cleanup_manifest"]).exists():
                from train_and_split_tool_GUI import cleanup_disk_cache_manifest
                cleanup_disk_cache_manifest(cfg["cache_cleanup_manifest"])
            if code == 0:
                successful = True
                record["train_batch"] = batch
                record["training_run"] = details["run_dir"] or str(run_root / "training" / run_name)
                break
            if details["oom"] and settings["retry_oom"] and batch > 1 and attempt < attempts - 1:
                new_batch = max(1, batch // 2)
                emit("message", text=f"CUDA OOM for {label} at batch {batch}; process exited, retrying at {new_batch}.")
                batch = new_batch
                continue
            record.update(status="failed", error=f"Training exit {code}. See {logfile.name}")
            emit("message", text=f"Training failed for {label}; see {logfile}")
            break
        if not successful:
            _save_summary(run_root, results, "in_progress", settings)
            emit("result", index=index, total=total, result=record)
            continue
        record["train_seconds"] = round(time.monotonic() - training_started, 2)
        train_run = Path(record["training_run"])
        best = train_run / "weights" / "best.pt"
        record["weights"] = str(best)
        if best.is_file():
            record["weights_mb"] = round(best.stat().st_size / (1024 ** 2), 2)
        if not best.is_file():
            record.update(status="failed", error="Training completed without weights/best.pt")
            _save_summary(run_root, results, "in_progress", settings)
            emit("result", index=index, total=total, result=record)
            continue
        try:
            from training_results import load_run
            record["best_epoch"] = (load_run(train_run)["peak"] or {}).get("epoch")
        except (OSError, ValueError):
            pass
        target_split = "val" if settings["screening_mode"] else "test"
        emit("model", index=index, total=total, model=label, phase=target_split)
        eval_settings = {"weights": str(best), "dataset_yaml": settings["existing_data_yaml"],
                         "split": target_split,
                         "imgsz": settings["imgsz"], "batch": settings["eval_batch"],
                         "device": settings["device"], "workers": settings["workers"],
                         "project": str(run_root / target_split), "name": stem}
        eval_json = run_root / f"{stem}_eval_settings.json"
        log_path = run_root / f"{stem}_test.log"
        eval_result = run_root / f"{stem}_test_result.json"
        eval_settings["result_json"] = str(eval_result)
        eval_ok = False
        while True:
            eval_json.write_text(json.dumps(eval_settings, indent=2), encoding="utf-8")
            emit("message", text=f"Evaluating {label} on YAML {target_split} split (batch={eval_settings['batch']})")
            code, detail = _run_child([sys.executable, "-u", str(Path(__file__).parent / "run_model_comparison_eval.py"),
                                       str(eval_json)], log_path)
            if code == 0 and eval_result.is_file():
                metrics = json.loads(eval_result.read_text(encoding="utf-8"))
                record.update({key: metrics.get(key) for key in METRICS})
                record["test_run"] = metrics.get("save_dir", "")
                record["status"] = "completed"
                eval_ok = True
                break
            if detail["oom"] and eval_settings["batch"] > 1:
                old = eval_settings["batch"]
                eval_settings["batch"] = max(1, old // 2)
                emit("message", text=f"Test-set CUDA OOM at batch {old}; retrying with batch {eval_settings['batch']}.")
                continue
            record.update(status="failed", error=f"{target_split} evaluation exit {code}; see {log_path.name}")
            break
        _save_summary(run_root, results, "in_progress", settings)
        emit("result", index=index, total=total, result=record)
        emit("message", text=f"{label}: " + (
            f"{target_split} mAP50–95={record['map50_95']:.4f}, precision={record['precision']:.4f}, recall={record['recall']:.4f}"
            if eval_ok else record["error"]))
    status = "complete" if all(x["status"] == "completed" for x in results) else "partial"
    _save_summary(run_root, results, status, settings)
    emit("finished", status=status, output_dir=str(run_root), results=results)
    return status


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        raise SystemExit("Usage: run_model_comparison_job.py settings.json")
    settings = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    run_comparison(settings)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.exit(1)
