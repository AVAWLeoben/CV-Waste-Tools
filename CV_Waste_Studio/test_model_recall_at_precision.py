# -*- coding: utf-8 -*-
"""CV Waste Tools: fast, persistent Recall@Precision evaluation.

One batched Ultralytics predict pass at conf=0.01. Cache raw detections, then
compute an exact, confidence-ranked, one-to-one IoU match against ground truth.
Change target precision or GT annotations without running inference again.

This is instance-level precision/recall at the configured matching IoU, NOT
mass-based sorting purity. Predictions below 0.01 are not in the cache.
"""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import time
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import traceback

from recall_report import load_evaluation, saved_evaluations, save_evaluation_report, REPORT_NAME

os.environ.setdefault("MPLBACKEND", "Agg")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CONF_FLOOR = 0.01
CACHE_VERSION = 2


class EvaluationCancelled(Exception):
    """Raised when the user requests a safe stop of Recall@Precision."""


def _check_cancelled(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise EvaluationCancelled("Recall@Precision evaluation stopped by user.")


def parse_target_precision(text: str) -> float:
    raw = str(text).strip().replace(",", ".")
    if raw.endswith("%"):
        raw = raw[:-1].strip()
        value = float(raw) / 100.0
    else:
        value = float(raw)
        if value > 1:
            value /= 100.0
    if not 0 < value <= 1:
        raise ValueError("Target precision must be between 0 and 100% (exclusive of zero).")
    return value


def discover_dataset_yaml(folder: Path) -> Path | None:
    for name in ("autosplit_data.yaml", "data.yaml", "dataset.yaml",
                 "autosplit_data.yml", "data.yml", "dataset.yml"):
        file = folder / name
        if file.is_file():
            return file
    return None


def _normal_path(path: Path | str) -> str:
    return os.path.normcase(os.path.normpath(str(Path(path).resolve())))


def _image_files(folder: Path, recursive: bool) -> list[Path]:
    it = folder.rglob("*") if recursive else folder.iterdir()
    return sorted((p.resolve() for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
                  key=lambda p: str(p).lower())


def _candidate_paths(value: str, roots: list[Path]) -> list[Path]:
    p = Path(os.path.expandvars(value)).expanduser()
    if p.is_absolute():
        return [p]
    return [(root / p) for root in roots]


def _find_path(value: str, roots: list[Path]) -> Path:
    candidates = _candidate_paths(value, roots)
    for p in candidates:
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(f"Dataset entry not found: {value}\nTried: " + "; ".join(map(str, candidates)))


def _expand_split_entry(item: str | Path, roots: list[Path], seen_manifest: set[Path]) -> list[Path]:
    entry = _find_path(str(item), roots)
    if entry.is_dir():
        return _image_files(entry, recursive=True)
    if entry.suffix.lower() in IMAGE_EXTENSIONS:
        return [entry]
    if entry.suffix.lower() == ".txt":
        if entry in seen_manifest:
            raise ValueError(f"Recursive dataset list: {entry}")
        seen_manifest.add(entry)
        lines = entry.read_text(encoding="utf-8-sig").splitlines()
        images = []
        for line in lines:
            line = line.strip().strip('"')
            if not line or line.startswith("#"):
                continue
            # Ultralytics split manifests usually resolve './' paths relative
            # to the manifest directory, not to the current working directory.
            images.extend(_expand_split_entry(line, [entry.parent, *roots], seen_manifest))
        return images
    raise ValueError(f"Unsupported dataset entry: {entry}")


def collect_images(settings: dict) -> list[Path]:
    if settings["mode"] == "Image folder":
        root = Path(settings["source"])
        if not root.is_dir():
            raise ValueError(f"Image folder does not exist: {root}")
        images = _image_files(root, recursive=False)
    else:
        import yaml
        yaml_path = Path(settings["source"]).resolve()
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError(f"Invalid dataset YAML: {yaml_path}")
        split = settings["split"]
        if not data.get(split):
            raise ValueError(f"Dataset YAML does not contain a '{split}' split: {yaml_path}")
        root_str = data.get("path")
        if root_str:
            dataset_root = Path(os.path.expandvars(str(root_str))).expanduser()
            if not dataset_root.is_absolute():
                dataset_root = yaml_path.parent / dataset_root
        else:
            dataset_root = yaml_path.parent
        roots = [dataset_root.resolve(), yaml_path.parent.resolve()]
        entries = data[split]
        if not isinstance(entries, list):
            entries = [entries]
        images = []
        for entry in entries:
            images.extend(_expand_split_entry(str(entry), roots, set()))
    images = list(dict.fromkeys(p.resolve() for p in images))
    if not images:
        raise ValueError("No images found in the selected dataset/split.")
    return images


def image_to_label(image: Path) -> Path:
    """Standard Ultralytics images/ -> labels/ mapping; otherwise same folder."""
    parts = list(image.parts)
    index = next((i for i in range(len(parts) - 2, -1, -1)
                  if parts[i].lower() == "images"), None)
    if index is not None:
        parts[index] = "labels"
        paired = Path(*parts).with_suffix(".txt")
        if paired.is_file():
            return paired
        same_dir = image.with_suffix(".txt")
        return same_dir if same_dir.is_file() else paired
    return image.with_suffix(".txt")


def _parse_yolo_label(path: Path):
    """Return (class, xyxy) labels, including 5-column and polygon datasets."""
    import numpy as np
    if not path.is_file():
        return np.empty((0, 5), dtype=np.float32)
    output = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            v = np.asarray([float(x) for x in line.split()], dtype=np.float64)
            if not np.isfinite(v).all() or v[0] < 0 or int(v[0]) != v[0]:
                raise ValueError("non-finite value or invalid class ID")
            if len(v) in (5, 6):
                xc, yc, w, h = v[1:5]
                if w <= 0 or h <= 0:
                    raise ValueError("box dimensions must be positive")
                x1, y1, x2, y2 = xc-w/2, yc-h/2, xc+w/2, yc+h/2
            elif len(v) >= 7 and len(v) % 2 == 1:
                xy = v[1:].reshape(-1, 2)
                x1, y1 = xy.min(axis=0)
                x2, y2 = xy.max(axis=0)
            else:
                raise ValueError(f"expected 5/6-column bbox or class + polygon points; got {len(v)} columns")
            output.append([int(v[0]), x1, y1, x2, y2])
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Invalid annotation at {path}:{lineno}: {exc}") from exc
    return np.asarray(output, dtype=np.float32).reshape(-1, 5)


def ground_truth(images: list[Path], emit, cancel_event=None):
    """Read labels afresh every run; editing labels does not invalidate predictions."""
    labels = []
    missing = 0
    targets = 0
    for image in images:
        _check_cancelled(cancel_event)
        label = image_to_label(image)
        if not label.is_file():
            missing += 1
        boxes = _parse_yolo_label(label)
        labels.append(boxes)
        targets += len(boxes)
    _check_cancelled(cancel_event)
    emit("log", f"Ground truth: {targets} boxes in {len(images)} images; "
                f"{missing} images without label files (treated as background).")
    if targets == 0:
        raise ValueError("No ground-truth objects found. Check images/labels paths and split selection.")
    return labels


def _cache_identity(settings: dict, images: list[Path]) -> str:
    """Identity of model, prediction settings, and ordered image files only."""
    model = Path(settings["model"]).resolve()
    h = hashlib.sha256()
    identity = {"cache_version": CACHE_VERSION, "model": _normal_path(model),
                "model_size": model.stat().st_size, "model_mtime_ns": model.stat().st_mtime_ns,
                "imgsz": settings["imgsz"], "conf": CONF_FLOOR,
                "nms_iou": settings["nms_iou"], "max_det": settings["max_det"]}
    # Retain the old cache key when disabled, so previous inference caches still work.
    if settings.get("agnostic_nms", False):
        identity["agnostic_nms"] = True
    h.update(json.dumps(identity, sort_keys=True).encode("utf-8"))
    for image in images:
        stat = image.stat()
        h.update(_normal_path(image).encode("utf-8"))
        h.update(f"|{stat.st_size}|{stat.st_mtime_ns}\n".encode("ascii"))
    return h.hexdigest()[:24]


def _gpu_visible_via_nvidia_smi() -> str | None:
    smi = shutil.which("nvidia-smi")
    if not smi and os.name == "nt":
        for exe in (r"C:\Windows\System32\nvidia-smi.exe",
                    r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"):
            if Path(exe).is_file():
                smi = exe
                break
    if not smi:
        return None
    try:
        cp = subprocess.run([smi, "--query-gpu=name", "--format=csv,noheader"],
                            capture_output=True, text=True, timeout=8)
        if cp.returncode == 0 and cp.stdout.strip():
            return cp.stdout.splitlines()[0].strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def choose_device(selection: str, emit) -> str:
    """Refuse silent CPU fallback when NVIDIA hardware is detected."""
    if selection == "CPU":
        emit("log", "Device explicitly set to CPU.")
        return "cpu"
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)
        arches = torch.cuda.get_arch_list()
        # Runtime computation detects arch-incompatible wheels more reliably
        # than torch.cuda.is_available() by itself.
        try:
            tensor = torch.ones((2,), device="cuda:0")
            assert tensor.sum().item() == 2.0
            torch.cuda.synchronize()
        except Exception as exc:
            raise RuntimeError(f"CUDA detected {name} (sm_{cc[0]}{cc[1]}) but GPU "
                               f"compute failed. Install a compatible CUDA PyTorch build. "
                               f"Compiled architectures: {arches}. Error: {exc}") from exc
        emit("log", f"GPU inference enabled: {name} (sm_{cc[0]}{cc[1]}) "
                    f"| torch {torch.__version__} | CUDA runtime {torch.version.cuda}")
        return "0"
    gpu = _gpu_visible_via_nvidia_smi()
    if gpu:
        raise RuntimeError(
            f"NVIDIA GPU detected ({gpu}), but this Python environment has no working CUDA PyTorch. "
            "The launcher selected a CPU-only environment. Use 'Recheck / repair environment' "
            "and verify torch.cuda.is_available() in that environment. Inference was NOT run on CPU."
        )
    if selection == "GPU 0":
        raise RuntimeError("GPU 0 requested, but torch.cuda.is_available() is false.")
    emit("log", "No NVIDIA GPU detected; using CPU for inference.")
    return "cpu"


def _names_dict(names):
    return ({str(int(k)): str(v) for k, v in names.items()} if isinstance(names, dict)
            else {str(i): str(name) for i, name in enumerate(names)})


def predict_and_cache(settings: dict, images: list[Path], cache_path: Path,
                      meta_path: Path, emit, cancel_event=None):
    """One *streaming/batched* predictor invocation for all images."""
    import numpy as np
    from ultralytics import YOLO
    _check_cancelled(cancel_event)
    inference_started = time.perf_counter()
    device = choose_device(settings["device"], emit)
    emit("status", "Loading model and running batched inference ...")
    model = YOLO(settings["model"])
    names = _names_dict(model.names)
    index_of = {_normal_path(p): i for i, p in enumerate(images)}
    counts = np.zeros(len(images), dtype=np.int32)
    visited = set()
    chunks = []
    with tempfile.TemporaryDirectory(prefix="cvw_recall_predict_") as tmp:
        manifest = Path(tmp) / "images.txt"
        manifest.write_text("\n".join(str(p) for p in images) + "\n", encoding="utf-8")
        emit("log", f"Predicting {len(images)} images in batches of {settings['batch']} "
                    f"| imgsz={settings['imgsz']} | conf={CONF_FLOOR} "
                    f"| NMS IoU={settings['nms_iou']} "
                    f"| agnostic_nms={settings.get('agnostic_nms', False)} | device={device}")
        stream = model.predict(source=str(manifest), conf=CONF_FLOOR,
                               imgsz=settings["imgsz"], batch=settings["batch"],
                               iou=settings["nms_iou"], max_det=settings["max_det"],
                               agnostic_nms=settings.get("agnostic_nms", False),
                               device=device, stream=True, save=False, verbose=False)
        try:
            for count, result in enumerate(stream, 1):
                _check_cancelled(cancel_event)
                key = _normal_path(result.path)
                if key not in index_of:
                    raise RuntimeError(f"Unexpected result path from Ultralytics: {result.path}")
                idx = index_of[key]
                if idx in visited:
                    raise RuntimeError(f"Duplicate inference result: {result.path}")
                visited.add(idx)
                if result.boxes is not None and len(result.boxes):
                    xyxy = result.boxes.xyxyn.cpu().numpy().astype(np.float32)
                    conf = result.boxes.conf.cpu().numpy().astype(np.float32)
                    cls = result.boxes.cls.cpu().numpy().astype(np.int32)
                    counts[idx] = len(conf)
                    chunks.append((np.full(len(conf), idx, np.int32), cls, conf, xyxy))
                if count % 100 == 0 or count == len(images):
                    emit("status", f"GPU/CPU inference: {count}/{len(images)} images")
                    emit("log", f"Processed {count}/{len(images)} images")
        finally:
            # Stop consuming results and release the streaming predictor cleanly.
            close = getattr(stream, "close", None)
            if callable(close):
                close()
    _check_cancelled(cancel_event)
    if len(visited) != len(images):
        raise RuntimeError(f"Incomplete inference: {len(visited)}/{len(images)} images returned. "
                           "No partial cache was saved.")
    if chunks:
        image_index = np.concatenate([a[0] for a in chunks])
        classes = np.concatenate([a[1] for a in chunks])
        confidence = np.concatenate([a[2] for a in chunks])
        xyxy = np.concatenate([a[3] for a in chunks])
    else:
        image_index = np.empty((0,), np.int32)
        classes = np.empty((0,), np.int32)
        confidence = np.empty((0,), np.float32)
        xyxy = np.empty((0, 4), np.float32)
    # Save atomically so a crash cannot leave a reusable partial cache.
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_npz = cache_path.with_name(cache_path.name + ".tmp")
    try:
        with tmp_npz.open("wb") as file:
            np.savez_compressed(file, image_index=image_index, classes=classes,
                                confidence=confidence, xyxy=xyxy)
        _check_cancelled(cancel_event)
        os.replace(tmp_npz, cache_path)
    finally:
        tmp_npz.unlink(missing_ok=True)
    metadata = {
        "cache_version": CACHE_VERSION, "image_count": len(images),
        "prediction_count": int(len(confidence)), "model": settings["model"],
        "dataset": settings["source"], "class_names": names,
        "device": device, "created": datetime.now().isoformat(timespec="seconds"),
        "conf_floor": CONF_FLOOR, "imgsz": settings["imgsz"],
        "batch": settings["batch"], "nms_iou": settings["nms_iou"],
        "agnostic_nms": settings.get("agnostic_nms", False),
        "max_det": settings["max_det"],
        "inference_seconds": round(time.perf_counter() - inference_started, 3),
    }
    tmp_json = meta_path.with_suffix(meta_path.suffix + ".tmp")
    try:
        tmp_json.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        _check_cancelled(cancel_event)
        os.replace(tmp_json, meta_path)
    finally:
        tmp_json.unlink(missing_ok=True)
    emit("log", f"Saved {len(confidence)} detections to persistent cache: {cache_path}")
    emit("log", f"One-pass inference + model loading: {metadata['inference_seconds']:.2f} s")
    return {"image_index": image_index, "classes": classes,
            "confidence": confidence, "xyxy": xyxy}, metadata


def load_cached(cache_path: Path, meta_path: Path, n_images: int):
    import numpy as np
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata.get("cache_version") != CACHE_VERSION or metadata.get("image_count") != n_images:
        raise ValueError("Cache metadata does not match the dataset; rerun inference.")
    with np.load(cache_path, allow_pickle=False) as data:
        result = {key: data[key].copy() for key in ("image_index", "classes", "confidence", "xyxy")}
    n = len(result["confidence"])
    if any(len(result[k]) != n for k in result) or result["xyxy"].shape != (n, 4):
        raise ValueError("Prediction cache is inconsistent; rerun inference.")
    if n and (result["image_index"].min() < 0 or result["image_index"].max() >= n_images):
        raise ValueError("Prediction cache contains invalid image indices.")
    return result, metadata


def box_iou(box: "np.ndarray", candidates: "np.ndarray"):
    import numpy as np
    lt = np.maximum(box[:2], candidates[:, :2])
    rb = np.minimum(box[2:], candidates[:, 2:])
    size = np.maximum(rb - lt, 0)
    intersect = size[:, 0] * size[:, 1]
    a = max(0., float(box[2] - box[0])) * max(0., float(box[3] - box[1]))
    b = np.maximum(0., candidates[:, 2] - candidates[:, 0]) * np.maximum(0., candidates[:, 3] - candidates[:, 1])
    return intersect / np.maximum(a + b - intersect, 1e-9)


def compute_curve(pred: dict, gt_by_image: list, food_class: int,
                  target: float, match_iou: float, cancel_event=None):
    """Exact greedy confidence-ranked class-aware IoU matching, run only on CPU.

    A target-class detection is TP iff it matches an unmatched target-class
    ground-truth box with IoU >= match_iou. Other classes never count as TP.
    """
    import numpy as np
    gt_count = sum(int(np.count_nonzero(gt[:, 0] == food_class)) for gt in gt_by_image)
    if not gt_count:
        raise ValueError(f"There are no ground-truth instances of target class {food_class} in this split.")
    pred_mask = (pred["classes"] == food_class) & (pred["confidence"] >= CONF_FLOOR)
    conf = np.asarray(pred["confidence"][pred_mask], dtype=np.float64)
    image_indices = np.asarray(pred["image_index"][pred_mask], dtype=np.int32)
    boxes = np.asarray(pred["xyxy"][pred_mask], dtype=np.float64)
    tp = np.zeros(len(conf), dtype=np.int8)
    if len(conf):
        # Group detections by image. The confidence-sorted matching decisions
        # are independent of the threshold; changing cutoff only truncates
        # this ordered detection sequence.
        ordered_images = np.argsort(image_indices, kind="stable")
        image_sorted = image_indices[ordered_images]
        boundaries = np.searchsorted(image_sorted, np.arange(len(gt_by_image) + 1))
        for idx, gt in enumerate(gt_by_image):
            _check_cancelled(cancel_event)
            if boundaries[idx] == boundaries[idx + 1]:
                continue
            local = ordered_images[boundaries[idx]:boundaries[idx + 1]]
            local = local[np.argsort(-conf[local], kind="stable")]
            food_boxes = np.asarray(gt[gt[:, 0] == food_class, 1:5], dtype=np.float64)
            if not len(food_boxes):
                continue
            used = np.zeros(len(food_boxes), dtype=bool)
            for j in local:
                _check_cancelled(cancel_event)
                ious = box_iou(boxes[j], food_boxes)
                ious[used] = -1
                best = int(np.argmax(ious))
                if ious[best] >= match_iou:
                    used[best] = True
                    tp[j] = 1
        ranking = np.argsort(-conf, kind="stable")
        ranked_conf = conf[ranking]
        ranked_tp = tp[ranking]
        cum_tp = np.cumsum(ranked_tp, dtype=np.int64)
        # Include each actual score: unlike an interpolated PR curve, the
        # reported TP/FP/purity is achieved at the saved confidence cutoff.
        thresholds = np.unique(np.concatenate((ranked_conf, [CONF_FLOOR, 1.0])))
        n_pred = np.searchsorted(-ranked_conf, -thresholds, side="right")
        true_positives = np.where(n_pred > 0, cum_tp[np.maximum(n_pred - 1, 0)], 0)
    else:
        thresholds = np.array([CONF_FLOOR, 1.0], dtype=np.float64)
        n_pred = np.zeros(len(thresholds), dtype=np.int64)
        true_positives = np.zeros(len(thresholds), dtype=np.int64)
    false_positives = n_pred - true_positives
    precision = np.divide(true_positives, n_pred,
                          out=np.zeros(len(n_pred), dtype=np.float64), where=n_pred > 0)
    recall = true_positives / gt_count
    valid = np.flatnonzero((n_pred > 0) & (precision + 1e-12 >= target))
    result = {
        "confidence": None, "precision": None, "recall": None,
        "tp": None, "fp": None, "fn": None, "target_reached": bool(len(valid)),
        "target_precision": float(target), "food_class": food_class,
        "gt_count": int(gt_count), "prediction_count_food": int(len(conf)),
        "thresholds": thresholds, "curve_precision": precision, "curve_recall": recall,
        "curve_tp": true_positives, "curve_fp": false_positives,
        "curve_fn": gt_count - true_positives,
    }
    if len(valid):
        # Smallest threshold among maximum-recall valid cutoffs.
        best_recall = max(recall[valid])
        best_idx = int(valid[np.flatnonzero(recall[valid] >= best_recall - 1e-12)[0]])
        result.update(confidence=float(thresholds[best_idx]),
                      precision=float(precision[best_idx]),
                      recall=float(recall[best_idx]),
                      tp=int(true_positives[best_idx]),
                      fp=int(false_positives[best_idx]),
                      fn=int(gt_count - true_positives[best_idx]))
    return result


def _save_plots(output: Path, curve: dict):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    c = curve["thresholds"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(c, curve["curve_precision"], label="Target-class precision")
    ax.plot(c, curve["curve_recall"], label="Target-class recall")
    ax.axhline(curve["target_precision"], linestyle="--", linewidth=1,
               label=f"Target precision ({curve['target_precision']:.0%})")
    if curve["confidence"] is not None:
        ax.axvline(curve["confidence"], linestyle=":", linewidth=1,
                   label=f"Operating threshold ({curve['confidence']:.4f})")
    ax.set(xlabel="Confidence threshold", ylabel="Fraction", ylim=(0, 1.02), xlim=(CONF_FLOOR, 1.))
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "food_precision_recall_vs_confidence.png", dpi=150)
    plt.close(fig)


def evaluate(settings: dict, emit, *, cache_only: bool = False, cancel_event=None):
    """Cache inference once, recompute PR from saved detections on every run."""
    analysis_started = time.perf_counter()
    _check_cancelled(cancel_event)
    model_file = Path(settings["model"])
    if not model_file.is_file():
        raise ValueError(f"Model file does not exist: {model_file}")
    dest = Path(settings["output"])
    dest.mkdir(parents=True, exist_ok=True)
    emit("status", "Resolving images and cache ...")
    images = collect_images(settings)
    _check_cancelled(cancel_event)
    emit("log", f"Found {len(images)} dataset images.")
    digest = _cache_identity(settings, images)
    cache_dir = dest / "prediction_cache"
    cache_path = cache_dir / f"detections_{digest}.npz"
    meta_path = cache_dir / f"detections_{digest}.json"
    cache_exists = cache_path.is_file() and meta_path.is_file()
    if cache_only and not cache_exists:
        raise ValueError("No matching prediction cache found for this model, image set, "
                         "image size and NMS settings. Run inference first.")
    if cache_exists and (cache_only or settings.get("reuse_cache", True)):
        emit("status", "Loading cached detections (no model inference) ...")
        pred, meta = load_cached(cache_path, meta_path, len(images))
        _check_cancelled(cancel_event)
        emit("log", f"CACHE HIT — {len(pred['confidence'])} detections, "
                    f"original device: {meta.get('device')}. No model or GPU initialization.")
        used_cache = True
    else:
        if cache_only:
            raise ValueError("Prediction cache unavailable.")
        pred, meta = predict_and_cache(settings, images, cache_path, meta_path, emit,
                                       cancel_event=cancel_event)
        used_cache = False
    emit("status", "Matching detections to annotations and sweeping confidence ...")
    gt = ground_truth(images, emit, cancel_event=cancel_event)
    class_id = settings["food_class"]
    names = meta.get("class_names", {})
    if names and str(class_id) not in names:
        raise ValueError(f"Class {class_id} is not in model classes: {names}")
    curve = compute_curve(pred, gt, class_id, settings["precision"], settings["match_iou"],
                          cancel_event=cancel_event)
    _check_cancelled(cancel_event)
    run_dir = dest / ("precision_recall_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    run_dir.mkdir(parents=True)
    curve_path = run_dir / "food_precision_recall_by_confidence.csv"
    try:
        with curve_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["confidence", "precision", "recall", "TP", "FP", "FN"])
            writer.writerows(zip(curve["thresholds"], curve["curve_precision"],
                                 curve["curve_recall"], curve["curve_tp"],
                                 curve["curve_fp"], curve["curve_fn"]))
        _check_cancelled(cancel_event)
        if settings.get("save_plots", False):
            _save_plots(run_dir, curve)
        _check_cancelled(cancel_event)
    except EvaluationCancelled:
        shutil.rmtree(run_dir)
        raise
    summary = {
        "model": str(model_file), "dataset": settings["source"],
        "source_mode": settings["mode"], "split": settings["split"],
        "image_count": len(images), "class_id": class_id,
        "class_name": names.get(str(class_id), f"class_{class_id}"),
        "iou_match_threshold": settings["match_iou"],
        "nms_iou_threshold": settings["nms_iou"],
        "agnostic_nms": settings.get("agnostic_nms", False),
        "confidence_floor": CONF_FLOOR,
        "target_precision": settings["precision"],
        "target_reached": curve["target_reached"],
        "confidence_threshold": curve["confidence"],
        "achieved_precision": curve["precision"],
        "recall_at_precision": curve["recall"],
        "true_positives": curve["tp"], "false_positives": curve["fp"],
        "false_negatives": curve["fn"],
        "food_ground_truth_count": curve["gt_count"],
        "food_prediction_count_at_floor": curve["prediction_count_food"],
        "prediction_device": meta.get("device"),
        "initial_inference_seconds": meta.get("inference_seconds"),
        "evaluation_seconds": round(time.perf_counter() - analysis_started, 3),
        "used_cached_predictions": used_cache,
        "prediction_cache": str(cache_path),
        "output_directory": str(run_dir),
        "curve_csv": str(curve_path),
        "report_png": None,
        "metric_definition": "Instance-level precision/recall; one-to-one greedy same-class IoU matching, not mass-weighted purity",
    }
    try:
        _check_cancelled(cancel_event)
        try:
            report_path = save_evaluation_report(summary)
            summary["report_png"] = str(report_path)
            emit("log", f"Saved evaluation graphic: {report_path}")
        except Exception as exc:
            # A plotting failure must never invalidate an otherwise completed
            # evaluation or require re-running expensive GPU inference.
            emit("log", f"Warning: Could not create evaluation graphic: {exc}")
        _check_cancelled(cancel_event)
        (run_dir / "operating_point_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        _check_cancelled(cancel_event)
    except EvaluationCancelled:
        shutil.rmtree(run_dir)
        raise
    if curve["target_reached"]:
        emit("log", f"Confidence {curve['confidence']:.5f} | Precision {curve['precision']:.2%} "
                    f"| Recall {curve['recall']:.2%} | TP={curve['tp']} FP={curve['fp']} FN={curve['fn']}")
    else:
        emit("log", f"Target precision {settings['precision']:.1%} not reached "
                    f"within the confidence range [{CONF_FLOOR}, 1.0].")
    emit("log", f"Saved result: {run_dir}")
    emit("log", f"Current evaluation elapsed: {summary['evaluation_seconds']:.2f} s"
                + (" (cache only)" if used_cache else " (including inference)"))
    emit("status", "Completed (cached)" if used_cache else "Completed (inference + analysis)")
    emit("done", summary)
    return summary


def open_results_folder(path: str | Path):
    """Open a completed evaluation directory using the system file manager."""
    folder = Path(path).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Results folder does not exist: {folder}")
    if sys.platform == "win32":
        os.startfile(str(folder))
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(folder)])
    else:
        subprocess.Popen(["xdg-open", str(folder)])


class RecallTestGUI:
    def __init__(self, root, *, embedded=False, on_log=None):
        self.embedded = embedded
        self.on_log = on_log
        self.root = root.winfo_toplevel()
        if not embedded:
            self.root.title("CV Waste Tools · Cached Recall at Precision")
            self.root.geometry("980x790")
            self.root.minsize(820, 690)
        self.host = root
        self.messages = queue.Queue()
        self.busy = False
        self.cancel_event = threading.Event()
        self.last_results_dir = None
        self.recent_paths = []
        self.recent_var = tk.StringVar()
        self.model_var = tk.StringVar()
        self.mode_var = tk.StringVar(value="Dataset YAML")
        self.source_var = tk.StringVar()
        self.split_var = tk.StringVar(value="test")
        self.output_var = tk.StringVar(value=str(Path.cwd() / "precision_recall_results"))
        self.precision_var = tk.StringVar(value="95")
        self.class_var = tk.StringVar(value="0")
        self.imgsz_var = tk.StringVar(value="1280")
        self.batch_var = tk.StringVar(value="8")
        self.nms_var = tk.StringVar(value="0.7")
        self.match_var = tk.StringVar(value="0.5")
        self.maxdet_var = tk.StringVar(value="300")
        self.device_var = tk.StringVar(value="Auto")
        self.reuse_var = tk.BooleanVar(value=True)
        self.plots_var = tk.BooleanVar(value=False)
        self.agnostic_nms_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Choose a model and dataset. Inference at conf=0.01 is cached.")
        self.result_var = tk.StringVar(value="No evaluation run yet.")
        self.inputs = []
        self._build_ui()
        if not self.embedded:
            self.root.protocol("WM_DELETE_WINDOW", self._close_standalone)
        root.after(150, self._poll)

    def _build_ui(self):
        if not self.embedded:
            try:
                ttk.Style(self.root).theme_use("clam")
            except tk.TclError:
                pass
        frame = ttk.Frame(self.host, padding=16)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(16, weight=1)
        ttk.Label(frame, text="Cached Recall at Target Precision", font=("Segoe UI", 17, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 5))
        ttk.Button(frame, text="Open saved test...", command=self._browse_saved_result).grid(
            row=0, column=2, sticky="e", pady=(0, 5))
        ttk.Label(frame, text="One Ultralytics batched prediction pass at conf=0.01. Reuse saved boxes for instant confidence sweeps.",
                  foreground="#555555").grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 12))

        def entry_row(row, label, var, browse=None):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 15), pady=5)
            e = ttk.Entry(frame, textvariable=var)
            e.grid(row=row, column=1, sticky="ew", pady=5)
            self.inputs.append(e)
            if browse:
                b = ttk.Button(frame, text="Browse...", command=browse)
                b.grid(row=row, column=2, sticky="ew", padx=(10, 0), pady=5)
                self.inputs.append(b)

        entry_row(2, "YOLO model (.pt)", self.model_var, self._choose_model)
        mode_row = ttk.Frame(frame)
        mode_row.grid(row=3, column=0, columnspan=3, sticky="w", pady=5)
        ttk.Label(mode_row, text="Dataset source").pack(side="left", padx=(0, 18))
        self.mode_choice = ttk.Combobox(mode_row, state="readonly", width=19,
                                        values=("Dataset YAML", "Image folder"), textvariable=self.mode_var)
        self.mode_choice.pack(side="left")
        self.mode_choice.bind("<<ComboboxSelected>>", lambda _e: self._mode_changed())
        self.inputs.append(self.mode_choice)
        self.source_label = ttk.Label(frame, text="Dataset YAML")
        self.source_label.grid(row=4, column=0, sticky="w", padx=(0, 15), pady=5)
        source = ttk.Entry(frame, textvariable=self.source_var)
        source.grid(row=4, column=1, sticky="ew", pady=5)
        browse = ttk.Button(frame, text="Browse...", command=self._choose_source)
        browse.grid(row=4, column=2, sticky="ew", padx=(10, 0), pady=5)
        self.inputs.extend((source, browse))
        self.source_hint = ttk.Label(frame, foreground="#555555", wraplength=700,
                                     text="Choose data.yaml or autosplit_data.yaml. Supports split folders and .txt manifests.")
        self.source_hint.grid(row=5, column=0, columnspan=2, sticky="w", pady=(0, 8))
        choose_folder = ttk.Button(frame, text="Select dataset folder...", command=self._choose_dataset_folder)
        choose_folder.grid(row=5, column=2, sticky="ew", padx=(10, 0), pady=(0, 8))
        self.inputs.append(choose_folder)
        entry_row(6, "Results and cache folder", self.output_var, self._choose_output)

        controls = ttk.Frame(frame)
        controls.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(8, 4))
        specs = [("Target precision %", self.precision_var), ("Target Class", self.class_var),
                 ("Image size", self.imgsz_var), ("Batch", self.batch_var),
                 ("Match IoU", self.match_var), ("NMS IoU", self.nms_var),
                 ("Max detections", self.maxdet_var)]
        for i, (label, var) in enumerate(specs):
            controls.columnconfigure(i, weight=1)
            ttk.Label(controls, text=label).grid(row=0, column=i, sticky="w", padx=(0, 9))
            entry = ttk.Entry(controls, textvariable=var, width=9)
            entry.grid(row=1, column=i, sticky="ew", padx=(0, 9), pady=5)
            self.inputs.append(entry)
        settings_row = ttk.Frame(frame)
        settings_row.grid(row=8, column=0, columnspan=3, sticky="w", pady=7)
        ttk.Label(settings_row, text="Split").pack(side="left", padx=(0, 7))
        self.split_choice = ttk.Combobox(settings_row, textvariable=self.split_var,
                                         values=("test", "val", "train"), state="readonly", width=9)
        self.split_choice.pack(side="left")
        self.inputs.append(self.split_choice)
        ttk.Label(settings_row, text="Device").pack(side="left", padx=(22, 7))
        device = ttk.Combobox(settings_row, values=("Auto", "GPU 0", "CPU"),
                              state="readonly", textvariable=self.device_var, width=9)
        device.pack(side="left")
        self.inputs.append(device)
        ttk.Label(settings_row, text="Inference conf = 0.01 (fixed)",
                  foreground="#555555").pack(side="left", padx=(22, 0))
        options = ttk.Frame(frame)
        options.grid(row=9, column=0, columnspan=3, sticky="w", pady=5)
        reuse = ttk.Checkbutton(options, text="Reuse existing prediction cache", variable=self.reuse_var)
        reuse.pack(side="left", padx=(0, 25))
        plots = ttk.Checkbutton(options, text="Save precision/recall plot", variable=self.plots_var)
        plots.pack(side="left", padx=(0, 25))
        agnostic = ttk.Checkbutton(options, text="Class-agnostic NMS", variable=self.agnostic_nms_var)
        agnostic.pack(side="left")
        self.inputs.extend((reuse, plots, agnostic))
        actions = ttk.Frame(frame)
        actions.grid(row=10, column=0, columnspan=3, sticky="w", pady=(5, 10))
        run = ttk.Button(actions, text="Run / reuse cached predictions", command=lambda: self._start(False))
        self.run_button = run
        run.pack(side="left", padx=(0, 8))
        fast = ttk.Button(actions, text="Recalculate from cache only", command=lambda: self._start(True))
        fast.pack(side="left", padx=(0, 8))
        self.stop_button = ttk.Button(actions, text="Stop evaluation", command=self.cancel,
                                      state="disabled")
        self.stop_button.pack(side="left", padx=(0, 8))
        self.open_results_button = ttk.Button(actions, text="Open Results Folder",
                                              command=self._open_results_folder, state="disabled")
        self.open_results_button.pack(side="left", padx=(0, 8))
        self.view_report_button = ttk.Button(actions, text="View graphic",
                                             state="disabled", command=self._view_report)
        self.view_report_button.pack(side="left")
        self.inputs.extend((run, fast))
        previous = ttk.Frame(frame)
        previous.grid(row=11, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        ttk.Label(previous, text="Previous evaluations").pack(side="left", padx=(0, 8))
        self.recent_choice = ttk.Combobox(previous, textvariable=self.recent_var,
                                          state="readonly", width=35)
        self.recent_choice.pack(side="left", padx=(0, 6))
        self.load_recent_button = ttk.Button(previous, text="Load selected",
                                              command=self._load_selected_result)
        self.load_recent_button.pack(side="left", padx=(0, 6))
        ttk.Button(previous, text="Browse saved result...",
                   command=self._browse_saved_result).pack(side="left", padx=(0, 6))
        ttk.Button(previous, text="Refresh", command=self._refresh_previous_runs).pack(side="left")
        self._refresh_previous_runs()
        ttk.Label(frame, textvariable=self.status_var).grid(row=12, column=0, columnspan=3, sticky="w")
        self.progress = ttk.Progressbar(frame, mode="indeterminate")
        self.progress.grid(row=13, column=0, columnspan=3, sticky="ew", pady=6)
        ttk.Label(frame, textvariable=self.result_var, font=("Segoe UI", 11, "bold"),
                  wraplength=920).grid(row=14, column=0, columnspan=3, sticky="w", pady=(3, 10))
        ttk.Label(frame, text="Metrics: one-to-one target-class IoU matching (object count), NOT mass-weighted sorting purity.",
                  foreground="#555555").grid(row=15, column=0, columnspan=3, sticky="w", pady=(0, 8))
        log_frame = ttk.Frame(frame)
        log_frame.grid(row=16, column=0, columnspan=3, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log = tk.Text(log_frame, state="disabled", height=8, wrap="word",
                           font=("Consolas", 9), background="#f5f6f8")
        self.log.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, command=self.log.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scrollbar.set)
        if self.embedded:
            log_frame.grid_remove()
            frame.rowconfigure(16, weight=0)

    def _refresh_previous_runs(self):
        self.recent_paths = saved_evaluations(self.output_var.get())[:50]
        display = [path.name.removeprefix("precision_recall_") for path in self.recent_paths]
        self.recent_choice.configure(values=display)
        self.recent_var.set(display[0] if display else "")
        self.load_recent_button.configure(state="normal" if display else "disabled")

    def _load_selected_result(self):
        index = self.recent_choice.current()
        if index < 0 or index >= len(self.recent_paths):
            return
        self._load_saved_result(self.recent_paths[index])

    def _browse_saved_result(self):
        folder = filedialog.askdirectory(parent=self.root,
            title="Select a completed precision_recall_* results folder",
            initialdir=self.output_var.get() if Path(self.output_var.get()).is_dir() else str(Path.cwd()))
        if folder:
            self._load_saved_result(folder)

    def _load_saved_result(self, folder):
        if self.busy:
            messagebox.showinfo("Evaluation running", "Stop the current evaluation before opening an older result.",
                                parent=self.root)
            return
        try:
            summary = load_evaluation(folder)
            report = Path(summary["output_directory"]) / REPORT_NAME
            if not report.is_file():
                try:
                    save_evaluation_report(summary)
                except Exception as exc:
                    self._append_log(f"Could not create graphic for this older result: {exc}")
            self._show_result(summary, from_disk=True)
            self._append_log(f"Reopened saved evaluation: {summary['output_directory']}")
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            messagebox.showerror("Cannot load evaluation", str(exc), parent=self.root)

    def _show_result(self, summary, *, from_disk=False):
        self.last_results_dir = str(summary["output_directory"])
        self.open_results_button.configure(state="normal")
        self.view_report_button.configure(
            state="normal" if (Path(self.last_results_dir) / REPORT_NAME).is_file() else "disabled")
        if summary["target_reached"]:
            self.result_var.set(
                f"{summary['class_name']} · confidence {summary['confidence_threshold']:.5f} · "
                f"precision {summary['achieved_precision']:.2%} · "
                f"recall {summary['recall_at_precision']:.2%} "
                f"(TP {summary['true_positives']}, FP {summary['false_positives']}, "
                f"FN {summary['false_negatives']})"
            )
        else:
            self.result_var.set(f"Target {summary['target_precision']:.1%} not reached.")
        self.status_var.set(("Loaded saved evaluation: " if from_disk else "Saved in ")
                            + self.last_results_dir)

    def _view_report(self):
        if not self.last_results_dir:
            return
        path = Path(self.last_results_dir) / REPORT_NAME
        if not path.is_file():
            messagebox.showerror("Graphic unavailable", "No summary graphic was saved for this evaluation.",
                                 parent=self.root)
            return
        try:
            from PIL import Image, ImageTk
            image = Image.open(path)
            preview = tk.Toplevel(self.root)
            preview.title(f"Evaluation graphic · {path.parent.name}")
            preview.geometry("1120x750")
            preview.minsize(640, 450)
            holder = ttk.Frame(preview, padding=10)
            holder.pack(fill="both", expand=True)
            panel = ttk.Label(holder)
            panel.pack(fill="both", expand=True)
            def update_preview(_event=None):
                width = max(1, panel.winfo_width() - 8)
                height = max(1, panel.winfo_height() - 8)
                if width < 20 or height < 20:
                    return
                frame = image.copy()
                frame.thumbnail((width, height), Image.Resampling.LANCZOS)
                picture = ImageTk.PhotoImage(frame, master=preview)
                panel.configure(image=picture)
                panel.image = picture
            panel.bind("<Configure>", update_preview)
            ttk.Button(preview, text="Open Results Folder",
                       command=self._open_results_folder).pack(pady=(0, 8))
            preview.after(75, update_preview)
        except Exception as exc:
            messagebox.showerror("Cannot view graphic", str(exc), parent=self.root)

    def _mode_changed(self):
        yaml = self.mode_var.get() == "Dataset YAML"
        self.source_var.set("")
        self.source_label.configure(text="Dataset YAML" if yaml else "Image directory")
        self.source_hint.configure(text=("Choose a data.yaml or autosplit_data.yaml, with folder or .txt split paths."
                                         if yaml else "Choose images/ with sibling labels/, or a folder containing image.jpg + image.txt."))
        self.split_choice.configure(state="readonly" if yaml else "disabled")

    def _choose_dataset_folder(self):
        folder = filedialog.askdirectory(parent=self.root, title="Choose YOLO dataset folder", mustexist=True)
        if folder:
            selected = Path(folder)
            config = discover_dataset_yaml(selected)
            self.mode_var.set("Dataset YAML" if config else "Image folder")
            self._mode_changed()
            self.source_var.set(str(config or (selected / "images" if (selected / "images").is_dir() else selected)))

    def _choose_model(self):
        path = filedialog.askopenfilename(parent=self.root, title="Select YOLO checkpoint",
                                          filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")])
        if path:
            self.model_var.set(path)

    def _choose_source(self):
        if self.mode_var.get() == "Dataset YAML":
            path = filedialog.askopenfilename(parent=self.root, title="Select YOLO YAML",
                                              filetypes=[("YAML", "*.yaml *.yml"), ("All files", "*.*")])
        else:
            path = filedialog.askdirectory(parent=self.root, title="Select image folder", mustexist=True)
        if path:
            self.source_var.set(path)

    def _choose_output(self):
        path = filedialog.askdirectory(parent=self.root, title="Choose results folder", mustexist=True)
        if path:
            self.output_var.set(path)
            self._refresh_previous_runs()

    def _settings(self):
        model_text = self.model_var.get().strip().strip('"')
        source_text = self.source_var.get().strip().strip('"')
        output_text = self.output_var.get().strip().strip('"')
        mode = self.mode_var.get()
        model = Path(model_text).expanduser()
        source = Path(source_text).expanduser()
        if not model_text or not model.is_file() or model.suffix.lower() != ".pt":
            raise ValueError("Select an existing .pt model file.")
        if not source_text or not (source.is_file() if mode == "Dataset YAML" else source.is_dir()):
            raise ValueError("Select an existing dataset YAML or image folder.")
        if mode == "Dataset YAML" and source.suffix.lower() not in (".yaml", ".yml"):
            raise ValueError("Choose a .yaml or .yml dataset configuration.")
        if not output_text:
            raise ValueError("Choose a results/cache directory.")
        try:
            precision = parse_target_precision(self.precision_var.get())
            cls = int(self.class_var.get().strip())
            imgsz = int(self.imgsz_var.get().strip())
            batch = int(self.batch_var.get().strip())
            match_iou = float(self.match_var.get().strip())
            nms_iou = float(self.nms_var.get().strip())
            max_det = int(self.maxdet_var.get().strip())
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid number: {exc}") from exc
        if cls < 0 or imgsz < 32 or batch < 1 or max_det < 1 or not (0 < match_iou <= 1) or not (0 < nms_iou <= 1):
            raise ValueError("Class >=0, size >=32, batch/max detections >=1, IoUs in (0, 1].")
        return dict(model=str(model.resolve()), mode=mode, source=str(source.resolve()),
                    output=str(Path(output_text).expanduser().resolve()),
                    split=self.split_var.get(), precision=precision, food_class=cls,
                    imgsz=imgsz, batch=batch, match_iou=match_iou,
                    nms_iou=nms_iou, max_det=max_det, device=self.device_var.get(),
                    agnostic_nms=self.agnostic_nms_var.get(),
                    reuse_cache=self.reuse_var.get(), save_plots=self.plots_var.get())

    def _start(self, cache_only):
        if self.busy:
            return
        try:
            settings = self._settings()
        except ValueError as exc:
            messagebox.showerror("Invalid settings", str(exc), parent=self.root)
            return
        self.cancel_event.clear()
        self.busy = True
        self.last_results_dir = None
        self.open_results_button.configure(state="disabled")
        self.view_report_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        for widget in self.inputs:
            widget.configure(state="disabled")
        self.status_var.set("Reading cache..." if cache_only else "Preparing predictions...")
        self.result_var.set("Processing...")
        self.progress.start(15)

        def worker():
            try:
                evaluate(settings, lambda kind, value: self.messages.put((kind, value)),
                         cache_only=cache_only, cancel_event=self.cancel_event)
            except EvaluationCancelled:
                self.messages.put(("cancelled", "Evaluation stopped by user."))
            except Exception as exc:
                self.messages.put(("error", str(exc), traceback.format_exc()))
        threading.Thread(target=worker, name="RecallPrecisionCached", daemon=True).start()

    def cancel(self):
        """Ask the worker to stop at the next safe checkpoint."""
        if self.busy and not self.cancel_event.is_set():
            self.cancel_event.set()
            self.stop_button.configure(state="disabled")
            self.status_var.set("Stopping evaluation (finishing current inference step)…")
            self._append_log("Stop requested; waiting for current inference step to finish.")

    def _open_results_folder(self):
        if not self.last_results_dir:
            return
        try:
            open_results_folder(self.last_results_dir)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Cannot open results folder", str(exc), parent=self.root)

    def _close_standalone(self):
        if self.busy:
            if not messagebox.askyesno("Evaluation running",
                    "Stop evaluation and close after the current inference step?",
                    parent=self.root):
                return
            self.cancel()
            self._close_after_stop()
        else:
            self.root.destroy()

    def _close_after_stop(self):
        if self.busy:
            self.root.after(200, self._close_after_stop)
        else:
            self.root.destroy()

    def _append_log(self, message):
        if self.on_log is not None:
            self.on_log(str(message))
        self.log.configure(state="normal")
        self.log.insert("end", str(message) + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")
        print("[CV Waste Recall Test] " + str(message), flush=True)

    def _poll(self):
        try:
            while True:
                message = self.messages.get_nowait()
                kind = message[0]
                value = message[1]
                if kind == "log":
                    self._append_log(value)
                elif kind == "status":
                    self.status_var.set(value)
                elif kind == "done":
                    self._finish()
                    self._show_result(value)
                    self._refresh_previous_runs()
                elif kind == "cancelled":
                    self._finish()
                    self.status_var.set("Evaluation stopped.")
                    self.result_var.set("Cancelled — no partial evaluation result saved.")
                    self._append_log(str(value))
                elif kind == "error":
                    self._finish()
                    details = message[2] if len(message) > 2 else str(value)
                    self._append_log(details)
                    self.status_var.set("Evaluation failed — inspect log.")
                    self.result_var.set(str(value))
                    messagebox.showerror("Recall evaluation failed", str(value), parent=self.root)
        except queue.Empty:
            pass
        if self.root.winfo_exists():
            self.root.after(180, self._poll)

    def _finish(self):
        self.busy = False
        self.progress.stop()
        self.stop_button.configure(state="disabled")
        for widget in self.inputs:
            widget.configure(state="normal")
        self.mode_choice.configure(state="readonly")
        self.split_choice.configure(state="readonly" if self.mode_var.get() == "Dataset YAML" else "disabled")
        for widget in self.inputs:
            if isinstance(widget, ttk.Combobox):
                if widget is self.split_choice and self.mode_var.get() != "Dataset YAML":
                    widget.configure(state="disabled")
                else:
                    widget.configure(state="readonly")


def main():
    root = tk.Tk()
    RecallTestGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
