# -*- coding: utf-8 -*-
"""Headless YOLO bounding-box pre-annotation for CV Waste Studio.

Existing labels are preserved by default. Model class IDs are written as-is;
pretrained COCO IDs must be remapped before use with a different taxonomy.
"""
from __future__ import annotations

import math
import os
import re
from pathlib import Path
import tempfile

IMAGE_SUFFIXES = frozenset({".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png",
                            ".tif", ".tiff", ".webp", ".pfm", ".heic"})


def annotation_paths(dataset_folder, labels_folder=None):
    """Resolve image directory and YOLO labels root without moving images."""
    source = Path(dataset_folder).expanduser().resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"Image/dataset folder does not exist: {source}")
    if (source / "images").is_dir():
        images = source / "images"
        default_labels = source / "labels"
    elif source.name.lower() == "images":
        images = source
        default_labels = source.parent / "labels"
    else:
        images = source
        # For mixed image/label folders, produce labels next to images.
        default_labels = source
    labels = (Path(labels_folder).expanduser().resolve() if labels_folder else default_labels)
    if images == labels and images.name.lower() == "images":
        raise ValueError("The label output must not be the images directory for a YOLO images/labels dataset.")
    return images, labels


def _settings(dataset_folder, model, conf, iou, max_det, imgsz, device,
              labels_folder, overwrite):
    images, labels = annotation_paths(dataset_folder, labels_folder)
    model = str(model).strip()
    if not model:
        raise ValueError("Select YOLO pretrained weights or a trained .pt model.")
    # Recognised pretrained model names may be downloaded automatically by
    # Ultralytics. Anything else must refer to an existing local weight file.
    pretrained = re.fullmatch(r"(?:yolov8|yolo11|yolo26)[nsmxl]\.pt", model, re.I)
    if not pretrained:
        model_file = Path(model).expanduser()
        if model_file.suffix.lower() != ".pt" or not model_file.is_file():
            raise FileNotFoundError(f"Select an existing YOLO .pt weights file: {model_file}")
        model = str(model_file.resolve())
    try:
        conf, iou = float(conf), float(iou)
        max_det, imgsz = int(max_det), int(imgsz)
    except (TypeError, ValueError) as exc:
        raise ValueError("Confidence, IoU, max detections and image size must be numeric.") from exc
    if not math.isfinite(conf) or not (0 < conf <= 1):
        raise ValueError("Confidence must be between 0 (exclusive) and 1.")
    if not math.isfinite(iou) or not (0 <= iou <= 1):
        raise ValueError("IoU must be between 0 and 1.")
    if max_det < 1 or imgsz < 32:
        raise ValueError("Max detections must be >= 1 and image size >= 32.")
    device = str(device).strip().lower()
    if device not in ("auto", "cpu", "0"):
        raise ValueError("Device must be auto, CPU or GPU 0.")
    return images, labels, model, conf, iou, max_det, imgsz, device, bool(overwrite)


def preannotate_dataset(dataset_folder, model, conf=0.25, iou=0.7, max_det=300,
                        imgsz=1280, device="auto", labels_folder=None,
                        overwrite=False):
    """Predict boxes for images and write normalized YOLO class/x/y/w/h lines.

    Skips already-labelled files by default. Writes every *new* label file,
    including empty files for images without detections, so they are not
    repeatedly inferred in later runs. Never creates a dataset split.
    """
    images_dir, labels_dir, model, conf, iou, max_det, imgsz, device, overwrite = _settings(
        dataset_folder, model, conf, iou, max_det, imgsz, device, labels_folder, overwrite
    )
    images = sorted((p for p in images_dir.rglob("*")
                     if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
                    key=lambda p: str(p).casefold())
    if not images:
        raise ValueError(f"No supported images found in {images_dir}")

    # Reject collisions such as image001.jpg + image001.png, which would map
    # to the same image001.txt and silently overwrite one prediction.
    names = set()
    for image in images:
        relative = image.relative_to(images_dir).with_suffix(".txt")
        key = str(relative).casefold()
        if key in names:
            raise ValueError(f"Two images would share the same label file: {relative}")
        names.add(key)

    print(f"Pre-annotation model: {model}", flush=True)
    print(f"Images: {images_dir} ({len(images)} files)", flush=True)
    print(f"YOLO labels: {labels_dir} | existing labels: {'overwrite' if overwrite else 'skip'}", flush=True)
    print(f"conf={conf:g} | iou={iou:g} | max_det={max_det} | imgsz={imgsz} | device={device}", flush=True)
    print("NOTE: Class IDs come from the selected model. Check the model's class names "
          "against your dataset YAML before training.", flush=True)

    # Lazy import: UI/tests do not need Ultralytics or torch installed.
    from ultralytics import YOLO
    detector = YOLO(model)
    if hasattr(detector, "task") and detector.task not in (None, "detect"):
        raise ValueError(f"Only object-detection YOLO models are supported (got {detector.task!r}).")
    print(f"Model classes: {getattr(detector, 'names', {})}", flush=True)
    predict_args = {"conf": conf, "iou": iou, "max_det": max_det,
                    "imgsz": imgsz, "verbose": False, "stream": True, "save": False}
    if device != "auto":
        predict_args["device"] = device

    saved = skipped = empty = detections = 0
    for i, image in enumerate(images, 1):
        label = labels_dir / image.relative_to(images_dir).with_suffix(".txt")
        if label.exists() and not overwrite:
            skipped += 1
            continue
        results = detector.predict(source=str(image), **predict_args)
        lines = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls = int(box.cls.item())
                xywh = box.xywhn[0].tolist()
                if (len(xywh) != 4 or
                        not all(math.isfinite(float(v)) for v in xywh)):
                    raise ValueError(f"Non-finite detection coordinates for {image}")
                vals = [min(1.0, max(0.0, float(v))) for v in xywh]
                lines.append(f"{cls} " + " ".join(f"{v:.6f}" for v in vals))
        label.parent.mkdir(parents=True, exist_ok=True)
        # Never leave an incomplete label when prediction or writing fails.
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                             suffix=".tmp", prefix=".cv_preannotate_",
                                             dir=str(label.parent), delete=False) as tmp:
                temp_path = Path(tmp.name)
                if lines:
                    tmp.write("\n".join(lines) + "\n")
            # Recheck file existence just before replacement; normal operations
            # use one worker and the GUI never edits labels concurrently.
            if label.exists() and not overwrite:
                skipped += 1
                temp_path.unlink(missing_ok=True)
            else:
                os.replace(temp_path, label)
                saved += 1
                empty += not bool(lines)
                detections += len(lines)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
        if i % 25 == 0 or i == len(images):
            print(f"Pre-annotation progress: {i}/{len(images)} images | "
                  f"created {saved} | skipped {skipped} | boxes {detections}", flush=True)
    print(f"PRE-ANNOTATION COMPLETE: {saved} labels written, {skipped} preserved, "
          f"{empty} empty, {detections} boxes", flush=True)
    return {"images": len(images), "created": saved, "skipped": skipped,
            "empty": empty, "detections": detections, "labels_dir": str(labels_dir)}
