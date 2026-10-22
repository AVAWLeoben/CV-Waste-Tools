"""Tkinter GUI for auditing YOLO annotations.

Select a model, image folder, label folder and output folder. Only top-level
images are scanned. The tool verifies corresponding YOLO labels before inference,
supports BBOX/polygon annotations (including save_conf=True BBOX labels),
and skips images without readable labels.
Low label-file coverage is blocked unless the user explicitly overrides it.
Output remains compatible with the CV Waste Web REVIEW task.

Dependencies: pip install ultralytics numpy
Run: python audit_dataset_review_gui.py

No annotation files are modified.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
import queue
import re
import tempfile
import shutil
import subprocess
import threading
import time
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

def csv_filename_for(image_dir: Path) -> str:
    """Create a readable, portable CSV filename identifying the image folder."""
    folder = image_dir.name or "images"
    # Preserve meaningful Unicode while avoiding invalid Windows filename chars.
    folder = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", folder).strip(" .")
    folder = folder[:100].rstrip(" .") or "images"
    return f"suspicious_annotations_{folder}.csv"


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CSV_FIELDS = [
    "image", "issue", "gt_class", "gt_name", "pred_class", "pred_name",
    "confidence", "iou", "pred_index", "gt_index",
    "pred_x1", "pred_y1", "pred_x2", "pred_y2",
    "gt_x1", "gt_y1", "gt_x2", "gt_y2",
    "image_width", "image_height",
]


def box_iou(box, boxes):
    """Match the original audit script's bounding-box IoU calculation."""
    if len(boxes) == 0:
        return np.array([], dtype=float)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    area2 = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0, boxes[:, 3] - boxes[:, 1]
    )
    return intersection / (area1 + area2 - intersection + 1e-9)


def normalized_xyxy(box, image_width, image_height):
    """Normalize original-image pixel boxes to [0, 1], not model input size."""
    bounds = np.array([image_width, image_height, image_width, image_height], dtype=float)
    return [round(float(v), 6) for v in np.clip(np.asarray(box) / bounds, 0, 1)]


def read_yolo_annotations(label_path: Path, format_counts=None):
    """Load valid YOLO BBOX or segmentation labels, non-recursively.

    Returns (boxes_xyxy_normalized, class_ids, status, invalid_rows).
    Accepts 5-value YOLO BBOX and 6-value BBOX+confidence (save_conf=True).
    The stored model confidence is ignored for geometric matching.
    An empty .txt is an unannotated image, not a positive missing-object label.
    """
    if not label_path.is_file():
        return np.empty((0, 4)), np.empty(0, dtype=int), "MISSING_FILE", 0

    rows = label_path.read_text(encoding="utf-8-sig").splitlines()
    if not any(line.strip() for line in rows):
        return np.empty((0, 4)), np.empty(0, dtype=int), "EMPTY_FILE", 0

    boxes, classes = [], []
    invalid_rows = 0
    for line in rows:
        values = line.strip().split()
        if not values:
            continue
        try:
            class_id = int(values[0])
            coordinates = np.array([float(v) for v in values[1:]], dtype=float)
            if class_id < 0 or not np.isfinite(coordinates).all():
                raise ValueError("invalid class or coordinates")

            if len(values) in (5, 6):
                # Both standard YOLO labels and Ultralytics save_conf=True
                # predictions use class, xc, yc, w, h in the first 5 columns.
                # Column 6 is confidence metadata, not a box coordinate.
                if len(values) == 6:
                    saved_confidence = float(values[5])
                    if not (0 <= saved_confidence <= 1):
                        raise ValueError("invalid saved confidence")
                xc, yc, width, height = coordinates[:4]
                if width <= 0 or height <= 0:
                    raise ValueError("non-positive width/height")
                box = np.array([xc - width / 2, yc - height / 2,
                                xc + width / 2, yc + height / 2])
                label_format = "BBOX_CONF" if len(values) == 6 else "BBOX"
            elif len(coordinates) >= 6 and len(coordinates) % 2 == 0:
                # Segmentation polygon: use bounding rectangle for audit.
                points = coordinates.reshape(-1, 2)
                box = np.array([points[:, 0].min(), points[:, 1].min(),
                                points[:, 0].max(), points[:, 1].max()])
                label_format = "POLYGON"
            else:
                raise ValueError("unexpected number of coordinates")

            if (box[0] >= box[2] or box[1] >= box[3]
                    or np.any(box < -0.001) or np.any(box > 1.001)):
                raise ValueError("invalid normalized bounding box")
            boxes.append(np.clip(box, 0, 1))
            classes.append(class_id)
            if format_counts is not None:
                format_counts[label_format] += 1
        except (ValueError, IndexError):
            invalid_rows += 1

    if not boxes:
        return np.empty((0, 4)), np.empty(0, dtype=int), "INVALID_FILE", invalid_rows
    return (np.asarray(boxes, dtype=float).reshape(-1, 4),
            np.asarray(classes, dtype=int), "VALID", invalid_rows)


def select_inference_device(requested: str, emit) -> str:
    """Prefer real CUDA for Auto, and never silently use CPU on an NVIDIA host."""
    import torch

    requested = str(requested).strip()
    if requested == "CPU":
        emit(("log", "Inference device: CPU (explicit selection)."))
        return "cpu"

    if torch.cuda.is_available():
        try:
            # Verify that installed PyTorch kernels actually run on this GPU.
            gpu_name = torch.cuda.get_device_name(0)
            check = torch.ones(2, device="cuda:0").sum().item()
            if check != 2.0:
                raise RuntimeError("CUDA tensor computation did not return the expected result")
        except Exception as exc:
            raise RuntimeError(
                f"CUDA hardware was found, but the installed PyTorch cannot compute on it: {exc}. "
                "Recheck/repair the environment in the CV Waste Tools launcher."
            ) from exc
        emit(("log", f"Inference device: GPU 0 ({gpu_name})"))
        return "0"

    # If NVIDIA hardware is visible, treat a CPU-only PyTorch install as an
    # environment problem, even when the user selected Auto.
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            probe = subprocess.run(
                [smi, "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=8, check=False,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            if probe.returncode == 0 and probe.stdout.strip():
                raise RuntimeError(
                    f"NVIDIA GPU detected ({probe.stdout.splitlines()[0].strip()}), "
                    "but this Conda environment has no usable CUDA-enabled PyTorch. "
                    "Use 'Recheck / repair environment' in the launcher, then rerun the audit. "
                    "To use the CPU intentionally, select CPU in the audit GUI."
                )
        except subprocess.TimeoutExpired:
            pass
    if requested == "GPU 0":
        raise RuntimeError(
            "GPU 0 was requested, but PyTorch reports CUDA unavailable. "
            "Recheck the launcher environment and GPU driver."
        )
    emit(("log", "No usable NVIDIA CUDA GPU found. Inference device: CPU (Auto)."))
    return "cpu"


def build_source_manifest(paths: list[Path], directory: Path) -> Path:
    """Write paths for Ultralytics LoadImagesAndVideos, avoiding eager list loading.

    A Python list handed to model.predict() may load every image as one batch,
    ignoring the batch option. A text manifest uses the proper batched loader.
    """
    target = directory / "audit_images.txt"
    target.write_text("".join(f"{p.resolve()}\n" for p in paths), encoding="utf-8")
    return target


def audit_dataset(model_path, image_dir, label_dir, output_dir,
                  conf_threshold, iou_threshold, allow_low_coverage,
                  cancel_event, emit, imgsz=1280, batch_size=8, device="Auto (prefer GPU)"):
    """Audit labeled top-level images via ONE batched Ultralytics predictor.

    The model is loaded once, and model.predict() is called once with a
    .txt manifest of only the eligible (labeled) top-level images. Ultralytics
    then loads image batches from disk rather than starting a new predictor for
    every file. Per-image CSV rows remain backward-compatible with CV Waste.
    """
    from collections import Counter

    images = sorted(
        image_dir / filename
        for filename in os.listdir(image_dir)
        if (image_dir / filename).is_file()
        and Path(filename).suffix.lower() in IMAGE_EXTENSIONS
    )
    emit(("images", len(images)))
    if not images:
        raise ValueError(f"No top-level .jpg/.jpeg/.png images in {image_dir}")

    counts = Counter()
    examples = {}
    annotations = {}
    bad_rows = 0
    format_counts = Counter()
    for image_path in images:
        if cancel_event.is_set():
            emit(("cancelled",))
            return
        label_path = label_dir / f"{image_path.stem}.txt"
        boxes, classes, status, invalid_rows = read_yolo_annotations(
            label_path, format_counts=format_counts
        )
        counts[status] += 1
        bad_rows += invalid_rows
        if status != "VALID" and status not in examples:
            examples[status] = str(label_path)
        if status == "VALID":
            annotations[image_path.resolve()] = (boxes, classes)

    total = len(images)
    valid = counts["VALID"]
    coverage = valid / total
    emit(("log", f"PREFLIGHT: {total} images; {valid} with valid annotations "
                 f"({coverage:.1%}); {counts['MISSING_FILE']} missing .txt; "
                 f"{counts['EMPTY_FILE']} empty .txt; "
                 f"{counts['INVALID_FILE']} invalid .txt; "
                 f"{bad_rows} ignored invalid annotation rows."))
    emit(("log", f"LABEL FORMATS: standard BBOX={format_counts['BBOX']}; "
                 f"BBOX+confidence={format_counts['BBOX_CONF']}; "
                 f"segmentation polygons={format_counts['POLYGON']}."))
    if format_counts['BBOX_CONF']:
        emit(("log", "NOTE: save_conf=True labels were recognized; saved confidences "
                     "are ignored for geometry matching."))
    for status, path in examples.items():
        emit(("log", f"Example {status}: {path}"))

    if valid == 0:
        raise ValueError(
            "No readable annotations were found. Verify the LABEL DIRECTORY and YOLO filenames."
        )
    if coverage < 0.80 and not allow_low_coverage:
        raise ValueError(
            f"Only {valid}/{total} images ({coverage:.1%}) have readable annotations. "
            f"Missing files: {counts['MISSING_FILE']}; empty files: "
            f"{counts['EMPTY_FILE']}; invalid files: {counts['INVALID_FILE']}. "
            "If excluded background images explain this, enable 'Skip unannotated/background images'."
        )
    skipped = total - valid
    if skipped:
        emit(("log", f"Skipping {skipped} images without readable ground-truth labels."))
    eligible_images = sorted(annotations)
    emit(("images", valid))
    if cancel_event.is_set():
        emit(("cancelled",))
        return

    from ultralytics import YOLO
    selected_device = select_inference_device(device, emit)
    emit(("log", f"Loading YOLO model: {model_path}"))
    model = YOLO(str(model_path))
    model_names = model.names

    def get_class_name(class_id):
        if class_id < 0:
            return ""
        if isinstance(model_names, dict):
            name = model_names.get(class_id, model_names.get(str(class_id)))
        else:
            name = model_names[class_id] if class_id < len(model_names) else None
        return str(name) if name is not None else f"Class {class_id}"

    findings = []
    started = time.perf_counter()
    processed = set()
    emit(("log", f"Starting one streaming batched prediction run: {valid} labeled images; "
                 f"imgsz={imgsz}, batch={batch_size}, conf={conf_threshold:g}, "
                 f"device={selected_device}."))
    with tempfile.TemporaryDirectory(prefix="cvw_audit_manifest_") as tmp:
        source = build_source_manifest(eligible_images, Path(tmp))
        # Ultralytics handles the batch preprocessing, inference, and NMS.
        # The .txt source is essential: unlike a Python list, it enforces batch_size.
        stream = model.predict(
            source=str(source), stream=True, batch=batch_size,
            conf=conf_threshold, imgsz=imgsz, device=selected_device,
            verbose=False, save=False,
        )
        try:
            for result in stream:
                if cancel_event.is_set():
                    emit(("cancelled",))
                    return

                image_path = Path(result.path).resolve()
                if image_path not in annotations:
                    raise RuntimeError(
                        f"Prediction returned an unexpected image: {image_path}. "
                        "Aborting rather than associating boxes with the wrong ground truth."
                    )
                if image_path in processed:
                    raise RuntimeError(f"Duplicate prediction result for {image_path}")
                processed.add(image_path)

                height, width = result.orig_shape
                gt_norm, gt_classes = annotations[image_path]
                scale = np.array([width, height, width, height], dtype=float)
                gt_boxes = gt_norm * scale

                for pred_idx, pred in enumerate(result.boxes):
                    pred_box = pred.xyxy[0].cpu().numpy()
                    pred_cls = int(pred.cls[0].item())
                    confidence = float(pred.conf[0].item())
                    overlaps = box_iou(pred_box, gt_boxes)
                    best_gt_idx = int(np.argmax(overlaps))
                    best_iou = float(overlaps[best_gt_idx])

                    gt_cls = -1
                    gt_box = None
                    if best_iou >= iou_threshold:
                        gt_cls = int(gt_classes[best_gt_idx])
                        gt_box = gt_boxes[best_gt_idx]
                        if pred_cls == gt_cls:
                            continue
                        issue = "CLASS_MISMATCH"
                    elif best_iou > 0:
                        gt_cls = int(gt_classes[best_gt_idx])
                        gt_box = gt_boxes[best_gt_idx]
                        issue = "MISSING_OR_BAD_BOX"
                    else:
                        best_gt_idx = -1
                        issue = "POSSIBLE_MISSING_LABEL"

                    px1, py1, px2, py2 = normalized_xyxy(pred_box, width, height)
                    gx1, gy1, gx2, gy2 = (
                        normalized_xyxy(gt_box, width, height)
                        if gt_box is not None else ("", "", "", "")
                    )
                    findings.append({
                        "image": str(image_path),
                        "issue": issue,
                        "gt_class": gt_cls,
                        "gt_name": get_class_name(gt_cls),
                        "pred_class": pred_cls,
                        "pred_name": get_class_name(pred_cls),
                        "confidence": round(confidence, 4),
                        "iou": round(best_iou, 4),
                        "pred_index": pred_idx,
                        "gt_index": best_gt_idx,
                        "pred_x1": px1, "pred_y1": py1, "pred_x2": px2, "pred_y2": py2,
                        "gt_x1": gx1, "gt_y1": gy1, "gt_x2": gx2, "gt_y2": gy2,
                        "image_width": width, "image_height": height,
                    })

                count = len(processed)
                emit(("progress", count, valid, len(findings)))
                if count % 100 == 0 or count == valid:
                    emit(("log", f"Audited {count}/{valid} labeled images; "
                                 f"{len(findings)} findings"))
        finally:
            # Close on cancellation so temporary files can be deleted on Windows.
            if hasattr(stream, "close"):
                stream.close()

    if cancel_event.is_set():
        emit(("cancelled",))
        return
    if len(processed) != valid:
        raise RuntimeError(
            f"Ultralytics returned only {len(processed)}/{valid} results. "
            "No CSV was written; please check the prediction source and image formats."
        )

    findings.sort(key=lambda row: (row["issue"] != "CLASS_MISMATCH",
                                   -row["confidence"]))

    output_csv = output_dir / csv_filename_for(image_dir)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8-sig", dir=output_dir,
            prefix=".suspicious_annotations_", suffix=".tmp", delete=False
        ) as handle:
            temp_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(findings)
        if cancel_event.is_set():
            emit(("cancelled",))
            return
        os.replace(temp_path, output_csv)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    elapsed = time.perf_counter() - started
    unique_images = len({row["image"] for row in findings})
    emit(("log", f"Completed {len(processed)} labeled images; skipped {skipped} "
                 f"without readable annotations. Device: {selected_device}; "
                 f"elapsed: {elapsed:.1f}s"))
    emit(("complete", valid, len(findings), unique_images,
          str(output_csv), elapsed))


class AuditGUI:
    def __init__(self, root, *, embedded=False, on_log=None):
        self.embedded = embedded
        self.on_log = on_log
        self.root = root.winfo_toplevel()
        if not embedded:
            self.root.title("YOLO Dataset Annotation Audit — Label-verified")
            self.root.geometry("1050x650")
            self.root.minsize(890, 590)

        self.messages = queue.Queue()
        self.cancel_event = threading.Event()
        self.worker = None
        self.closing = False
        self.total_images = 0
        self.busy = False

        self.model_var = tk.StringVar()
        self.image_var = tk.StringVar()
        self.label_var = tk.StringVar()
        self.output_var = tk.StringVar(value=str(Path.cwd()))
        self.conf_var = tk.StringVar(value="0.80")
        self.iou_var = tk.StringVar(value="0.50")
        self.imgsz_var = tk.StringVar(value="1280")
        self.batch_var = tk.StringVar(value="8")
        self.device_var = tk.StringVar(value="Auto (prefer GPU)")
        self.allow_low_coverage_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Ready. Select the folders and start the audit.")
        self.csv_name_var = tk.StringVar()
        self.image_var.trace_add("write", self._update_csv_filename)
        self._update_csv_filename()

        container = ttk.Frame(root, padding=18)
        container.pack(fill="both", expand=True)
        container.columnconfigure(1, weight=1)

        ttk.Label(container, text="YOLO Annotation Audit", font=("Segoe UI", 16, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 5)
        )
        ttk.Label(container, text="Choose a model and input folders. Labels are read-only.").grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(0, 17)
        )

        self.inputs = []
        self._path_row(container, 2, "YOLO model", self.model_var, self._choose_model)
        self._path_row(container, 3, "Image directory", self.image_var, self._choose_images)
        self._path_row(
            container, 4, "Label directory", self.label_var, self._choose_labels,
            same_as_images=True,
        )
        self._path_row(container, 5, "Output directory", self.output_var, self._choose_output)

        options = ttk.Frame(container)
        options.grid(row=6, column=0, columnspan=3, sticky="w", pady=(15, 4))
        ttk.Label(options, text="Confidence threshold").pack(side="left")
        conf_entry = ttk.Entry(options, width=8, textvariable=self.conf_var)
        conf_entry.pack(side="left", padx=(8, 28))
        ttk.Label(options, text="IoU threshold").pack(side="left")
        iou_entry = ttk.Entry(options, width=8, textvariable=self.iou_var)
        iou_entry.pack(side="left", padx=(8, 0))
        self.inputs.extend([conf_entry, iou_entry])
        coverage_box = ttk.Checkbutton(
            options,
            text="Skip unannotated/background images (audit labeled only)",
            variable=self.allow_low_coverage_var,
        )
        coverage_box.pack(side="left", padx=(25, 0))
        self.inputs.append(coverage_box)

        inference_options = ttk.Frame(container)
        inference_options.grid(row=7, column=0, columnspan=3, sticky="w", pady=(8, 3))
        ttk.Label(inference_options, text="Image size (imgsz)").pack(side="left")
        image_size_entry = ttk.Combobox(
            inference_options, textvariable=self.imgsz_var,
            values=("640", "960", "1280", "1536"), width=7,
        )
        image_size_entry.pack(side="left", padx=(8, 24))
        ttk.Label(inference_options, text="Batch size").pack(side="left")
        batch_entry = ttk.Combobox(
            inference_options, textvariable=self.batch_var,
            values=("1", "2", "4", "8", "16", "32"), width=6,
        )
        batch_entry.pack(side="left", padx=(8, 24))
        ttk.Label(inference_options, text="Inference device").pack(side="left")
        device_entry = ttk.Combobox(
            inference_options, textvariable=self.device_var,
            values=("Auto (prefer GPU)", "GPU 0", "CPU"),
            state="readonly", width=20,
        )
        device_entry.pack(side="left", padx=(8, 0))
        self.inputs.extend([image_size_entry, batch_entry, device_entry])

        ttk.Label(
            container, textvariable=self.csv_name_var,
        ).grid(row=8, column=0, columnspan=3, sticky="w", pady=(2, 14))

        actions = ttk.Frame(container)
        actions.grid(row=9, column=0, columnspan=3, sticky="ew")
        self.start_button = ttk.Button(actions, text="Start audit", command=self.start)
        self.start_button.pack(side="left")
        self.cancel_button = ttk.Button(actions, text="Cancel", command=self.cancel, state="disabled")
        self.cancel_button.pack(side="left", padx=(9, 0))

        ttk.Label(container, textvariable=self.status_var).grid(
            row=10, column=0, columnspan=3, sticky="w", pady=(16, 5)
        )
        self.progress = ttk.Progressbar(container, orient="horizontal", mode="determinate")
        self.progress.grid(row=11, column=0, columnspan=3, sticky="ew")

        if not embedded:
            ttk.Label(container, text="Activity log").grid(
                row=12, column=0, columnspan=3, sticky="w", pady=(16, 5)
            )
        log_frame = ttk.Frame(container)
        log_frame.grid(row=13, column=0, columnspan=3, sticky="nsew")
        container.rowconfigure(13, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_box = tk.Text(log_frame, height=8, wrap="word", state="disabled")
        self.log_box.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(log_frame, command=self.log_box.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log_box.configure(yscrollcommand=scroll.set)

        if embedded:
            # Keep per-tool logs in memory/GUI for compatibility but show the
            # consolidated activity console rather than a second log pane.
            log_frame.grid_remove()
            container.rowconfigure(13, weight=0)
        else:
            self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self.poll_messages)

    def _path_row(self, parent, row, label, variable, chooser, same_as_images=False):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 15), pady=5)
        field = ttk.Entry(parent, textvariable=variable)
        field.grid(row=row, column=1, sticky="ew", pady=5)
        controls = ttk.Frame(parent)
        controls.grid(row=row, column=2, sticky="e", padx=(10, 0), pady=5)
        browse_button = ttk.Button(controls, text="Browse…", width=12, command=chooser)
        browse_button.pack(side="left")
        self.inputs.extend([field, browse_button])
        if same_as_images:
            same_button = ttk.Button(
                controls, text="Same As Image Dir", command=self._use_image_dir_for_labels
            )
            same_button.pack(side="left", padx=(8, 0))
            self.inputs.append(same_button)

    def _update_csv_filename(self, *_):
        image_dir = self.image_var.get().strip().strip('"')
        filename = (csv_filename_for(Path(image_dir)) if image_dir
                    else "suspicious_annotations_<image_folder>.csv")
        self.csv_name_var.set(
            f"Top-level images only · Batched Ultralytics prediction · CSV: {filename}"
        )

    def _use_image_dir_for_labels(self):
        image_dir = self.image_var.get().strip()
        if not image_dir:
            messagebox.showinfo("Image directory needed", "Select an image directory first.")
            return
        self.label_var.set(image_dir)

    def _choose_model(self):
        chosen = filedialog.askopenfilename(
            title="Select YOLO model",
            filetypes=[("YOLO model files", "*.pt *.onnx *.engine *.torchscript *.tflite"),
                       ("All files", "*.*")],
        )
        if chosen:
            self.model_var.set(chosen)

    def _choose_images(self):
        chosen = filedialog.askdirectory(title="Select image directory", mustexist=True)
        if chosen:
            self.image_var.set(chosen)

    def _choose_labels(self):
        chosen = filedialog.askdirectory(title="Select YOLO label directory", mustexist=True)
        if chosen:
            self.label_var.set(chosen)

    def _choose_output(self):
        chosen = filedialog.askdirectory(title="Select output directory", mustexist=True)
        if chosen:
            self.output_var.set(chosen)

    def log(self, text):
        if self.on_log is not None:
            self.on_log(str(text))
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def set_busy(self, busy):
        self.busy = busy
        for widget in self.inputs:
            widget.configure(state="disabled" if busy else "normal")
        self.start_button.configure(state="disabled" if busy else "normal")
        self.cancel_button.configure(state="normal" if busy else "disabled")

    def start(self):
        if self.busy:
            return
        model_path = Path(self.model_var.get().strip().strip('"'))
        image_dir = Path(self.image_var.get().strip().strip('"'))
        label_dir = Path(self.label_var.get().strip().strip('"'))
        output_dir = Path(self.output_var.get().strip().strip('"'))

        # Check empty strings separately because Path("") becomes Path(".").
        if not self.model_var.get().strip() or not model_path.is_file():
            messagebox.showerror("Invalid model", "Select an existing YOLO model file.")
            return
        for name, raw, directory in (
            ("image", self.image_var.get(), image_dir),
            ("label", self.label_var.get(), label_dir),
            ("output", self.output_var.get(), output_dir),
        ):
            if not raw.strip() or not directory.is_dir():
                messagebox.showerror("Invalid directory", f"Select an existing {name} directory.")
                return

        try:
            confidence = float(self.conf_var.get())
            iou = float(self.iou_var.get())
            if not (0 <= confidence <= 1 and 0 <= iou <= 1):
                raise ValueError
            if not np.isfinite([confidence, iou]).all():
                raise ValueError
            imgsz = int(self.imgsz_var.get().strip())
            batch_size = int(self.batch_var.get().strip())
            if imgsz < 32 or imgsz > 4096 or batch_size < 1 or batch_size > 256:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid settings", "Thresholds must be 0–1, "
                                 "image size 32–4096, and batch size 1–256.")
            return

        output_csv = output_dir / csv_filename_for(image_dir)
        if output_csv.exists() and not messagebox.askyesno(
            "Overwrite CSV?", f"This file already exists:\n{output_csv}\n\nOverwrite it when the audit completes?"
        ):
            return

        allow_low_coverage = self.allow_low_coverage_var.get()
        selected_device = self.device_var.get()
        self.cancel_event.clear()
        self.progress.configure(mode="indeterminate", value=0)
        self.progress.start(12)
        self.status_var.set("Loading YOLO model ...")
        self.log(f"--- New audit ---")
        self.set_busy(True)

        def run():
            try:
                audit_dataset(
                    model_path, image_dir, label_dir, output_dir,
                    confidence, iou, allow_low_coverage,
                    self.cancel_event, self.messages.put,
                    imgsz=imgsz, batch_size=batch_size,
                    device=selected_device,
                )
            except Exception as exc:
                self.messages.put(("error", str(exc), traceback.format_exc()))

        self.worker = threading.Thread(target=run, name="YOLOAuditWorker", daemon=True)
        self.worker.start()

    def cancel(self):
        if self.busy:
            self.cancel_event.set()
            self.cancel_button.configure(state="disabled")
            self.status_var.set("Cancelling after the current image ...")
            self.log("Cancellation requested; existing CSV will be kept.")

    def poll_messages(self):
        try:
            while True:
                event = self.messages.get_nowait()
                kind = event[0]
                if kind == "log":
                    self.log(event[1])
                elif kind == "images":
                    self.total_images = event[1]
                    self.progress.stop()
                    self.progress.configure(mode="determinate", maximum=max(1, self.total_images), value=0)
                    self.status_var.set(f"Found {self.total_images} images. Auditing ...")
                elif kind == "progress":
                    _, completed, total, findings = event
                    self.progress.configure(maximum=max(1, total), value=completed)
                    if not self.cancel_event.is_set():
                        self.status_var.set(f"Audited {completed}/{total} images · {findings} findings")
                elif kind == "complete":
                    _, num_images, num_findings, unique_images, path, seconds = event
                    self.set_busy(False)
                    self.progress.stop()
                    self.progress.configure(mode="determinate", value=max(1, num_images))
                    self.status_var.set(f"Completed: {num_images} images, {num_findings} findings")
                    self.log(f"Analyzed images: {num_images}")
                    self.log(f"Suspicious cases: {num_findings}")
                    self.log(f"Images with findings: {unique_images}")
                    self.log(f"Saved CSV: {path}")
                    self.log(f"Elapsed time: {seconds:.1f} s")
                    if not self.closing:
                        messagebox.showinfo("Audit completed", f"Saved {num_findings} findings to:\n{path}")
                elif kind == "cancelled":
                    self.set_busy(False)
                    self.progress.stop()
                    self.progress.configure(mode="determinate")
                    self.status_var.set("Cancelled. No new CSV was saved.")
                    self.log("Audit cancelled. No new CSV was saved.")
                elif kind == "error":
                    self.set_busy(False)
                    self.progress.stop()
                    self.progress.configure(mode="determinate")
                    self.status_var.set("Audit failed. Check the log for details.")
                    self.log(event[2])
                    if not self.closing:
                        messagebox.showerror("Audit failed", event[1] or "Unknown error")
        except queue.Empty:
            pass

        if self.closing and not self.busy and not self.embedded:
            self.root.destroy()
        else:
            self.root.after(100, self.poll_messages)

    def on_close(self):
        if not self.busy:
            self.root.destroy()
        elif messagebox.askyesno("Cancel audit?", "Cancel the running audit and close the window?"):
            self.closing = True
            self.cancel()


def main():
    root = tk.Tk()
    try:
        ttk.Style(root).theme_use("clam")
    except tk.TclError:
        pass
    AuditGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
