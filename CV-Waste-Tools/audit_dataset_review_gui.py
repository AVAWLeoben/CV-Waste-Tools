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


def audit_dataset(model_path, image_dir, label_dir, output_dir,
                  conf_threshold, iou_threshold, allow_low_coverage,
                  cancel_event, emit):
    """Audit labeled images only; never invent missing-label findings from absent GT.

    Without explicit override, reject low annotation coverage before loading YOLO.
    Emits the original web-compatible CSV schema.
    """
    images = sorted(
        image_dir / filename
        for filename in os.listdir(image_dir)
        if (image_dir / filename).is_file()
        and Path(filename).suffix.lower() in IMAGE_EXTENSIONS
    )
    emit(("images", len(images)))
    if not images:
        raise ValueError(f"No top-level .jpg/.jpeg/.png images in {image_dir}")

    # Preflight: check the exact filenames and the actual content, not only folder existence.
    from collections import Counter
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
            annotations[image_path] = (boxes, classes)

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
        emit(("log", "NOTE: save_conf=True labels were recognized. "
                     "Saved confidences are ignored during ground-truth matching; "
                     "model-generated labels should still be manually verified."))
    for status, path in examples.items():
        emit(("log", f"Example {status}: {path}"))

    if valid == 0:
        raise ValueError(
            "No readable annotations were found. Verify the LABEL DIRECTORY and the "
            "YOLO .txt filenames. The audit was stopped before inference."
        )
    if coverage < 0.80 and not allow_low_coverage:
        raise ValueError(
            f"Only {valid}/{total} images ({coverage:.1%}) have readable annotations. "
            f"Missing files: {counts['MISSING_FILE']}; empty files: "
            f"{counts['EMPTY_FILE']}; invalid files: {counts['INVALID_FILE']}. "
            "The selected label folder may be incorrect or annotations may have "
            "another filename/format. No audit was run. If incomplete coverage is "
            "intentional, check 'Allow low label coverage' to analyze only the "
            "images that DO have readable ground truth."
        )
    skipped = total - valid
    if skipped:
        emit(("log", f"Skipping {skipped} images without readable GT: they will "
                     "NOT be reported as missing-object findings."))

    if cancel_event.is_set():
        emit(("cancelled",))
        return

    from ultralytics import YOLO
    emit(("log", f"Loading model: {model_path}"))
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
    for image_no, image_path in enumerate(images, start=1):
        if cancel_event.is_set():
            emit(("cancelled",))
            return
        if image_path not in annotations:
            emit(("progress", image_no, total, len(findings)))
            continue

        result = model.predict(
            str(image_path), conf=conf_threshold, imgsz=1280, verbose=False
        )[0]
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
                # There IS an annotation: review extent/position, not missing label.
                gt_cls = int(gt_classes[best_gt_idx])
                gt_box = gt_boxes[best_gt_idx]
                issue = "MISSING_OR_BAD_BOX"
            else:
                # This is only a candidate missing object: image has other GT boxes.
                # A high-confidence incorrect model detection can produce this too.
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

        emit(("progress", image_no, total, len(findings)))
        if image_no % 100 == 0:
            emit(("log", f"Audited {image_no}/{total} images; "
                         f"{len(findings)} findings"))

    if cancel_event.is_set():
        emit(("cancelled",))
        return

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
    emit(("log", f"Processed {valid} labeled images; skipped {skipped} "
                 "images without readable annotations."))
    emit(("complete", total, len(findings), unique_images,
          str(output_csv), elapsed))


class AuditGUI:
    def __init__(self, root):
        self.root = root
        root.title("YOLO Dataset Annotation Audit — Label-verified")
        root.geometry("950x580")
        root.minsize(800, 520)

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

        ttk.Label(
            container, textvariable=self.csv_name_var,
        ).grid(row=7, column=0, columnspan=3, sticky="w", pady=(2, 14))

        actions = ttk.Frame(container)
        actions.grid(row=8, column=0, columnspan=3, sticky="ew")
        self.start_button = ttk.Button(actions, text="Start audit", command=self.start)
        self.start_button.pack(side="left")
        self.cancel_button = ttk.Button(actions, text="Cancel", command=self.cancel, state="disabled")
        self.cancel_button.pack(side="left", padx=(9, 0))

        ttk.Label(container, textvariable=self.status_var).grid(
            row=9, column=0, columnspan=3, sticky="w", pady=(16, 5)
        )
        self.progress = ttk.Progressbar(container, orient="horizontal", mode="determinate")
        self.progress.grid(row=10, column=0, columnspan=3, sticky="ew")

        ttk.Label(container, text="Activity log").grid(
            row=11, column=0, columnspan=3, sticky="w", pady=(16, 5)
        )
        log_frame = ttk.Frame(container)
        log_frame.grid(row=12, column=0, columnspan=3, sticky="nsew")
        container.rowconfigure(12, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_box = tk.Text(log_frame, height=8, wrap="word", state="disabled")
        self.log_box.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(log_frame, command=self.log_box.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log_box.configure(yscrollcommand=scroll.set)

        root.protocol("WM_DELETE_WINDOW", self.on_close)
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
            f"Top-level .jpg/.jpeg/.png only  •  Inference size: 1280  •  CSV: {filename}"
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
        except ValueError:
            messagebox.showerror("Invalid thresholds", "Use numeric thresholds between 0 and 1.")
            return

        output_csv = output_dir / csv_filename_for(image_dir)
        if output_csv.exists() and not messagebox.askyesno(
            "Overwrite CSV?", f"This file already exists:\n{output_csv}\n\nOverwrite it when the audit completes?"
        ):
            return

        allow_low_coverage = self.allow_low_coverage_var.get()
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
                    self.progress.configure(value=completed)
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

        if self.closing and not self.busy:
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
