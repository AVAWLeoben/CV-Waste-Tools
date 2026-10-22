# -*- coding: utf-8 -*-
"""CV Waste Tools: find the maximum Food recall at a requested precision.

Works with either an existing Ultralytics data.yaml/autosplit_data.yaml, or
an image folder in a standard YOLO layout (images/ + labels/, or paired
image.jpg/image.txt files in one folder). Image/annotation contents are not edited (Ultralytics may create label caches).

This analysis uses Ultralytics' *validation precision/recall curves*; its
object-level precision is not the same as industrial mass-based sorting purity.
"""
from __future__ import annotations

import csv
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import traceback

# Matplotlib chart creation inside a worker thread must not use TkAgg.
os.environ.setdefault("MPLBACKEND", "Agg")


def parse_target_precision(text: str) -> float:
    """Accept 95, 95%, 0.95, and comma decimals."""
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


def operating_point(box, food_class: int, target: float):
    """Use Ultralytics' per-class curves, accounting for absent class IDs.

    The row index in p_curve/r_curve need not be equal to the class ID when
    the validation dataset lacks some classes; ap_class_index maps rows to IDs.
    """
    import numpy as np

    thresholds = np.asarray(getattr(box, "px", []), dtype=float).reshape(-1)
    precision = np.asarray(getattr(box, "p_curve", []), dtype=float)
    recall = np.asarray(getattr(box, "r_curve", []), dtype=float)
    if precision.ndim == 1:
        precision = precision.reshape(1, -1)
    if recall.ndim == 1:
        recall = recall.reshape(1, -1)
    if not len(thresholds) or precision.ndim != 2 or recall.ndim != 2:
        raise RuntimeError("The validation result does not contain P/R confidence curves.")
    if (precision.shape != recall.shape or precision.shape[1] != len(thresholds)):
        raise RuntimeError("Ultralytics returned inconsistent P/R curves and confidence axis.")

    class_indices = getattr(box, "ap_class_index", None)
    if class_indices is not None and len(class_indices):
        class_indices = [int(x) for x in class_indices]
        if food_class not in class_indices:
            raise ValueError(
                f"Class {food_class} has no validation instances/curve. "
                f"Available class IDs: {class_indices}. Check the selected split and class ID."
            )
        row = class_indices.index(food_class)
    elif 0 <= food_class < precision.shape[0]:
        # Older Ultralytics releases may expose curves without class index info.
        row = food_class
    else:
        raise ValueError(f"No validation precision/recall curve for class {food_class}.")
    if row >= precision.shape[0]:
        raise RuntimeError("Ultralytics class-to-curve mapping is inconsistent.")

    p = precision[row]
    r = recall[row]
    finite = np.isfinite(p) & np.isfinite(r) & np.isfinite(thresholds)
    valid = np.flatnonzero(finite & (p >= target))
    result = {
        "target_precision": float(target), "class_id": food_class,
        "curve_conf": thresholds, "curve_precision": p, "curve_recall": r,
        "achieved": bool(len(valid)), "confidence": None,
        "precision": None, "recall": None,
    }
    if len(valid):
        # First max-recall point, matching the existing console script.
        index = int(valid[int(np.argmax(r[valid]))])
        result.update(confidence=float(thresholds[index]),
                      precision=float(p[index]), recall=float(r[index]))
    return result


def discover_dataset_yaml(folder: Path) -> Path | None:
    """Check common dataset configurations without recursively scanning huge data."""
    for name in ("autosplit_data.yaml", "data.yaml", "dataset.yaml",
                 "autosplit_data.yml", "data.yml", "dataset.yml"):
        file = folder / name
        if file.is_file():
            return file
    return None


def _class_names(model) -> dict[int, str]:
    names = model.names
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(name) for i, name in enumerate(names)}


def _make_image_folder_yaml(image_dir: Path, temp_dir: Path, model):
    """Build a small temporary YAML. Ultralytics finds corresponding label paths.

    images/A.jpg -> labels/A.txt, or mixed/A.jpg -> mixed/A.txt.
    Files are not copied, relinked or moved.
    """
    import yaml
    from ultralytics.data.utils import img2label_paths

    image_files = sorted(p for p in image_dir.iterdir()
                         if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})
    if not image_files:
        raise ValueError(f"No images found directly inside {image_dir}.")
    sample_labels = [Path(p) for p in img2label_paths([str(p) for p in image_files])]
    count_labels = sum(p.is_file() for p in sample_labels)
    if not count_labels:
        raise ValueError(
            "No corresponding YOLO labels were found. Expected images/A.jpg + "
            "labels/A.txt, or image.jpg + image.txt in the same folder. "
            "For a custom layout, select an existing data.yaml instead."
        )

    yaml_file = temp_dir / "evaluation_data.yaml"
    # Absolute paths keep the temporary YAML independent of source placement.
    dataset = {
        "path": str(image_dir.parent),
        "train": str(image_dir),
        "val": str(image_dir),
        "test": str(image_dir),
        "names": _class_names(model),
    }
    yaml_file.write_text(yaml.safe_dump(dataset, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return yaml_file, len(image_files), count_labels


def evaluate(settings: dict, emit):
    """Run evaluation in a worker; all Tk interactions occur on the main thread."""
    import numpy as np
    import yaml
    from ultralytics import YOLO

    model_file = Path(settings["model"])
    dest = Path(settings["output"])
    dest.mkdir(parents=True, exist_ok=True)
    name = "precision_recall_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    emit("status", "Loading YOLO model ...")
    emit("log", f"Loading model: {model_file}")
    model = YOLO(str(model_file))
    model_names = _class_names(model)
    cls = settings["food_class"]
    if cls not in model_names:
        raise ValueError(f"Class ID {cls} is not present in the model. Model classes: {model_names}")
    emit("log", f"Target class: {model_names[cls]} (ID {cls})")

    with tempfile.TemporaryDirectory(prefix="cvw_precision_data_") as tmp:
        if settings["mode"] == "Image folder":
            yaml_file, image_count, label_count = _make_image_folder_yaml(
                Path(settings["source"]), Path(tmp), model)
            emit("log", f"Direct image folder: {image_count} images; {label_count} label files found.")
            # The temporary file is only needed during model.val().
            split = "test"
        else:
            yaml_file = Path(settings["source"])
            dataset = yaml.safe_load(yaml_file.read_text(encoding="utf-8-sig"))
            if not isinstance(dataset, dict):
                raise ValueError(f"Invalid dataset YAML: {yaml_file}")
            split = settings["split"]
            if not dataset.get(split):
                raise ValueError(f"Dataset YAML does not contain a '{split}' split: {yaml_file}")
        emit("log", f"Dataset: {yaml_file} | split: {split}")
        emit("status", "Validating model and generating P/R curves ...")
        args = dict(data=str(yaml_file), split=split,
                    conf=0.001, plots=True, workers=settings["workers"],
                    batch=settings["batch"], imgsz=settings["imgsz"],
                    project=str(dest), name=name, exist_ok=True)
        if settings["device"] != "Auto":
            args["device"] = "0" if settings["device"] == "GPU 0" else "cpu"
        emit("log", "Running Ultralytics validation (confidence floor 0.001) ...")
        print(f"[CV Waste Recall Test] {model_file} | {yaml_file} | {split}", flush=True)
        metrics = model.val(**args)

    result = operating_point(metrics.box, cls, settings["precision"])
    output_folder = Path(getattr(metrics, "save_dir", dest / name))
    output_folder.mkdir(parents=True, exist_ok=True)

    curve_csv = output_folder / "food_precision_recall_by_confidence.csv"
    with curve_csv.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["confidence", "precision", "recall"])
        writer.writerows(zip(result["curve_conf"], result["curve_precision"], result["curve_recall"]))

    summary = {
        "model": str(model_file), "dataset": str(settings["source"]),
        "source_mode": settings["mode"], "split": split,
        "class_id": cls, "class_name": model_names[cls],
        "target_precision": result["target_precision"],
        "target_reached": result["achieved"],
        "confidence_threshold": result["confidence"],
        "achieved_precision": result["precision"],
        "recall_at_precision": result["recall"],
        "validation_conf_floor": 0.001,
        "output_directory": str(output_folder),
        "curve_csv": str(curve_csv),
    }
    (output_folder / "operating_point_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if result["achieved"]:
        emit("log", f"Confidence {result['confidence']:.4f} | "
                    f"Precision {result['precision']:.2%} | Recall {result['recall']:.2%}")
    else:
        emit("log", f"Requested precision {result['target_precision']:.1%} was not reached.")
    emit("log", f"Saved validation plots, curves and summary to: {output_folder}")
    emit("done", summary)


class RecallTestGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CV Waste Tools · Recall at Target Precision")
        self.root.geometry("900x735")
        self.root.minsize(760, 645)
        self.messages = queue.Queue()
        self.busy = False
        self.model_var = tk.StringVar()
        self.mode_var = tk.StringVar(value="Dataset YAML")
        self.source_var = tk.StringVar()
        self.split_var = tk.StringVar(value="test")
        self.output_var = tk.StringVar(value=str(Path.cwd() / "precision_recall_results"))
        self.precision_var = tk.StringVar(value="95")
        self.class_var = tk.StringVar(value="0")
        self.imgsz_var = tk.StringVar(value="640")
        self.batch_var = tk.StringVar(value="4")
        self.workers_var = tk.StringVar(value="0")
        self.device_var = tk.StringVar(value="Auto")
        self.status_var = tk.StringVar(value="Select a model and evaluation dataset.")
        self.result_var = tk.StringVar(value="No evaluation run yet.")
        self._build_ui()
        self.root.after(120, self._poll)

    def _build_ui(self):
        try:
            ttk.Style(self.root).theme_use("clam")
        except tk.TclError:
            pass
        frame = ttk.Frame(self.root, padding=18)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(13, weight=1)

        ttk.Label(frame, text="Recall at Target Precision", font=("Segoe UI", 17, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 5))
        ttk.Label(frame, text="Determine the maximum Food recall while enforcing a minimum detection precision.",
                  foreground="#555555").grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 16))

        self.inputs = []
        def entry_row(row, caption, variable, browse=None):
            ttk.Label(frame, text=caption).grid(row=row, column=0, sticky="w", padx=(0, 15), pady=6)
            widget = ttk.Entry(frame, textvariable=variable)
            widget.grid(row=row, column=1, sticky="ew", pady=6)
            self.inputs.append(widget)
            if browse:
                button = ttk.Button(frame, text="Browse...", command=browse)
                button.grid(row=row, column=2, sticky="ew", padx=(10, 0), pady=6)
                self.inputs.append(button)
            return widget

        entry_row(2, "YOLO model (.pt)", self.model_var, self._choose_model)
        mode_row = ttk.Frame(frame)
        mode_row.grid(row=3, column=0, columnspan=3, sticky="ew", pady=5)
        ttk.Label(mode_row, text="Dataset source").pack(side="left", padx=(0, 24))
        self.mode_choice = ttk.Combobox(mode_row, state="readonly", width=18,
                                        values=("Dataset YAML", "Image folder"),
                                        textvariable=self.mode_var)
        self.mode_choice.pack(side="left")
        self.inputs.append(self.mode_choice)
        self.mode_choice.bind("<<ComboboxSelected>>", lambda _event: self._mode_changed())
        self.source_label = ttk.Label(frame, text="Dataset YAML")
        self.source_label.grid(row=4, column=0, sticky="w", padx=(0, 15), pady=6)
        self.source_entry = ttk.Entry(frame, textvariable=self.source_var)
        self.source_entry.grid(row=4, column=1, sticky="ew", pady=6)
        self.source_browse = ttk.Button(frame, text="Browse...", command=self._choose_source)
        self.source_browse.grid(row=4, column=2, sticky="ew", padx=(10, 0), pady=6)
        self.inputs.extend((self.source_entry, self.source_browse))
        self.source_hint = ttk.Label(frame, text="Choose data.yaml or autosplit_data.yaml. The chosen split is resolved by Ultralytics.",
                                     foreground="#555555", wraplength=635)
        self.source_hint.grid(row=5, column=0, columnspan=2, sticky="w", pady=(0, 8))
        self.dataset_folder_button = ttk.Button(frame, text="Choose dataset folder...",
                                                 command=self._choose_dataset_folder)
        self.dataset_folder_button.grid(row=5, column=2, sticky="ew", padx=(10, 0), pady=(0, 8))
        self.inputs.append(self.dataset_folder_button)

        entry_row(6, "Save results in", self.output_var, self._choose_output)
        settings = ttk.Frame(frame)
        settings.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(9, 7))
        for col in range(6):
            settings.columnconfigure(col, weight=1)
        fields = [
            ("Target precision (%)", self.precision_var, 9),
            ("Food class ID", self.class_var, 7),
            ("Image size", self.imgsz_var, 7),
            ("Batch", self.batch_var, 5),
            ("Workers", self.workers_var, 5),
        ]
        for i, (caption, variable, width) in enumerate(fields):
            ttk.Label(settings, text=caption).grid(row=0, column=i, sticky="w", padx=(0, 8))
            control = ttk.Entry(settings, textvariable=variable, width=width)
            control.grid(row=1, column=i, sticky="ew", padx=(0, 10), pady=(5, 0))
            self.inputs.append(control)
        ttk.Label(settings, text="Device").grid(row=0, column=5, sticky="w")
        device = ttk.Combobox(settings, values=("Auto", "GPU 0", "CPU"), state="readonly",
                              textvariable=self.device_var, width=9)
        device.grid(row=1, column=5, sticky="ew", pady=(5, 0))
        self.inputs.append(device)

        split_row = ttk.Frame(frame)
        split_row.grid(row=8, column=0, columnspan=3, sticky="w", pady=(4, 10))
        ttk.Label(split_row, text="Validation split:").pack(side="left", padx=(0, 10))
        self.split_choice = ttk.Combobox(split_row, textvariable=self.split_var,
                                         values=("test", "val", "train"),
                                         state="readonly", width=12)
        self.split_choice.pack(side="left")
        self.inputs.append(self.split_choice)
        ttk.Label(split_row, text="Confidence floor = 0.001 (for the complete P/R curve)",
                  foreground="#555555").pack(side="left", padx=(20, 0))

        actions = ttk.Frame(frame)
        actions.grid(row=9, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        self.run_button = ttk.Button(actions, text="Run validation & find threshold",
                                     command=self._start)
        self.run_button.pack(side="left")
        self.inputs.append(self.run_button)
        ttk.Label(frame, textvariable=self.status_var).grid(row=10, column=0, columnspan=3, sticky="w")
        self.progress = ttk.Progressbar(frame, mode="indeterminate")
        self.progress.grid(row=11, column=0, columnspan=3, sticky="ew", pady=(6, 8))
        self.result_label = ttk.Label(frame, textvariable=self.result_var,
                                      font=("Segoe UI", 11, "bold"), wraplength=820)
        self.result_label.grid(row=12, column=0, columnspan=3, sticky="w", pady=(2, 10))

        log_frame = ttk.Frame(frame)
        log_frame.grid(row=13, column=0, columnspan=3, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log = tk.Text(log_frame, state="disabled", wrap="word", height=9,
                           background="#f5f6f8", font=("Consolas", 9))
        self.log.grid(row=0, column=0, sticky="nsew")
        scr = ttk.Scrollbar(log_frame, command=self.log.yview)
        scr.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scr.set)
        ttk.Label(frame, text="Precision/recall are object-level detection metrics, not mass-weighted sorting purity.",
                  foreground="#555555").grid(row=14, column=0, columnspan=3, sticky="w", pady=(9, 0))

    def _mode_changed(self):
        is_yaml = self.mode_var.get() == "Dataset YAML"
        self.source_var.set("")
        self.source_label.configure(text="Dataset YAML" if is_yaml else "Image directory")
        self.source_hint.configure(text=(
            "Choose data.yaml or autosplit_data.yaml. The chosen split is resolved by Ultralytics."
            if is_yaml else
            "Choose images/ with sibling labels/, or a folder with matching image.jpg/image.txt files."
        ))
        self.split_choice.configure(state="readonly" if is_yaml else "disabled")

    def _choose_dataset_folder(self):
        folder = filedialog.askdirectory(parent=self.root, title="Select YOLO dataset folder",
                                         mustexist=True)
        if not folder:
            return
        selected = Path(folder)
        config = discover_dataset_yaml(selected)
        if config:
            self.mode_var.set("Dataset YAML")
            self._mode_changed()
            self.source_var.set(str(config))
        else:
            image_dir = selected / "images" if (selected / "images").is_dir() else selected
            self.mode_var.set("Image folder")
            self._mode_changed()
            self.source_var.set(str(image_dir))

    def _choose_model(self):
        file = filedialog.askopenfilename(parent=self.root, title="Select YOLO weights",
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")])
        if file:
            self.model_var.set(file)

    def _choose_source(self):
        if self.mode_var.get() == "Dataset YAML":
            path = filedialog.askopenfilename(parent=self.root, title="Select YOLO dataset YAML",
                filetypes=[("YAML dataset config", "*.yaml *.yml"), ("All files", "*.*")])
            if path:
                self.source_var.set(path)
        else:
            folder = filedialog.askdirectory(parent=self.root, title="Select image directory",
                                              mustexist=True)
            if folder:
                self.source_var.set(folder)

    def _choose_output(self):
        folder = filedialog.askdirectory(parent=self.root, title="Choose results directory",
                                          mustexist=True)
        if folder:
            self.output_var.set(folder)

    def _read_settings(self):
        model = Path(self.model_var.get().strip().strip('"')).expanduser()
        source = Path(self.source_var.get().strip().strip('"')).expanduser()
        output_text = self.output_var.get().strip().strip('"')
        if not self.model_var.get().strip() or not model.is_file():
            raise ValueError("Select an existing YOLO .pt model file.")
        if model.suffix.lower() != ".pt":
            raise ValueError("Please select PyTorch YOLO weights (.pt).")
        mode = self.mode_var.get()
        if not self.source_var.get().strip() or not (
            source.is_file() if mode == "Dataset YAML" else source.is_dir()
        ):
            raise ValueError("Select an existing data.yaml file or image directory.")
        if mode == "Dataset YAML" and source.suffix.lower() not in (".yaml", ".yml"):
            raise ValueError("Dataset source must be a .yaml or .yml file.")
        if not output_text:
            raise ValueError("Choose a results directory.")
        try:
            food_class = int(self.class_var.get().strip())
            imgsz = int(self.imgsz_var.get().strip())
            batch = int(self.batch_var.get().strip())
            workers = int(self.workers_var.get().strip())
            target = parse_target_precision(self.precision_var.get())
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid numeric setting: {exc}") from exc
        if food_class < 0 or imgsz <= 0 or batch <= 0 or workers < 0:
            raise ValueError("Class must be >=0, image size/batch >0 and workers >=0.")
        return dict(model=str(model.resolve()), mode=mode, source=str(source.resolve()),
                    output=str(Path(output_text).expanduser().resolve()),
                    split=self.split_var.get(), precision=target, food_class=food_class,
                    imgsz=imgsz, batch=batch, workers=workers, device=self.device_var.get())

    def _start(self):
        if self.busy:
            return
        try:
            settings = self._read_settings()
        except ValueError as exc:
            messagebox.showerror("Invalid evaluation settings", str(exc), parent=self.root)
            return
        self.busy = True
        for widget in self.inputs:
            widget.configure(state="disabled")
        self.status_var.set("Starting validation ...")
        self.result_var.set("Evaluating model ...")
        self.progress.start(14)
        def worker():
            try:
                evaluate(settings, lambda kind, value: self.messages.put((kind, value)))
            except Exception as exc:
                self.messages.put(("error", str(exc), traceback.format_exc()))
        threading.Thread(target=worker, daemon=True, name="RecallPrecisionValidation").start()

    def _append_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", str(text) + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")
        print("[CV Waste Recall Test] " + str(text), flush=True)

    def _poll(self):
        try:
            while True:
                item = self.messages.get_nowait()
                kind, value = item[:2]
                if kind == "log":
                    self._append_log(value)
                elif kind == "status":
                    self.status_var.set(value)
                elif kind == "done":
                    self._finish()
                    if value["target_reached"]:
                        self.result_var.set(
                            f"{value['class_name']} · confidence {value['confidence_threshold']:.4f} · "
                            f"precision {value['achieved_precision']:.2%} · "
                            f"recall {value['recall_at_precision']:.2%}"
                        )
                    else:
                        self.result_var.set(
                            f"Target precision {value['target_precision']:.1%} not reached "
                            f"for {value['class_name']} on the selected split."
                        )
                    self.status_var.set("Completed · results saved in " + value["output_directory"])
                elif kind == "error":
                    self._finish()
                    details = item[2] if len(item) > 2 else str(value)
                    self._append_log(details)
                    self.status_var.set("Evaluation failed. Inspect log.")
                    self.result_var.set(str(value))
                    messagebox.showerror("Validation failed", str(value), parent=self.root)
        except queue.Empty:
            pass
        if self.root.winfo_exists():
            self.root.after(180, self._poll)

    def _finish(self):
        self.busy = False
        self.progress.stop()
        for widget in self.inputs:
            widget.configure(state="normal")
        self.mode_choice.configure(state="readonly")
        self.split_choice.configure(state="readonly" if self.mode_var.get() == "Dataset YAML" else "disabled")
        for widget in self.inputs:
            if isinstance(widget, ttk.Combobox) and widget is not self.split_choice:
                widget.configure(state="readonly")


def main():
    root = tk.Tk()
    RecallTestGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
