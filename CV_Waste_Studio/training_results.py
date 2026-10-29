"""In-app review of native Ultralytics run outputs; no model inference required.

Reads results.csv and existing validation figures; never changes checkpoints or
source datasets. The training code remains Ultralytics YOLO.train().
"""
from __future__ import annotations

import csv
from datetime import datetime
import math
import os
from pathlib import Path
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from training_charts import LiveMetricsGraph

INK = "#19263B"
MUTED = "#60718A"
ACCENT = "#405DE6"
BORDER = "#DAE2EF"
SURFACE = "#FFFFFF"
GRAPH_KEYS = (
    "train/box_loss", "train/cls_loss", "train/dfl_loss",
    "val/box_loss", "val/cls_loss", "val/dfl_loss",
    "metrics/precision(B)", "metrics/recall(B)",
    "metrics/mAP50(B)", "metrics/mAP50-95(B)",
)
IMAGE_PLOTS = (
    ("Confusion matrix (normalised)", "confusion_matrix_normalized.png"),
    ("Confusion matrix", "confusion_matrix.png"),
    ("Precision-recall curve", "PR_curve.png"),
    ("F1 versus confidence", "F1_curve.png"),
    ("Precision versus confidence", "P_curve.png"),
    ("Recall versus confidence", "R_curve.png"),
    ("Original Ultralytics results", "results.png"),
)


def load_run(run_folder):
    """Parse the authoritative Ultralytics results.csv; reject empty/bad runs.

    The highest logged validation mAP50-95 is *not* necessarily the epoch of
    best.pt, because Ultralytics can select a checkpoint using model fitness.
    """
    folder = Path(run_folder).expanduser().resolve()
    source = folder / "results.csv"
    if not source.is_file():
        raise FileNotFoundError(f"No results.csv found in:\n{folder}")
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        table = csv.DictReader(stream)
        if not table.fieldnames or "epoch" not in [x.strip() for x in table.fieldnames]:
            raise ValueError("results.csv has no epoch column.")
        by_epoch = {}
        for raw in table:
            try:
                number = float(str(next(v for k, v in raw.items() if k.strip() == "epoch")).strip())
                if not math.isfinite(number) or not number.is_integer() or number < 1:
                    continue
                epoch = int(number)
            except (ValueError, TypeError, StopIteration):
                continue
            values = {}
            for key, val in raw.items():
                key = str(key).strip()
                if key not in GRAPH_KEYS or val is None:
                    continue
                try:
                    score = float(val.strip())
                    if math.isfinite(score):
                        values[key] = score
                except (ValueError, TypeError):
                    continue
            if values:
                by_epoch[epoch] = values
    if not by_epoch:
        raise ValueError("results.csv contains no usable epoch metrics.")
    epochs = [{"epoch": n, "epochs": max(by_epoch), "metrics": by_epoch[n]}
              for n in sorted(by_epoch)]
    key = "metrics/mAP50-95(B)"
    best = max((record for record in epochs if key in record["metrics"]),
               key=lambda record: record["metrics"][key], default=None)
    best_metric = key
    if best is None:
        best_metric = "metrics/mAP50(B)"
        best = max((record for record in epochs if best_metric in record["metrics"]),
                   key=lambda record: record["metrics"][best_metric], default=None)
    latest = epochs[-1]
    return {"folder": folder, "epochs": epochs, "latest": latest,
            "peak": best, "peak_metric": best_metric if best else None,
            "best_pt": folder / "weights" / "best.pt",
            "last_pt": folder / "weights" / "last.pt"}


def discover_runs(log_dir, limit=30):
    """Find previous runs in project folders recorded in Studio settings JSONs.

    Only checks immediate children of known output folders, never scans the
    image dataset recursively or probes arbitrary locations.
    """
    import json
    log_dir = Path(log_dir)
    projects = []
    for settings in sorted(log_dir.glob("train_*_settings.json"), reverse=True)[:100]:
        try:
            cfg = json.loads(settings.read_text(encoding="utf-8"))
            if cfg.get("project"):
                projects.append(Path(cfg["project"]).expanduser())
            elif cfg.get("existing_data_yaml"):
                src = Path(cfg["existing_data_yaml"]).expanduser()
                projects.append((src if src.is_dir() else src.parent) / "training_runs")
            elif cfg.get("output_dir"):
                projects.append(Path(cfg["output_dir"]).expanduser() / "training_runs")
        except (OSError, ValueError, TypeError, KeyError):
            continue
    found = {}
    for project in projects:
        if not project.is_dir():
            continue
        try:
            for child in project.iterdir():
                if child.is_dir():
                    csv_file = child / "results.csv"
                    if csv_file.is_file():
                        found[str(child.resolve())] = (csv_file.stat().st_mtime, child)
        except OSError:
            continue
    return [item[1] for item in sorted(found.values(), key=lambda item: item[0], reverse=True)[:limit]]


def _format_metric(value, percent=False):
    return "—" if value is None else (f"{value * 100:.1f}%" if percent else f"{value:.3f}")


def _open_folder(folder):
    if os.name == "nt":
        os.startfile(str(folder))
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(folder)])
    else:
        subprocess.Popen(["xdg-open", str(folder)], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)


class TrainingResultsPanel(tk.Frame):
    """Scrollable in-app Ultralytics results dashboard and validation plot viewer."""

    def __init__(self, parent, logs_dir, on_log=None):
        super().__init__(parent, bg=SURFACE)
        self.logs_dir = Path(logs_dir)
        self.on_log = on_log
        self.loaded = None
        self.recent = {}
        self.figures = {}
        self.plot_name = tk.StringVar()
        self.zoom = tk.DoubleVar(value=100)
        self._picture = None
        self._resize_after = None
        self._build()
        self.refresh_runs()

    def _build(self):
        # One self-contained results page rather than extra toplevel windows.
        outer = tk.Frame(self, bg=SURFACE)
        outer.pack(fill="both", expand=True, padx=18, pady=12)
        toolbar = tk.Frame(outer, bg=SURFACE)
        toolbar.pack(fill="x")
        tk.Label(toolbar, text="Finished training runs", bg=SURFACE, fg=INK,
                 font=("Segoe UI", 13, "bold")).pack(side="left")
        ttk.Button(toolbar, text="Browse run folder…", command=self.browse_run).pack(side="right")
        ttk.Button(toolbar, text="Refresh runs", command=self.refresh_runs).pack(side="right", padx=8)
        picker = tk.Frame(outer, bg=SURFACE)
        picker.pack(fill="x", pady=(10, 5))
        tk.Label(picker, text="Recent run", bg=SURFACE, fg=MUTED).pack(side="left", padx=(0, 9))
        self.choice = tk.StringVar()
        self.run_choice = ttk.Combobox(picker, textvariable=self.choice, state="readonly")
        self.run_choice.pack(side="left", fill="x", expand=True)
        self.run_choice.bind("<<ComboboxSelected>>", lambda _: self._load_chosen())
        self.path_note = tk.StringVar(value="Select a finished Ultralytics run or browse to its results.csv folder.")
        tk.Label(outer, textvariable=self.path_note, bg=SURFACE, fg=MUTED,
                 anchor="w", font=("Segoe UI", 9)).pack(fill="x", pady=(0, 8))

        self.summary = tk.Frame(outer, bg=SURFACE)
        self.summary.pack(fill="x", pady=(0, 9))
        self.cards = []
        for name in ("Epochs recorded", "Peak val mAP50–95", "Precision at peak", "Recall at peak", "Final val mAP50"):
            card = tk.Frame(self.summary, bg="#F6F9FE", highlightbackground=BORDER,
                            highlightthickness=1)
            card.pack(side="left", fill="x", expand=True, padx=(0, 7))
            tk.Label(card, text=name, bg="#F6F9FE", fg=MUTED,
                     font=("Segoe UI", 8)).pack(anchor="w", padx=9, pady=(7, 2))
            value = tk.StringVar(value="—")
            tk.Label(card, textvariable=value, bg="#F6F9FE", fg=INK,
                     font=("Segoe UI", 15, "bold")).pack(anchor="w", padx=9, pady=(0, 8))
            self.cards.append(value)

        self.notice = tk.StringVar(value="Validation performance is measured on the run's validation split, not an independent test set.")
        tk.Label(outer, textvariable=self.notice, bg=SURFACE, fg=MUTED,
                 anchor="w", justify="left", font=("Segoe UI", 9),
                 wraplength=950).pack(fill="x", pady=(8, 3))
        tools = tk.Frame(outer, bg=SURFACE)
        tools.pack(fill="x", pady=(1, 0))
        self.checkpoints = tk.StringVar(value="best.pt: —    ·    last.pt: —")
        tk.Label(tools, textvariable=self.checkpoints, bg=SURFACE, fg=MUTED,
                 font=("Segoe UI", 9)).pack(side="left")
        ttk.Button(tools, text="Open results folder", command=self.open_folder).pack(side="right")

        self.tabs = ttk.Notebook(outer)
        self.tabs.pack(fill="both", expand=True)
        curves = tk.Frame(self.tabs, bg=SURFACE)
        figures = tk.Frame(self.tabs, bg=SURFACE)
        self.tabs.add(curves, text="Interactive training curves")
        self.tabs.add(figures, text="Validation plots & examples")
        table_page = tk.Frame(self.tabs, bg=SURFACE)
        self.tabs.add(table_page, text="Epoch metrics table")
        columns = (("epoch", "Epoch", 65), ("train/box_loss", "Train box", 95),
                   ("train/cls_loss", "Train cls", 95), ("train/dfl_loss", "Train dfl", 95),
                   ("val/box_loss", "Val box", 95), ("val/cls_loss", "Val cls", 95),
                   ("val/dfl_loss", "Val dfl", 95), ("metrics/precision(B)", "Precision", 90),
                   ("metrics/recall(B)", "Recall", 90), ("metrics/mAP50(B)", "mAP50", 90),
                   ("metrics/mAP50-95(B)", "mAP50–95", 100))
        self.table_columns = columns
        self.epoch_table = ttk.Treeview(table_page, columns=[c[0] for c in columns], show="headings")
        for key, title, width in columns:
            self.epoch_table.heading(key, text=title, command=lambda col=key: self._sort_epochs(col))
            self.epoch_table.column(key, anchor="center", width=width, minwidth=62, stretch=False)
        self.epoch_table.pack(side="top", fill="both", expand=True, padx=3, pady=(8, 0))
        xbar = ttk.Scrollbar(table_page, orient="horizontal", command=self.epoch_table.xview)
        xbar.pack(side="bottom", fill="x")
        self.epoch_table.configure(xscrollcommand=xbar.set)
        self._sort_reverse = {}
        hints = tk.Frame(curves, bg=SURFACE)
        hints.pack(fill="x", pady=(9, 4))
        tk.Label(hints, text="Click to enlarge · hover an enlarged chart to inspect exact epoch values · Esc / × to close",
                 bg=SURFACE, fg=MUTED, font=("Segoe UI", 9)).pack(side="left")
        self.graph = LiveMetricsGraph(curves)
        self.graph.configure(height=420)
        self.graph.pack(fill="both", expand=True, padx=2, pady=(0, 8))
        plot_control = tk.Frame(figures, bg=SURFACE)
        plot_control.pack(fill="x", pady=(9, 5))
        tk.Label(plot_control, text="Ultralytics plot", bg=SURFACE, fg=MUTED).pack(side="left", padx=(0, 7))
        self.plot_picker = ttk.Combobox(plot_control, textvariable=self.plot_name,
                                         state="readonly", width=36)
        self.plot_picker.pack(side="left", fill="x", expand=True)
        self.plot_picker.bind("<<ComboboxSelected>>", lambda _: self._render_plot())
        ttk.Button(plot_control, text="−", width=3, command=lambda: self.change_zoom(-25)).pack(side="left", padx=(9, 0))
        ttk.Button(plot_control, text="+", width=3, command=lambda: self.change_zoom(25)).pack(side="left", padx=(3, 0))
        self.zoom_text = tk.StringVar(value="100%")
        tk.Label(plot_control, textvariable=self.zoom_text, bg=SURFACE, fg=MUTED,
                 width=6).pack(side="left")
        self.image_canvas = tk.Canvas(figures, bg="#F8FAFD", bd=0, highlightthickness=0)
        self.image_canvas.pack(side="left", fill="both", expand=True)
        self.vscroll = ttk.Scrollbar(figures, orient="vertical", command=self.image_canvas.yview)
        self.vscroll.pack(side="right", fill="y")
        self.image_canvas.configure(yscrollcommand=self.vscroll.set)
        self.image_canvas.bind("<Configure>", self._fit_picture)
    def refresh_runs(self):
        paths = discover_runs(self.logs_dir)
        found = {}
        for p in paths:
            try:
                title = f"{datetime.fromtimestamp((p / 'results.csv').stat().st_mtime):%Y-%m-%d %H:%M} · {p.name} · {p.parent.name}"
                found[title] = p
            except OSError:
                continue
        if self.loaded is not None and self.loaded["folder"] not in found.values():
            found = {f"Current run · {self.loaded['folder'].name}": self.loaded["folder"], **found}
        self.recent = found
        self.run_choice.configure(values=list(self.recent))
        if self.loaded is not None:
            for label, path in self.recent.items():
                if path.resolve() == self.loaded["folder"]:
                    self.choice.set(label)
                    break

    def _load_chosen(self):
        folder = self.recent.get(self.choice.get())
        if folder:
            self.load_run(folder)

    def browse_run(self):
        folder = filedialog.askdirectory(parent=self, title="Choose Ultralytics training run folder")
        if folder:
            self.load_run(folder)

    def load_run(self, folder):
        try:
            data = load_run(folder)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Could not read training results", str(exc), parent=self)
            return False
        self.loaded = data
        self.path_note.set(str(data["folder"]))
        self.graph.reset()
        for event in data["epochs"]:
            self.graph.history.add(event)
        self.graph.redraw()
        self.epoch_table.delete(*self.epoch_table.get_children())
        for event in data["epochs"]:
            metric = event["metrics"]
            row = [str(event["epoch"])] + [f"{metric[k]:.4f}" if k in metric else "—"
                                             for k, _, _ in self.table_columns[1:]]
            self.epoch_table.insert("", "end", values=row)
        peak = data["peak"]
        latest = data["latest"]
        best_values = peak["metrics"] if peak else {}
        self.cards[0].set(str(len(data["epochs"])) + f" (to {latest['epoch']})")
        self.cards[1].set(_format_metric(best_values.get("metrics/mAP50-95(B)"), True))
        self.cards[2].set(_format_metric(best_values.get("metrics/precision(B)"), True))
        self.cards[3].set(_format_metric(best_values.get("metrics/recall(B)"), True))
        self.cards[4].set(_format_metric(latest["metrics"].get("metrics/mAP50(B)"), True))
        self.checkpoints.set("best.pt: " + ("available" if data["best_pt"].is_file() else "not found") +
                             "    ·    last.pt: " + ("available" if data["last_pt"].is_file() else "not found"))
        self.notice.set((f"Highest logged validation mAP at epoch {peak['epoch']}. " if peak else "") +
                        "Validation scores are not independent test-set performance. "
                        "Review the confusion matrix and PR/F1 curves before deployment.")
        self.figures = {name: data["folder"] / filename for name, filename in IMAGE_PLOTS
                        if (data["folder"] / filename).is_file()}
        for filename in sorted(data["folder"].glob("val_batch*_pred.jpg"))[:5]:
            self.figures["Validation example: " + filename.name] = filename
        self.plot_picker.configure(values=list(self.figures))
        self.plot_name.set(next(iter(self.figures), ""))
        self._render_plot()
        self.refresh_runs()
        if self.on_log:
            self.on_log(f"Loaded {len(data['epochs'])} epoch records from {data['folder']}")
        return True

    def _sort_epochs(self, key):
        items = list(self.epoch_table.get_children())
        index = [name for name, _, _ in self.table_columns].index(key)
        descending = self._sort_reverse.get(key, False)
        def score(item):
            value = self.epoch_table.item(item, "values")[index]
            try:
                return (1, float(value))
            except (ValueError, TypeError):
                return (0, 0)
        for position, item in enumerate(sorted(items, key=score, reverse=descending)):
            self.epoch_table.move(item, "", position)
        self._sort_reverse[key] = not descending

    def change_zoom(self, delta):
        self.zoom.set(max(25, min(300, self.zoom.get() + delta)))
        self._render_plot()

    def _fit_picture(self, _event=None):
        if self._resize_after is not None:
            try:
                self.after_cancel(self._resize_after)
            except tk.TclError:
                pass
        self._resize_after = self.after(120, self._render_plot)

    def _render_plot(self):
        self._resize_after = None
        canvas = self.image_canvas
        canvas.delete("all")
        self._picture = None
        filename = self.figures.get(self.plot_name.get())
        self.zoom_text.set(f"{self.zoom.get():.0f}%")
        if filename is None:
            canvas.create_text(120, 35, anchor="w", text="No saved validation figures found in this run.",
                               fill=MUTED, font=("Segoe UI", 10))
            return
        try:
            from PIL import Image, ImageTk
            with Image.open(filename) as img:
                img = img.convert("RGB")
                available_w = max(320, canvas.winfo_width() - 20)
                available_h = max(170, canvas.winfo_height() - 20)
                scale = min(available_w / img.width, available_h / img.height, 1.0)
                scale *= self.zoom.get() / 100.0
                new_size = (max(1, round(img.width * scale)),
                            max(1, round(img.height * scale)))
                img.thumbnail(new_size, Image.Resampling.LANCZOS)
                self._picture = ImageTk.PhotoImage(img, master=canvas)
                canvas.create_image(8, 8, anchor="nw", image=self._picture)
                canvas.configure(scrollregion=(0, 0, img.width + 16, img.height + 16))
        except (OSError, ImportError, ValueError) as exc:
            canvas.create_text(14, 25, anchor="w", text=f"Unable to display plot: {exc}",
                               fill="#B53A43", font=("Segoe UI", 10))

    def open_folder(self):
        if self.loaded is None:
            messagebox.showinfo("No training results", "Select a run first.", parent=self)
            return
        try:
            _open_folder(self.loaded["folder"])
        except OSError as exc:
            messagebox.showerror("Could not open results folder", str(exc), parent=self)
