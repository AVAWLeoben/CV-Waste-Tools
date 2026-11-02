"""Embedded Model Comparison page: sequential training and true test-set evaluation."""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from augmentation_options import edit_augmentation
from model_comparison_worker import parse_event, validate_settings
from training_charts import LiveMetricsGraph

INK = "#19263B"
MUTED = "#60718A"
BORDER = "#DAE2EF"
ACCENT = "#405DE6"
BG = "#FFFFFF"
PALETTE = ("#3C70D7", "#27A28C", "#E68C3A", "#A46EC4")
METRIC_NAMES = (("map50_95", "mAP50–95"), ("map50", "mAP50"),
                ("precision", "Precision"), ("recall", "Recall"))


def open_folder(path):
    if os.name == "nt":
        os.startfile(str(path))
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)


class ComparisonBars(tk.Canvas):
    """Dependency-free ranked comparison chart for evaluated TEST metrics."""
    def __init__(self, parent):
        super().__init__(parent, bg="#F8FAFD", highlightthickness=1,
                         highlightbackground=BORDER, height=260)
        self.results = []
        self.metric = "map50_95"
        self.split_name = "Validation"
        self.bind("<Configure>", lambda _: self.redraw())

    def update_data(self, results, metric=None):
        self.results = list(results)
        if metric:
            self.metric = metric
        self.redraw()

    def redraw(self):
        self.delete("all")
        w, h = max(120, self.winfo_width()), max(140, self.winfo_height())
        self.create_text(16, 16, anchor="nw", text=self.split_name + "-set performance",
                         font=("Segoe UI", 11, "bold"), fill=INK)
        rows = [(x["model"], float(x[self.metric])) for x in self.results
                if x.get("status") == "completed" and isinstance(x.get(self.metric), (float, int))]
        if not rows:
            self.create_text(w / 2, h / 2, text=f"{self.split_name}-set scores appear after each model is evaluated.",
                             font=("Segoe UI", 10), fill=MUTED)
            return
        rows = sorted(rows, key=lambda x: x[1], reverse=True)
        left, right = min(max(138, int(w * .18)), 220), w - 70
        top, bottom = 53, h - 28
        for tick in range(0, 5):
            x = left + (right - left) * tick / 4
            self.create_line(x, top - 8, x, bottom + 3, fill="#E1E7F1", dash=(3, 4))
            self.create_text(x, bottom + 7, anchor="n", text=f"{tick * 25}%",
                             fill=MUTED, font=("Segoe UI", 8))
        row_h = min(52, max(23, (bottom - top) / max(len(rows), 1)))
        for i, (label, score) in enumerate(rows):
            y = top + (i + .5) * ((bottom - top) / max(len(rows), 1))
            self.create_text(left - 12, y, anchor="e", text=label[:28],
                             font=("Segoe UI", 9), fill=INK)
            x = left + (right - left) * max(0, min(1, score))
            self.create_rectangle(left, y - row_h * .26, x, y + row_h * .26,
                                  fill=PALETTE[i % len(PALETTE)], outline="")
            self.create_text(min(w - 8, x + 8), y, anchor="w", text=f"{score * 100:.1f}%",
                             font=("Segoe UI", 9, "bold"), fill=INK)


class ModelComparisonPanel(tk.Frame):
    """Works independently of the existing training GUI and shared activity dock."""
    def __init__(self, parent, env_python, process_env, logs_dir, on_log=None, can_start=None):
        super().__init__(parent, bg=BG)
        self.env_python = env_python
        self.process_env = process_env
        self.logs_dir = Path(logs_dir)
        self.on_log = on_log
        self.can_start = can_start
        self.models = []
        self.results = []
        self.screening_run = True
        self.augmentation = {}
        self.job = None
        self.run_folder = None
        self.stopped = False
        self.summary_path = None
        self._build()

    @property
    def busy(self):
        return self.job is not None and self.job["process"].poll() is None

    def _build(self):
        frame = ttk.Frame(self, padding=(18, 12))
        frame.pack(fill="both", expand=True)
        heading = ttk.Frame(frame)
        heading.pack(fill="x", pady=(0, 9))
        ttk.Label(heading, text="Sequential YOLO comparison",
                  font=("Segoe UI", 14, "bold")).pack(side="left")
        ttk.Button(heading, text="Load previous comparison", command=self.load_previous).pack(
            side="right", padx=(8, 0))
        self.toggle_button = ttk.Button(heading, text="Show setup", command=self._toggle_setup)
        self.stop_button = ttk.Button(heading, text="Stop comparison", command=self.stop,
                                      style="Danger.TButton", state="disabled")
        self.stop_button.pack(side="right")
        self.toggle_button.pack(side="right", padx=9)
        self.toggle_button.pack_forget()
        self.status = tk.StringVar(value="Ready — choose candidate architectures and a dataset YAML with train/val splits")
        ttk.Label(frame, textvariable=self.status, style="Muted.TLabel").pack(fill="x", pady=(0, 10))

        self.setup = ttk.Frame(frame)
        self.setup.pack(fill="both", expand=True)
        self.setup_canvas = tk.Canvas(self.setup, bg=BG, highlightthickness=0)
        yscroll = ttk.Scrollbar(self.setup, orient="vertical", command=self.setup_canvas.yview)
        self.setup_canvas.configure(yscrollcommand=yscroll.set)
        yscroll.pack(side="right", fill="y")
        self.setup_canvas.pack(side="left", fill="both", expand=True)
        body = ttk.Frame(self.setup_canvas, padding=(4, 2, 12, 8))
        window = self.setup_canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>", lambda _: self.setup_canvas.configure(
            scrollregion=self.setup_canvas.bbox("all")))
        self.setup_canvas.bind("<Configure>", lambda e: self.setup_canvas.itemconfigure(
            window, width=e.width))
        body.columnconfigure(1, weight=1)
        self.yaml = tk.StringVar()
        self._input_row(body, 0, "Dataset YAML (train/val; test optional for screening)", self.yaml,
                        browse=lambda: self._pick_file(self.yaml, "YOLO data YAML", [("YAML", "*.yaml *.yml")]))
        self.screening = tk.BooleanVar(value=True)
        mode_row = ttk.Frame(body)
        mode_row.grid(row=1, column=0, columnspan=3, sticky="w", pady=(1, 9))
        ttk.Checkbutton(mode_row, variable=self.screening,
                        text="Architecture screening (evaluate on validation; keep test split untouched)").pack(side="left")
        ttk.Label(mode_row, text="Uncheck for the original full test-set comparison.",
                  style="Muted.TLabel").pack(side="left", padx=(12, 0))
        ttk.Separator(body).grid(row=2, column=0, columnspan=3, sticky="ew", pady=(4, 12))
        ttk.Label(body, text="Models to train (in order)", font=("Segoe UI", 12, "bold")).grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(0, 8))
        self.version = tk.StringVar(value="11")
        self.size = tk.StringVar(value="l")
        row = ttk.Frame(body)
        row.grid(row=4, column=0, columnspan=3, sticky="ew")
        ttk.Label(row, text="YOLO version").pack(side="left")
        ttk.Combobox(row, textvariable=self.version, values=("8", "10", "11", "12", "26"), state="readonly",
                     width=7).pack(side="left", padx=(6, 16))
        ttk.Label(row, text="Size").pack(side="left")
        ttk.Combobox(row, textvariable=self.size, values=("n", "s", "m", "l", "x"), state="readonly",
                     width=6).pack(side="left", padx=(6, 12))
        ttk.Button(row, text="+ Add pretrained", command=self.add_pretrained).pack(side="left", padx=(0, 8))
        ttk.Button(row, text="+ Add .pt weights", command=self.add_weights).pack(side="left")
        tableframe = ttk.Frame(body)
        tableframe.grid(row=5, column=0, columnspan=3, sticky="nsew", pady=(8, 4))
        self.model_table = ttk.Treeview(tableframe, columns=("number", "model", "source"),
                                        show="headings", height=4, selectmode="browse")
        for col, name, width in (("number", "#", 42), ("model", "Model", 185), ("source", "Source", 530)):
            self.model_table.heading(col, text=name)
            self.model_table.column(col, width=width, stretch=(col == "source"), anchor="w")
        self.model_table.pack(side="left", fill="x", expand=True)
        sc = ttk.Scrollbar(tableframe, command=self.model_table.yview)
        sc.pack(side="right", fill="y")
        self.model_table.configure(yscrollcommand=sc.set)
        actions = ttk.Frame(body)
        actions.grid(row=6, column=0, columnspan=3, sticky="w", pady=(0, 7))
        for caption, callback in (("Move up", lambda: self._move(-1)),
                                  ("Move down", lambda: self._move(1)),
                                  ("Remove selected", self._remove)):
            ttk.Button(actions, text=caption, command=callback).pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="+ Sweep sizes (n/s/m/l/x)", command=self.add_size_sweep).pack(side="left", padx=(8, 6))
        ttk.Button(actions, text="+ Sweep generations (m/l)", command=self.add_generation_sweep).pack(side="left")
        ttk.Separator(body).grid(row=7, column=0, columnspan=3, sticky="ew", pady=10)
        ttk.Label(body, text="Shared training parameters", font=("Segoe UI", 12, "bold")).grid(
            row=8, column=0, columnspan=3, sticky="w", pady=(0, 6))
        self.duration_type = tk.StringVar(value="Epochs")
        self.epochs = tk.StringVar(value="30")
        self.batch = tk.StringVar(value="4")
        self.imgsz = tk.StringVar(value="640")
        self.cache = tk.StringVar(value="disk")
        self.workers = tk.StringVar(value="8")
        self.cls_weight = tk.StringVar(value="0.5")
        self.eval_batch = tk.StringVar(value="4")
        self.device = tk.StringVar(value="auto")
        self.cleanup = tk.BooleanVar(value=True)
        self.retry_oom = tk.BooleanVar(value=True)
        self.auto_install_cuda = tk.BooleanVar(value=False)
        self.compact_progress = tk.BooleanVar(value=True)
        duration_frame = ttk.Frame(body)
        duration_frame.grid(row=9, column=1, sticky="w", pady=4)
        ttk.Combobox(duration_frame, textvariable=self.duration_type, values=("Epochs", "Hours"),
                     state="readonly", width=10).pack(side="left")
        ttk.Entry(duration_frame, textvariable=self.epochs, width=12).pack(side="left", padx=8)
        ttk.Label(body, text="Training duration").grid(row=9, column=0, sticky="w", pady=4)
        self._entry_row(body, 10, "Batch size", self.batch)
        self._entry_row(body, 11, "Image size (imgsz)", self.imgsz)
        ttk.Label(body, text="Cache").grid(row=12, column=0, sticky="w", pady=4)
        ttk.Combobox(body, textvariable=self.cache, values=("off", "disk", "ram"),
                     state="readonly", width=15).grid(row=12, column=1, sticky="w", pady=4)
        self._entry_row(body, 13, "Workers", self.workers)
        self._entry_row(body, 14, "CLS loss weight", self.cls_weight)
        self._entry_row(body, 15, "Evaluation batch (val/test)", self.eval_batch)
        ttk.Label(body, text="Training device").grid(row=16, column=0, sticky="w", pady=4)
        ttk.Combobox(body, textvariable=self.device, values=("auto", "gpu", "cpu"),
                     state="readonly", width=15).grid(row=16, column=1, sticky="w", pady=4)
        ttk.Checkbutton(body, variable=self.cleanup, text="Delete newly-created .npy disk cache files after each model").grid(
            row=17, column=1, columnspan=2, sticky="w", pady=3)
        ttk.Checkbutton(body, variable=self.retry_oom, text="Automatically retry CUDA OOM with smaller batch (max 2 retries)").grid(
            row=18, column=1, columnspan=2, sticky="w", pady=3)
        ttk.Checkbutton(body, variable=self.auto_install_cuda,
                        text="Install CUDA PyTorch if needed (opt-in)").grid(
            row=19, column=1, columnspan=2, sticky="w", pady=3)
        ttk.Checkbutton(body, variable=self.compact_progress,
                        text="Compact training output (live progress remains visible)").grid(
            row=20, column=1, columnspan=2, sticky="w", pady=3)
        ttk.Label(body, text="Each model runs in a new Python process: dataloader workers and CUDA RAM are released between models.",
                  style="Muted.TLabel").grid(row=21, column=1, columnspan=2, sticky="w", pady=(0, 8))
        aug = ttk.Frame(body)
        aug.grid(row=22, column=1, sticky="w", pady=3)
        ttk.Button(aug, text="Augmentation options…", command=lambda: edit_augmentation(
            self, self.augmentation, self._set_augmentation)).pack(side="left")
        self.augmentation_label = ttk.Label(aug, text="Library defaults", style="Muted.TLabel")
        self.augmentation_label.pack(side="left", padx=10)
        self.output = tk.StringVar()
        self._input_row(body, 23, "Comparison results folder", self.output,
                        browse=lambda: self._pick_dir(self.output))
        ttk.Separator(body).grid(row=24, column=0, columnspan=3, sticky="ew", pady=12)
        self.start_button = ttk.Button(body, text="Train & compare queued models",
                                       command=self.start, style="Accent.TButton")
        self.start_button.grid(row=25, column=2, sticky="e", pady=(0, 8))

        self.monitor = ttk.Frame(frame)
        self.monitor_info = tk.StringVar(value="No comparison running")
        ttk.Label(self.monitor, textvariable=self.monitor_info,
                  font=("Segoe UI", 11, "bold")).pack(fill="x", pady=(2, 5))
        self.overall_label = tk.StringVar(value="Models complete 0 / 0")
        ttk.Label(self.monitor, textvariable=self.overall_label,
                  style="Muted.TLabel").pack(fill="x")
        self.overall = ttk.Progressbar(self.monitor, maximum=100, value=0)
        self.overall.pack(fill="x", pady=(2, 8))
        self.epoch_label = tk.StringVar(value="Waiting for first training epoch")
        ttk.Label(self.monitor, textvariable=self.epoch_label,
                  style="Muted.TLabel").pack(fill="x")
        self.epoch_bar = ttk.Progressbar(self.monitor, maximum=100, value=0)
        self.epoch_bar.pack(fill="x", pady=(2, 7))
        tabs = ttk.Notebook(self.monitor)
        tabs.pack(fill="both", expand=True, pady=(1, 7))
        ranking = ttk.Frame(tabs, padding=(5, 7))
        curves = ttk.Frame(tabs, padding=(5, 7))
        self.ranking_tab = tabs
        self.ranking_frame = ranking
        tabs.add(ranking, text="Validation screening")
        tabs.add(curves, text="Current model training curves")
        pick = ttk.Frame(ranking)
        pick.pack(fill="x", pady=(0, 6))
        ttk.Label(pick, text="Compare metric:").pack(side="left")
        self.metric_choice = tk.StringVar(value="mAP50–95")
        ttk.Combobox(pick, textvariable=self.metric_choice,
                     values=tuple(name for _key, name in METRIC_NAMES), state="readonly", width=16).pack(
                         side="left", padx=(8, 16))
        self.metric_choice.trace_add("write", lambda *_: self._update_bars())
        ttk.Button(pick, text="Open results folder", command=self.open_results).pack(side="right")
        ttk.Button(pick, text="Copy top checkpoint path", command=self.copy_top_checkpoint).pack(
            side="right", padx=(8, 0))
        ttk.Button(pick, text="Load previous comparison", command=self.load_previous).pack(side="right", padx=8)
        self.bars = ComparisonBars(ranking)
        self.bars.configure(height=255)
        self.bars.pack(fill="x")
        self.recommendation = tk.StringVar(value="Models will be compared by validation performance and training time.")
        ttk.Label(ranking, textvariable=self.recommendation, style="Muted.TLabel",
                  wraplength=850).pack(fill="x", pady=(4, 1))
        score_frame = ttk.Frame(ranking)
        score_frame.pack(fill="both", expand=True, pady=(8, 0))
        columns = ("model", "status", "precision", "recall", "map50", "map50_95", "best_epoch", "train_batch", "train_seconds", "weights_mb")
        self.score_table = ttk.Treeview(score_frame, columns=columns, show="headings", height=4)
        for column, title, width in (("model", "Model", 150), ("status", "Status", 97),
                                     ("precision", "Precision", 92), ("recall", "Recall", 92),
                                     ("map50", "mAP50", 92), ("map50_95", "mAP50–95", 92),
                                     ("best_epoch", "Peak val epoch", 100),
                                     ("train_batch", "Batch", 60),
                                     ("train_seconds", "Train sec", 78),
                                     ("weights_mb", "Weights MB", 85)):
            self.score_table.heading(column, text=title)
            self.score_table.column(column, width=width, minwidth=56,
                                    stretch=(column == "model"), anchor="center")
        self.score_table.pack(side="left", fill="both", expand=True)
        ybar = ttk.Scrollbar(score_frame, orient="vertical", command=self.score_table.yview)
        ybar.pack(side="right", fill="y")
        self.score_table.configure(yscrollcommand=ybar.set)
        self.score_table.bind("<Double-1>", self._details)
        self.live_graph = LiveMetricsGraph(curves)
        self.live_graph.pack(fill="both", expand=True)
        self.log_text = tk.Text(self.monitor, height=4, bg="#F8FAFD", bd=0,
                                font=("Consolas", 9), wrap="word")
        self.log_text.pack(fill="x", pady=(0, 3))
        self.log_text.configure(state="disabled")
        # Setup-first on entry; monitoring takes its place during the batch.
        self._setup_visible = True

    @staticmethod
    def _entry_row(parent, row, label, variable):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=variable, width=16).grid(row=row, column=1,
                                                                  sticky="w", pady=4)

    @staticmethod
    def _input_row(parent, row, label, variable, browse):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=4)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", pady=4)
        ttk.Button(parent, text="Browse…", command=browse).grid(row=row, column=2,
                                                                   sticky="w", padx=(10, 0))

    def _pick_file(self, target, label, formats):
        chosen = filedialog.askopenfilename(parent=self.winfo_toplevel(), title=label,
                                             filetypes=formats)
        if chosen:
            target.set(chosen)

    def _pick_dir(self, target):
        path = filedialog.askdirectory(parent=self.winfo_toplevel(), title="Comparison results folder")
        if path:
            target.set(path)

    def _set_augmentation(self, changes):
        self.augmentation = changes
        self.augmentation_label.configure(text=f"{len(changes)} overrides" if changes else "Library defaults")

    def add_pretrained(self):
        from train_and_split_tool_GUI import _build_model_filename
        try:
            label = _build_model_filename(self.version.get(), self.size.get())
            self.models.append({"version": self.version.get(), "size": self.size.get()})
            self._refresh_queue(select_last=True)
        except ValueError as exc:
            messagebox.showerror("Invalid model", str(exc), parent=self)

    def _append_unique_models(self, items):
        from train_and_split_tool_GUI import _build_model_filename
        existing = {str(Path(m["model_path"]).resolve()).lower() if m.get("model_path") else
                    _build_model_filename(m["version"], m["size"]) for m in self.models}
        for version, size in items:
            name = _build_model_filename(version, size)
            if name not in existing and len(self.models) < 20:
                self.models.append({"version": version, "size": size})
                existing.add(name)
        self._refresh_queue(select_last=True)

    def add_size_sweep(self):
        self._append_unique_models((self.version.get(), size) for size in ("n", "s", "m", "l", "x"))

    def add_generation_sweep(self):
        self._append_unique_models((version, size) for version in ("8", "10", "11", "12", "26")
                                   for size in ("m", "l"))

    def add_weights(self):
        filename = filedialog.askopenfilename(parent=self.winfo_toplevel(),
                                              title="Choose YOLO .pt weights", filetypes=[("YOLO weights", "*.pt")])
        if filename:
            self.models.append({"model_path": filename})
            self._refresh_queue(select_last=True)

    def _selected(self):
        sel = self.model_table.selection()
        return int(sel[0]) if sel else None

    def _refresh_queue(self, select_last=False):
        self.model_table.delete(*self.model_table.get_children())
        for i, model in enumerate(self.models):
            from train_and_split_tool_GUI import _build_model_filename
            label = Path(model["model_path"]).stem if model.get("model_path") else (
                _build_model_filename(model["version"], model["size"]).removesuffix(".pt"))
            source = model.get("model_path") or "Ultralytics pretrained weights"
            self.model_table.insert("", "end", iid=str(i), values=(i + 1, label, source))
        if self.models and select_last:
            self.model_table.selection_set(str(len(self.models) - 1))

    def _move(self, direction):
        index = self._selected()
        if index is None or not 0 <= index + direction < len(self.models):
            return
        self.models[index], self.models[index + direction] = (
            self.models[index + direction], self.models[index])
        self._refresh_queue()
        self.model_table.selection_set(str(index + direction))

    def _remove(self):
        index = self._selected()
        if index is not None:
            self.models.pop(index)
            self._refresh_queue()

    def settings(self):
        duration_mode = self.duration_type.get()
        duration_text = self.epochs.get().strip().replace(",", ".")
        return validate_settings({
            "existing_data_yaml": self.yaml.get().strip().strip('"'),
            "models": list(self.models), "screening_mode": self.screening.get(),
            "output_dir": self.output.get().strip().strip('"') or None,
            "epochs": int(duration_text) if duration_mode == "Epochs" else None,
            "time": float(duration_text) if duration_mode == "Hours" else None,
            "batch": int(self.batch.get()), "imgsz": int(self.imgsz.get()),
            "cache": self.cache.get(), "workers": int(self.workers.get()),
            "cls_weight": float(self.cls_weight.get().replace(",", ".")),
            "eval_batch": int(self.eval_batch.get()), "device": self.device.get(),
            "cleanup_disk_cache": self.cleanup.get(), "retry_oom": self.retry_oom.get(),
            "augmentation": dict(self.augmentation),
            "auto_install_cuda": self.auto_install_cuda.get(),
            "compact_progress": self.compact_progress.get(),
        })

    def _set_mode(self, setup):
        self._setup_visible = bool(setup)
        self.setup.pack_forget()
        self.monitor.pack_forget()
        if setup:
            self.setup.pack(fill="both", expand=True)
            self.toggle_button.configure(text="Show monitor")
        else:
            self.monitor.pack(fill="both", expand=True)
            self.toggle_button.configure(text="Show setup")
        if self.job or self.run_folder:
            self.toggle_button.pack(side="right", padx=9)

    def _toggle_setup(self):
        self._set_mode(not self._setup_visible)

    def _append_log(self, text):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", str(text)[:1000] + "\n")
        if int(self.log_text.index("end-1c").split(".")[0]) > 350:
            self.log_text.delete("1.0", "30.0")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        if self.on_log:
            self.on_log(text)

    def start(self):
        if self.busy:
            return
        if self.can_start is not None and not self.can_start():
            messagebox.showwarning("Another job running",
                "Finish the active training or pre-annotation job before model comparison.", parent=self)
            return
        try:
            settings = self.settings()
        except (ValueError, OSError, TypeError) as exc:
            messagebox.showerror("Invalid comparison settings", str(exc), parent=self)
            return
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        settings_file = self.logs_dir / f"comparison_{stamp}_settings.json"
        log_file = self.logs_dir / f"comparison_{stamp}.log"
        settings_file.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        python = self.env_python()
        env = self.process_env()
        flags = (subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0
        try:
            with log_file.open("w", encoding="utf-8") as output:
                proc = subprocess.Popen([str(python), "-u", str(Path(__file__).parent / "model_comparison_worker.py"),
                                         str(settings_file)], cwd=str(Path(__file__).parent), env=env,
                                        stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                        creationflags=flags, start_new_session=os.name != "nt")
        except OSError as exc:
            messagebox.showerror("Could not start comparison", str(exc), parent=self)
            return
        self.results = []
        self.screening_run = settings["screening_mode"]
        self._set_ranking_mode()
        self.run_folder = None
        self.summary_path = None
        self.job = {"process": proc, "file": log_file, "offset": 0, "pending": ""}
        self.stopped = False
        self.overall.configure(value=0)
        self.epoch_bar.configure(value=0)
        self.live_graph.reset()
        self._update_bars()
        self._refresh_table()
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status.set("Comparison running · training models sequentially")
        self._append_log(f"Started comparison process {proc.pid} · {log_file}")
        self._set_mode(False)

    def stop(self):
        if not self.busy:
            return
        if not messagebox.askyesno("Stop model comparison?",
                                   "Stop the comparison and its current training/test worker? "
                                   "Completed models and saved reports will remain on disk.", parent=self):
            return
        self.stopped = True
        proc = self.job["process"]
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                               capture_output=True, timeout=20, check=False,
                               creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                os.killpg(proc.pid, signal.SIGTERM)
        except (OSError, ProcessLookupError, subprocess.TimeoutExpired) as exc:
            self._append_log(f"Stop request error: {exc}")
        self.status.set("Stopping comparison…")
        self.stop_button.configure(state="disabled")

    def _apply_event(self, event):
        kind = event.get("kind")
        if kind == "started":
            self.run_folder = Path(event["output_dir"])
            self.monitor_info.set(f"Comparison started · {event['total']} models in queue")
        elif kind == "model":
            self.monitor_info.set(f"Model {event['index']}/{event['total']} · "
                                  f"{event['model']} · {event['phase']}")
            if event["phase"] == "training":
                self.live_graph.reset()
                self.epoch_bar.configure(value=0)
                self.epoch_label.set("Preparing training dataset / model")
            else:
                self.epoch_label.set("Evaluating best.pt on " + event["phase"].upper() + " split…")
                self.epoch_bar.configure(value=100)
        elif kind == "batch":
            ratio = event["iteration"] / max(1, event["iterations"])
            self.epoch_bar.configure(value=100 * ratio)
            self.epoch_label.set(f"Epoch {event['epoch']}/{event['epochs']} · "
                                 f"batch {event['iteration']}/{event['iterations']} · {ratio:.0%}")
        elif kind == "epoch":
            self.live_graph.record(event)
        elif kind == "result":
            row = event["result"]
            self.results.append(row)
            self.overall.configure(value=100 * event["index"] / event["total"])
            self.overall_label.set(f"Models complete {event['index']} / {event['total']}")
            self._refresh_table()
            self._update_bars()
        elif kind == "message":
            self._append_log(event.get("text", ""))
        elif kind == "finished":
            self.status.set("Comparison finished — " + event["status"])
            self.run_folder = Path(event["output_dir"])
            self.results = event["results"]
            self._update_bars()
            self._refresh_table()
            self._append_log("Report saved in " + str(self.run_folder))

    def poll(self):
        if not self.job:
            return
        job = self.job
        try:
            with job["file"].open("r", encoding="utf-8", errors="replace") as f:
                f.seek(job["offset"])
                content = f.read(180000)
                job["offset"] = f.tell()
            if content:
                lines = (job["pending"] + content).split("\n")
                job["pending"] = lines.pop()
                for line in lines:
                    if line:
                        event = parse_event(line)
                        if event is not None:
                            self._apply_event(event)
                        elif "Traceback" in line or "Error:" in line:
                            self._append_log(line)
            code = job["process"].poll()
            if code is not None and job["file"].stat().st_size <= job["offset"]:
                if job["pending"]:
                    event = parse_event(job["pending"])
                    if event:
                        self._apply_event(event)
                if self.stopped:
                    # A forced Windows taskkill cannot execute the child's finally
                    # block. Complete cache cleanup from its saved manifest.
                    if self.run_folder and self.run_folder.is_dir():
                        from train_and_split_tool_GUI import cleanup_disk_cache_manifest
                        for manifest in self.run_folder.glob("*_disk_cache_manifest.json"):
                            try:
                                count, errors = cleanup_disk_cache_manifest(manifest)
                                self._append_log(f"Post-stop image-cache cleanup: {count} files removed; {len(errors)} errors")
                            except (OSError, ValueError) as exc:
                                self._append_log(f"WARNING: Cleanup not completed: {exc}")
                    self.status.set("Comparison stopped. Completed model results preserved.")
                elif code != 0:
                    self.status.set(f"Comparison failed (exit {code}). Check the comparison log.")
                    self._append_log(f"Comparison failed. Full log: {job['file']}")
                self.job = None
                self.start_button.configure(state="normal")
                self.stop_button.configure(state="disabled")
                # Forced stop may precede final JSON write; previous models stay saved.
                if self.run_folder and (self.run_folder / "comparison.json").is_file():
                    try:
                        saved = json.loads((self.run_folder / "comparison.json").read_text(encoding="utf-8"))
                        self.results = saved.get("models", [])
                        self._update_bars()
                        self._refresh_table()
                    except (OSError, ValueError):
                        pass
        except OSError as exc:
            self._append_log("Comparison log read error: " + str(exc))

    def _refresh_table(self):
        self.score_table.delete(*self.score_table.get_children())
        for row in self.results:
            values = [row.get("model", ""), row.get("status", ""), *(
                "—" if row.get(key) is None else f"{float(row[key]):.3f}" for key in (
                    "precision", "recall", "map50", "map50_95")),
                row.get("best_epoch") or "—", row.get("train_batch", ""),
                row.get("train_seconds") if row.get("train_seconds") is not None else "—",
                row.get("weights_mb") if row.get("weights_mb") is not None else "—"]
            self.score_table.insert("", "end", values=values)

    def _details(self, _event=None):
        selected = self.score_table.selection()
        if selected:
            index = self.score_table.index(selected[0])
            if index < len(self.results):
                record = self.results[index]
                messagebox.showinfo("Model comparison details",
                    "\n".join(f"{key}: {value}" for key, value in record.items()), parent=self)

    def _set_ranking_mode(self):
        self.ranking_tab.tab(self.ranking_frame, text=("Validation screening" if self.screening_run
                                                     else "Independent test comparison"))
        self.bars.split_name = "Validation" if self.screening_run else "Independent test"
        self._update_bars()

    def _update_bars(self):
        by_label = {label: key for key, label in METRIC_NAMES}
        self.bars.update_data(self.results, by_label.get(self.metric_choice.get(), "map50_95"))
        completed = [r for r in self.results if r.get("status") == "completed" and
                     isinstance(r.get("map50_95"), (float, int))]
        if not completed:
            self.recommendation.set("Rank models on validation, then fine-tune finalists. Keep the test split for a final check.")
            return
        winner = max(completed, key=lambda r: r["map50_95"])
        timed = [r for r in completed if isinstance(r.get("train_seconds"), (float, int)) and
                 r["train_seconds"] > 0]
        speed = min(timed, key=lambda r: r["train_seconds"]) if timed else None
        self.recommendation.set(
            f"Highest {('validation' if self.screening_run else 'test')} mAP50–95: "
            f"{winner['model']} ({winner['map50_95']:.1%}). " +
            (f"Fastest completed training: {speed['model']} ({speed['train_seconds']/60:.1f} min). "
             if speed else "") +
            ("Use validation to shortlist architectures before an independent test." if self.screening_run
             else "Test results are for final comparison, not repeated hyperparameter selection."))

    def copy_top_checkpoint(self):
        completed = [record for record in self.results if record.get("status") == "completed"
                     and isinstance(record.get("map50_95"), (int, float))
                     and record.get("weights")]
        if not completed:
            messagebox.showinfo("No checkpoint selected", "Finish an evaluation first.", parent=self)
            return
        selected = self.score_table.selection()
        record = None
        if selected:
            index = self.score_table.index(selected[0])
            if index < len(self.results) and self.results[index] in completed:
                record = self.results[index]
        if record is None:
            record = max(completed, key=lambda r: r["map50_95"])
        self.clipboard_clear()
        self.clipboard_append(str(record["weights"]))
        self._append_log("Copied fine-tuning checkpoint path: " + str(record["weights"]))

    def open_results(self):
        if self.run_folder and self.run_folder.is_dir():
            try:
                open_folder(self.run_folder)
            except OSError as exc:
                messagebox.showerror("Could not open results folder", str(exc), parent=self)

    def load_previous(self):
        filename = filedialog.askopenfilename(parent=self.winfo_toplevel(),
                                              title="Select comparison.json", filetypes=[("Comparison results", "comparison.json")])
        if filename:
            try:
                content = json.loads(Path(filename).read_text(encoding="utf-8"))
                self.run_folder = Path(filename).parent
                self.results = content["models"]
                self.screening_run = bool(content.get("screening_mode", False))
                self.screening.set(self.screening_run)
                self._set_ranking_mode()
                self.monitor_info.set("Saved comparison · " + self.run_folder.name)
                self.status.set("Loaded previous comparison: " + content.get("status", "unknown"))
                self.overall.configure(value=100)
                self.overall_label.set(f"{len(self.results)} previous model results")
                self._refresh_table()
                self._update_bars()
                self._set_mode(False)
            except (KeyError, ValueError, OSError, TypeError) as exc:
                messagebox.showerror("Invalid comparison result", str(exc), parent=self)
