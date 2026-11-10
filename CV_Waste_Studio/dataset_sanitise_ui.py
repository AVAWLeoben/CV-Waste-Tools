"""Dataset sanitisation tab: preview first, explicit opt-in for deletion."""
from __future__ import annotations

from datetime import datetime
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

from dataset_sanitiser import load_plan, validate_settings


class DatasetSanitisePanel(ttk.Frame):
    def __init__(self, parent, *, env_python=None, export_python=None,
                 process_env=None, logs_dir=None, on_log=None, can_start=None):
        super().__init__(parent, padding=18)
        self.env_python = env_python or (lambda: sys.executable)
        self.export_python = export_python or self.env_python
        self.process_env = process_env or (lambda: dict(os.environ))
        self.logs_dir = Path(logs_dir or (Path(__file__).parent / "logs"))
        self.on_log = on_log or (lambda message: None)
        self.can_start = can_start or (lambda: True)
        self.job = None
        self.plan_path = None
        self.preview_settings = None
        self.summary = None
        self.busy = False
        self.changed_during_scan = False
        self.method = tk.StringVar(value="labels")
        self.action = tk.StringVar(value="copy")
        self.source = tk.StringVar()
        self.output = tk.StringVar()
        self.model = tk.StringVar()
        self.conf = tk.StringVar(value="0.25")
        self.imgsz = tk.StringVar(value="640")
        self.batch = tk.StringVar(value="1")
        self.device = tk.StringVar(value="auto")
        self.include_missing = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Select a dataset and scan to preview any removals.")
        self.columnconfigure(1, weight=1)
        self.rowconfigure(12, weight=1)
        ttk.Label(self, text="Sanitise Datasets", font=("Segoe UI", 12, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 5))
        ttk.Label(self, text="Preview all candidates before making changes. Source files are unchanged during scanning.",
                  wraplength=730, foreground="#60718A").grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 11))

        def caption(row, text):
            ttk.Label(self, text=text).grid(row=row, column=0, sticky="w", pady=4, padx=(0, 10))

        caption(2, "Dataset folder")
        ttk.Entry(self, textvariable=self.source).grid(row=2, column=1, sticky="ew", pady=4)
        ttk.Button(self, text="Browse…", command=self._choose_source).grid(row=2, column=2, padx=(9, 0))

        caption(3, "How to find empty images")
        methods = ttk.Frame(self)
        methods.grid(row=3, column=1, columnspan=2, sticky="w")
        ttk.Radiobutton(methods, text="Empty .txt labels", variable=self.method, value="labels").pack(side="left", padx=(0, 18))
        ttk.Radiobutton(methods, text="YOLO model finds no objects", variable=self.method, value="model").pack(side="left")

        caption(4, "Model (.pt / .engine / .onnx)")
        self.model_entry = ttk.Entry(self, textvariable=self.model)
        self.model_entry.grid(row=4, column=1, sticky="ew", pady=4)
        self.model_browse = ttk.Button(self, text="Browse…", command=self._choose_model)
        self.model_browse.grid(row=4, column=2, padx=(9, 0))
        caption(5, "Inference options")
        controls = ttk.Frame(self)
        controls.grid(row=5, column=1, columnspan=2, sticky="w")
        self.model_controls = []
        for label, variable, width in (("Conf", self.conf, 6), ("Size", self.imgsz, 7), ("Batch", self.batch, 5)):
            child = ttk.Label(controls, text=label)
            child.pack(side="left", padx=(0, 4))
            field = ttk.Entry(controls, textvariable=variable, width=width)
            field.pack(side="left", padx=(0, 12))
            self.model_controls.append((child, field))
        self.device_label = ttk.Label(controls, text="Device")
        self.device_label.pack(side="left", padx=(0, 4))
        self.device_combo = ttk.Combobox(controls, values=("auto", "0", "cpu"),
                                         textvariable=self.device, width=6, state="readonly")
        self.device_combo.pack(side="left")
        self.missing_toggle = ttk.Checkbutton(
            self, text="Also remove images with missing .txt labels (normally only empty existing .txt files)",
            variable=self.include_missing)
        self.missing_toggle.grid(row=6, column=1, columnspan=2, sticky="w", pady=(4, 10))

        caption(7, "Output behaviour")
        destinations = ttk.Frame(self)
        destinations.grid(row=7, column=1, columnspan=2, sticky="w")
        ttk.Radiobutton(destinations, text="Create cleaned copy (recommended)",
                        variable=self.action, value="copy").pack(side="left", padx=(0, 16))
        ttk.Radiobutton(destinations, text="Delete from original dataset", variable=self.action,
                        value="inplace").pack(side="left")
        caption(8, "New dataset folder")
        self.output_entry = ttk.Entry(self, textvariable=self.output)
        self.output_entry.grid(row=8, column=1, sticky="ew", pady=4)
        self.output_browse = ttk.Button(self, text="Choose…", command=self._choose_output)
        self.output_browse.grid(row=8, column=2, padx=(9, 0))
        self.danger_label = ttk.Label(self, text="In-place mode permanently deletes selected images and paired labels. "
            "No automatic undo. Preview and a typed confirmation are required.",
            foreground="#B53A43", wraplength=770)
        self.danger_label.grid(row=9, column=0, columnspan=3, sticky="w", pady=(5, 9))

        actions = ttk.Frame(self)
        actions.grid(row=10, column=0, columnspan=3, sticky="ew", pady=(1, 6))
        self.scan_btn = ttk.Button(actions, text="1. Scan / preview", command=self.start_scan)
        self.scan_btn.pack(side="left")
        self.apply_btn = ttk.Button(actions, text="2. Apply sanitisation", command=self.apply, state="disabled")
        self.apply_btn.pack(side="left", padx=(10, 0))
        self.preview_btn = ttk.Button(actions, text="Save candidate list…", command=self.save_candidates, state="disabled")
        self.preview_btn.pack(side="left", padx=(10, 0))
        self.stop_btn = ttk.Button(actions, text="Stop scan", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=(10, 0))
        self.progress = ttk.Progressbar(actions, mode="indeterminate", length=95)
        self.progress.pack(side="right", padx=(12, 0))
        ttk.Label(self, textvariable=self.status, foreground="#405DE6", wraplength=780).grid(
            row=11, column=0, columnspan=3, sticky="w", pady=(0, 5))
        results = ttk.Frame(self)
        results.grid(row=12, column=0, columnspan=3, sticky="nsew")
        results.columnconfigure(0, weight=1)
        results.rowconfigure(0, weight=1)
        self.preview = tk.Text(results, height=7, wrap="word", state="disabled",
                               font=("Consolas", 9), bg="#F8FAFD", fg="#19263B",
                               highlightthickness=1, highlightbackground="#DAE2EF")
        self.preview.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(results, orient="vertical", command=self.preview.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.preview.configure(yscrollcommand=scrollbar.set)
        self._write("Preview will list candidate image/label removals and progress.\n")
        for var in (self.method, self.action, self.source, self.output, self.model,
                    self.conf, self.imgsz, self.batch, self.device, self.include_missing):
            var.trace_add("write", self._invalidate)
        self._refresh_method()
        self.method.trace_add("write", lambda *_: self._refresh_method())
        self.action.trace_add("write", lambda *_: self._refresh_action())
        self._refresh_action()

    def _write(self, line):
        self.preview.configure(state="normal")
        self.preview.insert("end", str(line).rstrip("\n") + "\n")
        if int(self.preview.index("end-1c").split(".")[0]) > 350:
            self.preview.delete("1.0", "75.0")
        self.preview.see("end")
        self.preview.configure(state="disabled")

    def _invalidate(self, *_):
        if self.busy and self.job and self.job["phase"] == "scan":
            self.changed_during_scan = True
            self.status.set("Settings changed during scan; re-scan when it finishes.")
        if self.preview_settings is not None and not self.busy:
            self.plan_path = None
            self.preview_settings = None
            self.summary = None
            self.apply_btn.configure(state="disabled")
            self.preview_btn.configure(state="disabled")
            self.status.set("Settings changed. Scan again before applying changes.")

    def _refresh_method(self):
        model_mode = self.method.get() == "model"
        self.model_entry.configure(state="normal" if model_mode else "disabled")
        self.model_browse.configure(state="normal" if model_mode else "disabled")
        for label, field in self.model_controls:
            field.configure(state="normal" if model_mode else "disabled")
        self.device_combo.configure(state="readonly" if model_mode else "disabled")
        self.missing_toggle.configure(state="disabled" if model_mode else "normal")

    def _refresh_action(self):
        copy = self.action.get() == "copy"
        self.output_entry.configure(state="normal" if copy else "disabled")
        self.output_browse.configure(state="normal" if copy else "disabled")

    def _choose_source(self):
        folder = filedialog.askdirectory(parent=self, title="Select YOLO dataset root or images folder")
        if folder:
            from dataset_sanitiser import _root
            root = _root(folder)
            self.source.set(folder)
            if not self.output.get():
                self.output.set(str(root.parent / (root.name + "_sanitised")))

    def _choose_model(self):
        name = filedialog.askopenfilename(parent=self, title="Choose YOLO model",
            filetypes=[("YOLO models", "*.pt *.engine *.onnx"), ("All files", "*.*")])
        if name:
            self.model.set(name)

    def _choose_output(self):
        # Tk askdirectory returns an existing directory. Choose its parent and append
        # a NEW folder name, so existing datasets are never overwritten.
        folder = filedialog.askdirectory(parent=self, title="Select parent folder for cleaned dataset")
        if folder:
            src = Path(self.source.get().strip())
            name = src.name if src.name else "dataset"
            self.output.set(str(Path(folder) / (name + "_sanitised")))

    def _settings(self):
        return validate_settings({"source": self.source.get(), "method": self.method.get(),
            "action": self.action.get(), "output": self.output.get(), "model": self.model.get(),
            "conf": self.conf.get(), "imgsz": self.imgsz.get(), "batch": self.batch.get(),
            "device": self.device.get(), "include_missing": self.include_missing.get()})

    def _launch(self, phase, *, settings=None):
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        job_file = self.logs_dir / f"sanitise_{phase}_{stamp}_settings.json"
        log_file = self.logs_dir / f"sanitise_{phase}_{stamp}.log"
        if phase == "scan":
            self.plan_path = self.logs_dir / f"sanitise_{stamp}_preview.json"
        job = {"phase": phase, "plan": str(self.plan_path)}
        if settings is not None:
            job["settings"] = settings
        job_file.write_text(json.dumps(job, indent=2), encoding="utf-8")
        python = str(self.env_python())
        env = dict(self.process_env())
        if phase == "scan" and settings and settings["method"] == "model" and Path(settings["model"]).suffix.lower() in {".engine", ".onnx"}:
            from export_environment import make_export_process_env
            python = str(self.export_python())
            env = make_export_process_env(env, python, str(self.env_python()))
        env["YOLO_AUTOINSTALL"] = "False"
        command = [python, "-u", str(Path(__file__).with_name("run_dataset_sanitiser.py")), str(job_file)]
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        with log_file.open("w", encoding="utf-8") as out:
            proc = subprocess.Popen(command, cwd=str(Path(__file__).parent), env=env,
                stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                creationflags=flags, start_new_session=os.name != "nt")
        self.job = {"process": proc, "phase": phase, "log": log_file, "offset": 0}
        self.busy = True
        self.changed_during_scan = False
        self.scan_btn.configure(state="disabled")
        self.apply_btn.configure(state="disabled")
        self.preview_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal" if phase == "scan" else "disabled")
        self.progress.start(13)
        self.status.set(("Scanning" if phase == "scan" else "Applying") + " dataset…")
        self._write(f"{phase.upper()} started with Python: {python}\nLog: {log_file}")
        self.on_log(f"Sanitisation {phase} started. Log: {log_file}")

    def start_scan(self):
        if self.busy:
            return
        if not self.can_start():
            messagebox.showwarning("Another job is active", "Finish the active training, evaluation or export job first.", parent=self)
            return
        try:
            settings = self._settings()
            self.preview_settings = None
            self.summary = None
            self.apply_btn.configure(state="disabled")
            self.preview_btn.configure(state="disabled")
            self._launch("scan", settings=settings)
            self.preview_settings = settings
        except (OSError, ValueError) as exc:
            self.status.set("Cannot scan: " + str(exc))
            messagebox.showerror("Sanitise Datasets", str(exc), parent=self)

    def save_candidates(self):
        if not self.summary or not self.plan_path:
            return
        target = filedialog.asksaveasfilename(parent=self, title="Save sanitisation candidate list",
            defaultextension=".csv", initialfile="sanitise_candidates.csv",
            filetypes=[("CSV list", "*.csv")])
        if not target:
            return
        try:
            with Path(target).open("w", newline="", encoding="utf-8-sig") as out:
                writer = csv.writer(out)
                writer.writerow(["image_to_remove", "paired_label_to_remove"])
                for record in self.summary["remove"]:
                    writer.writerow([record["image"], record["label"] or ""])
            self._write("Candidate list saved: " + target)
        except OSError as exc:
            messagebox.showerror("Cannot save preview", str(exc), parent=self)

    def apply(self):
        if self.busy or self.plan_path is None or not self.summary:
            return
        if not self.can_start():
            messagebox.showwarning("Another job is active", "Finish the current job first.", parent=self)
            return
        try:
            current = self._settings()
            if current != self.preview_settings:
                raise ValueError("Settings changed since preview; scan the dataset again.")
            # Preflight validation and snapshots are checked again by the worker.
            n = len(self.summary["remove"])
            if current["action"] == "inplace":
                answer = simpledialog.askstring("Confirm permanent deletion",
                    f"Permanently delete {n} images and any paired labels from:\n\n"
                    f"{current['source']}\n\nThis cannot be undone.\nType DELETE to confirm:", parent=self)
                if answer != "DELETE":
                    return
            else:
                if not messagebox.askyesno("Create cleaned dataset",
                    f"Copy {self.summary['total']-n} retained images to:\n{current['output']}\n\n"
                    f"The original dataset remains unchanged. Continue?", parent=self):
                    return
            self._launch("apply")
        except (OSError, ValueError) as exc:
            messagebox.showerror("Cannot sanitise dataset", str(exc), parent=self)

    def stop(self):
        if not self.job or self.job["phase"] != "scan" or not self.busy:
            return
        if not messagebox.askyesno("Stop scan?", "Stop scanning? No dataset files have been modified.", parent=self):
            return
        proc = self.job["process"]
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                               capture_output=True, timeout=15, check=False,
                               creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                os.killpg(proc.pid, signal.SIGTERM)
        except (OSError, ProcessLookupError, subprocess.TimeoutExpired):
            proc.terminate()
        self.status.set("Stopping scan…")
        self.stop_btn.configure(state="disabled")

    def poll(self):
        if not self.job:
            return
        job = self.job
        try:
            with job["log"].open("r", encoding="utf-8", errors="replace") as stream:
                stream.seek(job["offset"])
                text = stream.read()
                job["offset"] = stream.tell()
            if text:
                lines = text.rstrip().splitlines()
                for line in lines[-15:]:
                    self._write(line)
        except OSError:
            pass
        rc = job["process"].poll()
        if rc is None:
            return
        self.job = None
        self.busy = False
        self.progress.stop()
        self.stop_btn.configure(state="disabled")
        self.scan_btn.configure(state="normal")
        if rc:
            self.plan_path = None
            self.summary = None
            self.preview_settings = None
            self.status.set(f"{job['phase'].capitalize()} failed (exit {rc}). See log for details.")
            self.on_log("Sanitisation failed; see " + str(job["log"]))
            return
        if job["phase"] == "scan":
            if self.changed_during_scan:
                self.plan_path = None
                self.preview_settings = None
                self.summary = None
                self.status.set("Settings changed during scan; run Scan / preview again.")
                return
            try:
                self.summary = load_plan(self.plan_path)
                n = len(self.summary["remove"])
                self.status.set(f"Preview ready: {n} images to remove, "
                                f"{self.summary['total'] - n} retained. No files changed.")
                self._write("\nPREVIEW — first 60 candidates:")
                for record in self.summary["remove"][:60]:
                    self._write(" • " + record["image"] + (" + " + record["label"] if record["label"] else " (no label)"))
                if n > 60:
                    self._write(f"…and {n-60} more; complete list in {self.plan_path}")
                self.apply_btn.configure(state="normal")
                self.preview_btn.configure(state="normal")
            except (OSError, ValueError, KeyError) as exc:
                self.status.set("Preview could not be loaded: " + str(exc))
                self.plan_path = None
        else:
            self.status.set("Sanitisation complete. See log for destination and summary.")
            self.apply_btn.configure(state="disabled")
            self.preview_btn.configure(state="disabled")
            self.plan_path = None
            self.summary = None
            self.preview_settings = None
        self.on_log(self.status.get())
