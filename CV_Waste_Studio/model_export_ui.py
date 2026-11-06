"""CV Waste Studio model-export panel: platform-aware native YOLO export."""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import webbrowser
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from model_export import validate_export_settings
from export_environment import (discover_export_environments, make_export_process_env,
                                load_selection, save_selection)

ACCENT = "#405DE6"
MUTED = "#60718A"
BORDER = "#DAE2EF"
FORMATS = {
    "onnx": ("ONNX", "Portable · CPU / GPU runtimes"),
    "engine": ("TensorRT Engine", "NVIDIA GPU · hardware-specific"),
    "litert": ("LiteRT", "Edge / CV-Waste Everywhere · Linux or Windows WSL2"),
}


class ModelExportPanel(tk.Frame):
    def __init__(self, parent, *, env_python, process_env, logs_dir,
                 can_start=None, on_log=None):
        super().__init__(parent, bg="white")
        self.env_python = env_python
        self.process_env = process_env
        self.logs_dir = Path(logs_dir)
        self.can_start = can_start
        self.on_log = on_log
        self.compat = None
        self.job = None
        self.install_job = None
        self._probe_pending = False
        self._probe_queue = queue.Queue()
        self._prompted_for_trt = False
        self._prompted_for_wsl = False
        self._last_result = None
        self._expanded_setup = False
        self._tab_opened = False
        self._env_options = {}
        self._build()
        self.after(500, self.check_compatibility)

    @property
    def busy(self):
        # Installing backend dependencies also modifies the managed runtime;
        # prevent training/repair from running concurrently with pip.
        return bool((self.job and self.job["process"].poll() is None) or
                    (self.install_job and self.install_job["process"].poll() is None))

    def _emit(self, message):
        if self.on_log:
            self.on_log(str(message))

    def _build(self):
        main = ttk.Frame(self, padding=(18, 13))
        main.pack(fill="both", expand=True)
        top = ttk.Frame(main)
        top.pack(fill="x", pady=(0, 7))
        ttk.Label(top, text="Export a trained model", font=("Segoe UI", 14, "bold")).pack(side="left")
        self.stop_btn = ttk.Button(top, text="Stop export", style="Danger.TButton",
                                   state="disabled", command=self.stop)
        self.stop_btn.pack(side="right")
        self.toggle_btn = ttk.Button(top, text="Show settings", command=self.toggle_setup)
        self.toggle_btn.pack(side="right", padx=8)
        self.toggle_btn.pack_forget()
        self.status = tk.StringVar(value="Checking export capabilities in the selected Conda environment…")
        ttk.Label(main, textvariable=self.status, style="Muted.TLabel",
                  wraplength=800).pack(fill="x", pady=(0, 7))
        self.setup = ttk.Frame(main)
        self.setup.pack(fill="both", expand=True)
        canvas = tk.Canvas(self.setup, bd=0, highlightthickness=0, bg="white")
        scrollbar = ttk.Scrollbar(self.setup, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        body = ttk.Frame(canvas, padding=(3, 3, 16, 14))
        window = canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>", lambda _: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window, width=e.width))
        self._setup_canvas = canvas

        def wheel(event):
            step = -1 if getattr(event, "num", None) == 4 else (
                1 if getattr(event, "num", None) == 5 else -int(event.delta / 120))
            canvas.yview_scroll(step, "units")
        def wheel_on(_):
            for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                canvas.bind_all(ev, wheel)
        def wheel_off(_):
            for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                canvas.unbind_all(ev)
        canvas.bind("<Enter>", wheel_on)
        canvas.bind("<Leave>", wheel_off)
        body.columnconfigure(1, weight=1)

        ttk.Label(body, text="01  Target format & environment", font=("Segoe UI", 12, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
        self.fmt = tk.StringVar(value="onnx")
        self.radios = {}
        self.format_status = {}
        for index, (fmt, (name, desc)) in enumerate(FORMATS.items(), start=1):
            wrapper = ttk.Frame(body)
            wrapper.grid(row=index, column=0, columnspan=3, sticky="ew", pady=2)
            radio = ttk.Radiobutton(wrapper, text=name, variable=self.fmt, value=fmt,
                                    command=self._format_changed, state="disabled")
            radio.pack(side="left", padx=(4, 10))
            ttk.Label(wrapper, text=desc, style="Muted.TLabel").pack(side="left", padx=(0, 8))
            note = tk.StringVar(value="Checking…")
            ttk.Label(wrapper, textvariable=note, style="Muted.TLabel").pack(side="right", padx=4)
            self.radios[fmt] = radio
            self.format_status[fmt] = note
        compatibility_block = ttk.Frame(body)
        compatibility_block.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(5, 10))
        environment_row = ttk.Frame(compatibility_block)
        environment_row.pack(fill="x", pady=(0, 5))
        ttk.Label(environment_row, text="Export Python environment", width=27).pack(side="left")
        self.export_env_choice = tk.StringVar()
        self.env_combo = ttk.Combobox(environment_row, textvariable=self.export_env_choice,
                                      state="readonly", width=52)
        self.env_combo.pack(side="left", padx=(11, 5), fill="x", expand=True)
        self.env_combo.bind('<<ComboboxSelected>>', self._environment_changed)
        ttk.Button(environment_row, text="Refresh", command=self.refresh_environments).pack(side="left", padx=3)
        ttk.Button(environment_row, text="Browse Python…", command=self.browse_environment).pack(side="left", padx=3)
        compatibility_row = ttk.Frame(compatibility_block)
        compatibility_row.pack(fill="x")
        ttk.Button(compatibility_row, text="Recheck environment",
                   command=self.check_compatibility).pack(side="left")
        self.install_btn = ttk.Button(compatibility_row, text="Install TensorRT…",
                                      state="disabled", command=self.install_tensorrt)
        self.install_btn.pack(side="left", padx=9)
        self.wsl_setup_btn = ttk.Button(compatibility_row, text="WSL status / setup…",
                                        command=self.setup_wsl, state="disabled")
        self.wsl_setup_btn.pack(side="left", padx=5)
        ttk.Label(compatibility_row,
                  text="External/Jetson environments are read-only. No automatic package installation during export.",
                  style="Muted.TLabel", wraplength=390).pack(side="left", fill="x", expand=True)

        ttk.Separator(body).grid(row=5, column=0, columnspan=3, sticky="ew", pady=(6, 12))
        ttk.Label(body, text="02  Source model & destination", font=("Segoe UI", 12, "bold")).grid(
            row=6, column=0, columnspan=3, sticky="w", pady=(0, 7))
        self.model = tk.StringVar()
        self.model_name = tk.StringVar()
        self.output = tk.StringVar(value=str(self.logs_dir / "exports"))
        self._row(body, 7, "YOLO checkpoint (.pt)", self.model,
                  lambda: self._choose_file(self.model, "Select trained model", [("PyTorch model", "*.pt")]))
        self._row(body, 8, "Exported model name", self.model_name)
        ttk.Label(body, text="For example: YOLO11L_1280_FP16. Extension is added automatically; blank = original name.",
                  style="Muted.TLabel").grid(row=9, column=1, columnspan=2, sticky="w", pady=(0, 5))
        self._row(body, 10, "Export root folder", self.output,
                  lambda: self._choose_folder(self.output))
        ttk.Label(body, text="Each export gets its own dated subfolder. The source checkpoint is never modified.",
                  style="Muted.TLabel").grid(row=11, column=1, columnspan=2, sticky="w", pady=(0, 5))

        ttk.Separator(body).grid(row=12, column=0, columnspan=3, sticky="ew", pady=(6, 12))
        ttk.Label(body, text="03  Export parameters", font=("Segoe UI", 12, "bold")).grid(
            row=13, column=0, columnspan=3, sticky="w", pady=(0, 7))
        self.imgsz = tk.StringVar(value="640")
        self.batch = tk.StringVar(value="1")
        self.device = tk.StringVar(value="auto")
        self.precision = tk.StringVar(value="fp32")
        self.dynamic = tk.BooleanVar(value=False)
        self.simplify = tk.BooleanVar(value=True)
        self.nms = tk.BooleanVar(value=False)
        self.opset = tk.StringVar()
        self.workspace = tk.StringVar()
        self.data = tk.StringVar()
        self.fraction = tk.StringVar(value="1.0")
        self._row(body, 14, "Image size (H×W or square)", self.imgsz)
        ttk.Label(body, text="Examples: 640, 960, 1088, 640x1088. Ultralytics aligns sizes to the model stride.",
                  style="Muted.TLabel").grid(row=15, column=1, columnspan=2, sticky="w", pady=(0, 5))
        self._row(body, 16, "Inference batch", self.batch)
        ttk.Label(body, text="Precision").grid(row=17, column=0, sticky="w", pady=4)
        self.precision_combo = ttk.Combobox(body, textvariable=self.precision, state="readonly",
                                            values=("fp32", "fp16", "int8"), width=16)
        self.precision_combo.grid(row=17, column=1, sticky="w", pady=4)
        self.precision_combo.bind("<<ComboboxSelected>>", lambda _: self._format_changed())
        ttk.Label(body, text="Device").grid(row=18, column=0, sticky="w", pady=4)
        self.device_combo = ttk.Combobox(body, textvariable=self.device,
                                         values=("auto", "0", "cpu"), state="readonly", width=16)
        self.device_combo.grid(row=18, column=1, sticky="w", pady=4)
        flags = ttk.Frame(body)
        flags.grid(row=19, column=1, columnspan=2, sticky="w", pady=5)
        self.dynamic_check = ttk.Checkbutton(flags, text="Dynamic input shapes", variable=self.dynamic)
        self.dynamic_check.pack(side="left", padx=(0, 18))
        self.simplify_check = ttk.Checkbutton(flags, text="Simplify ONNX graph", variable=self.simplify)
        self.simplify_check.pack(side="left", padx=(0, 18))
        self.nms_check = ttk.Checkbutton(flags, text="Embed NMS", variable=self.nms)
        self.nms_check.pack(side="left")
        self._row(body, 20, "ONNX opset (optional)", self.opset)
        self.workspace_row = ttk.Frame(body)
        self.workspace_row.grid(row=21, column=0, columnspan=3, sticky="ew", pady=2)
        ttk.Label(self.workspace_row, text="TensorRT workspace (GiB)", width=27).pack(side="left")
        self.workspace_entry = ttk.Entry(self.workspace_row, textvariable=self.workspace, width=16)
        self.workspace_entry.pack(side="left", padx=(11, 4))
        ttk.Label(self.workspace_row, text="Blank = TensorRT auto", style="Muted.TLabel").pack(side="left", padx=12)
        self._row(body, 22, "INT8 calibration data.yaml", self.data,
                  lambda: self._choose_file(self.data, "Choose representative calibration data.yaml",
                                            [("Dataset YAML", "*.yaml *.yml")]))
        self._row(body, 23, "INT8 calibration fraction", self.fraction)
        self.calibration_note = ttk.Label(body, text="INT8 requires representative images; avoid unrelated default datasets.",
                                          style="Muted.TLabel")
        self.calibration_note.grid(row=24, column=1, columnspan=2, sticky="w", pady=(2, 5))
        self._format_changed()
        self.refresh_environments(initial=True)

        footer = ttk.Frame(main)
        footer.pack(fill="x", pady=(9, 4))
        self.start_btn = ttk.Button(footer, text="Export model", style="Accent.TButton",
                                    command=self.start, state="disabled")
        self.start_btn.pack(side="left")
        self.open_btn = ttk.Button(footer, text="Open Results Folder",
                                   command=self.open_results, state="disabled")
        self.open_btn.pack(side="left", padx=8)
        self.progress = ttk.Progressbar(footer, mode="indeterminate")
        self.progress.pack(side="right", fill="x", expand=True, padx=(14, 0))
        self.progress.pack_forget()

        self.monitor = ttk.Frame(main)
        self.monitor.pack(fill="x", pady=(5, 1))
        ttk.Label(self.monitor, text="EXPORT ACTIVITY", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        monitor_frame = ttk.Frame(self.monitor)
        monitor_frame.pack(fill="both", expand=True, pady=(3, 0))
        self.log_text = tk.Text(monitor_frame, height=3, wrap="word", bg="#F8FAFD",
                                fg="#31445D", font=("Consolas", 9), padx=8, pady=7,
                                relief="flat", state="disabled")
        self.log_text.pack(side="left", fill="both", expand=True)
        bar = ttk.Scrollbar(monitor_frame, command=self.log_text.yview)
        bar.pack(side="right", fill="y")
        self.log_text.config(yscrollcommand=bar.set)

    def _row(self, parent, row, caption, var, browse=None):
        ttk.Label(parent, text=caption).grid(row=row, column=0, sticky="w", padx=(0, 15), pady=4)
        ttk.Entry(parent, textvariable=var, width=44).grid(row=row, column=1, sticky="ew", pady=4)
        if browse:
            ttk.Button(parent, text="Browse…", command=browse).grid(row=row, column=2, padx=(10, 0), pady=4)

    def _choose_file(self, variable, title, types):
        selected = filedialog.askopenfilename(parent=self, title=title, filetypes=types)
        if selected:
            variable.set(selected)

    def _choose_folder(self, variable):
        selected = filedialog.askdirectory(parent=self, title="Choose export destination")
        if selected:
            variable.set(selected)

    def _format_changed(self):
        fmt = self.fmt.get()
        if fmt == "litert":
            allowed = ("fp32", "int8")
            self.dynamic.set(False)
            self.nms.set(False)
            self.opset.set("")
            self.device.set("cpu")
        else:
            allowed = ("fp32", "fp16", "int8")
            if fmt == "engine":
                self.device.set("0")
        self.precision_combo.configure(values=allowed)
        if self.precision.get() not in allowed:
            self.precision.set("fp32")
        for widget, supported in ((self.dynamic_check, fmt != "litert"),
                                  (self.simplify_check, fmt != "litert"),
                                  (self.nms_check, fmt != "litert")):
            widget.configure(state="normal" if supported else "disabled")
        self.workspace_entry.configure(state="normal" if fmt == "engine" else "disabled")
        self.device_combo.configure(state="disabled" if fmt in ("engine", "litert") else "readonly")
        # The format radio buttons are the source of truth. Update the summary
        # immediately; a capability probe may have reported ONNX before the
        # operator subsequently selected TensorRT Engine.
        self._update_target_status()

    def _update_target_status(self):
        """Keep the visible target and Export button synced with the radio selection."""
        if not self.compat:
            return
        fmt = self.fmt.get()
        entry = self.compat.get("formats", {}).get(fmt, {})
        ready = bool(entry.get("available"))
        if hasattr(self, "start_btn"):
            self.start_btn.configure(state=(
                "normal" if ready and not self.busy and not self._probe_pending else "disabled"))
        name = FORMATS.get(fmt, (fmt.upper(), ""))[0]
        if ready:
            gpu = str(self.compat.get("gpu")) if self.compat.get("cuda") else "CPU/no CUDA GPU"
            self.status.set("Export capabilities checked · Ultralytics " +
                            str(self.compat.get("ultralytics") or "missing") +
                            " · " + gpu + " · Target: " + name + " (ready)")
        else:
            reason = str(entry.get("reason", "Unknown compatibility issue"))
            self.status.set("Selected target: " + name +
                            " — unavailable: " + reason +
                            ". Select a compatible export environment or recheck.")

    def toggle_setup(self):
        self._expanded_setup = not self._expanded_setup
        if self._expanded_setup:
            self.setup.pack(before=self.monitor, fill="both", expand=True)
            self.monitor.pack_configure(fill="x", expand=False)
            self.log_text.configure(height=3)
        else:
            self.setup.pack_forget()
            self.monitor.pack_configure(fill="both", expand=True)
            self.log_text.configure(height=22)
        self.toggle_btn.configure(text="Hide settings" if self._expanded_setup else "Show settings")

    def _append(self, line):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", str(line).rstrip("\n") + "\n")
        if int(self.log_text.index("end-1c").split(".")[0]) > 1200:
            self.log_text.delete("1.0", "120.0")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _export_python(self):
        return self._env_options.get(self.export_env_choice.get(), str(self.env_python()))

    def _export_env(self):
        return make_export_process_env(dict(self.process_env()), self._export_python(),
                                       str(self.env_python()))

    def _external_export_python(self):
        return Path(self._export_python()).resolve() != Path(self.env_python()).resolve()

    def refresh_environments(self, initial=False):
        preferred = load_selection() if initial else self._export_python()
        items = discover_export_environments(str(self.env_python()))
        self._env_options = {item['label'] + '  |  ' + item['python']: item['python'] for item in items}
        if preferred and Path(preferred).is_file() and preferred not in self._env_options.values():
            self._env_options['Custom · ' + preferred] = preferred
        self.env_combo.configure(values=tuple(self._env_options))
        name = next((label for label, python in self._env_options.items() if python == preferred), None)
        if name is None:
            name = next(iter(self._env_options), '')
        self.export_env_choice.set(name)
        if not initial:
            self._environment_changed()

    def browse_environment(self):
        path = filedialog.askopenfilename(parent=self, title='Select Python used for model export')
        if not path:
            return
        if not Path(path).is_file():
            messagebox.showerror('Invalid environment', 'Select an existing Python executable.', parent=self)
            return
        resolved = str(Path(path).resolve())
        label = 'Custom · ' + resolved
        self._env_options[label] = resolved
        self.env_combo.configure(values=tuple(self._env_options))
        self.export_env_choice.set(label)
        self._environment_changed()

    def _environment_changed(self, _event=None):
        if self.busy:
            return
        python = self._export_python()
        if python and Path(python).is_file():
            save_selection(python)
            self._prompted_for_trt = False
            self.compat = None
            self._append('Export environment selected: ' + python)
            self.check_compatibility()

    def _can_install_tensorrt(self):
        return (not self._external_export_python() and bool(self.compat)
                and not self.compat.get('jetson') and bool(self.compat.get('cuda'))
                and bool(self.compat.get('install_package')))

    def check_compatibility(self):
        if self._probe_pending or self.install_job is not None or self.busy:
            return
        self._probe_pending = True
        self.status.set("Checking export formats in: " + self._export_python())
        for radio in self.radios.values():
            radio.configure(state="disabled")
        self.start_btn.configure(state="disabled")
        self.install_btn.configure(state="disabled")
        self.wsl_setup_btn.configure(state="disabled")
        python = str(self._export_python())
        env = self._export_env()
        script = str(Path(__file__).with_name("probe_model_export.py"))
        def worker():
            try:
                result = subprocess.run([python, "-u", script], cwd=str(Path(script).parent),
                                        env=env, capture_output=True, text=True, errors="replace",
                                        timeout=150, check=False)
                payload = next((line.split("=", 1)[1] for line in result.stdout.splitlines()
                                if line.startswith("STUDIO_EXPORT_PROBE=")), None)
                if not payload:
                    raise RuntimeError((result.stderr or result.stdout or "Probe failed").strip()[-2000:])
                self._probe_queue.put(("probe", json.loads(payload)))
            except Exception as exc:
                self._probe_queue.put(("error", str(exc)))
        threading.Thread(target=worker, daemon=True, name="ExportCompatibilityProbe").start()

    def _apply_probe(self, report):
        self.compat = report
        for fmt in FORMATS:
            entry = report.get("formats", {}).get(fmt, {})
            enabled = bool(entry.get("available"))
            self.radios[fmt].configure(state="normal" if enabled else "disabled")
            if enabled:
                status_text = "✓ Ready" + (" · WSL2" if entry.get("backend") == "wsl" else "") + (
                    " · TFLite legacy" if entry.get("argument") == "tflite" else "")
            elif fmt == "engine":
                status_text = str(entry.get('reason', 'TensorRT missing'))[:85]
            elif fmt == "litert":
                status_text = ("WSL: " + str((report.get("wsl") or {}).get("phase", "unavailable"))
                               if report.get("platform") == "win32" else "Exporter unavailable")
            else:
                status_text = "Exporter unavailable"
            self.format_status[fmt].set(status_text)
        # Never silently switch the operator's selected format (for example,
        # TensorRT Engine) to the first available format (often ONNX). Retain
        # the selection, show the incompatibility reason, and disable Export.
        self._format_changed()
        trt_missing = (bool(report.get("cuda")) and bool(report.get("errors", {}).get("tensorrt"))
                       and not report["formats"]["engine"].get("available"))
        self.install_btn.configure(state="normal" if trt_missing and self._can_install_tensorrt() else "disabled")
        self.wsl_setup_btn.configure(state="normal" if report.get("platform") == "win32" else "disabled")
        self._update_target_status()
        if trt_missing and self._can_install_tensorrt() and self._tab_opened and not self._prompted_for_trt:
            self._prompted_for_trt = True
            self.after(150, self._offer_tensor_rt)
        if report.get('tensorrt'):
            self._append('Python TensorRT: ' + str(report['tensorrt']) +
                         ' · module: ' + str(report.get('tensorrt_module', 'unknown')))
        if report.get('host_tensorrt'):
            self._append('Host/JetPack TensorRT: ' + str(report['host_tensorrt']))
            if report.get('tensorrt_version_match') is False:
                self._append('WARNING: Python and host TensorRT versions differ; select the deployment environment.')
        if report.get('jetson'):
            self._append('Jetson detected: pip TensorRT installation disabled; using installed JetPack bindings.')
        if report.get("errors", {}).get("ultralytics"):
            self._append("Environment: " + report["errors"]["ultralytics"])
        for fmt in FORMATS:
            self._emit(f"{fmt.upper()} export: {report['formats'][fmt].get('reason')}")
        # Do not interpret an unsuccessful environment probe as permission to
        # offer installation. The operator can inspect the reason and choose
        # WSL status / setup explicitly, including when another distro works.
        if report.get("platform") == "win32":
            wsl = report.get("wsl") or {}
            self._append("WSL LiteRT: " + str(wsl.get("reason", "No diagnostic available")))
            for attempt in wsl.get("attempts", []):
                distro = str(attempt.get("distribution"))
                reason = str(attempt.get("reason"))
                # Probe diagnostics may already begin with the distribution.
                self._append("  " + (reason if reason.startswith(distro + ":")
                                     else distro + ": " + reason))

    def on_show(self):
        """Offer optional installation when the export section is first opened."""
        self._tab_opened = True
        if (self.compat and self.compat.get("cuda") and
                self.compat.get("errors", {}).get("tensorrt") and
                self._can_install_tensorrt() and not self._prompted_for_trt):
            self._prompted_for_trt = True
            self.after(120, self._offer_tensor_rt)
        # WSL setup is intentionally never prompted just for opening this tab.
        # See WSL status / setup for a diagnostic and explicit opt-in.

    def _offer_tensor_rt(self):
        if not self._can_install_tensorrt() or self.install_job or not self.winfo_exists():
            return
        package = self.compat.get("install_package", "tensorrt")
        command = f'"{self.env_python()}" -m pip install --upgrade {package}'
        if messagebox.askyesno(
            "TensorRT is not installed",
            "TensorRT full builder support is missing from your selected Conda environment.\n\n"
            "Install it now? This downloads NVIDIA packages and changes the selected environment.\n\n"
            f"Command:\n{command}\n\n"
            "Choosing No leaves TensorRT export disabled. ONNX and LiteRT remain available where supported.",
            parent=self):
            self.install_tensorrt()
        else:
            self._append("TensorRT install declined; Engine export remains disabled.")

    def install_tensorrt(self):
        if not self._can_install_tensorrt() or self.install_job or self.busy:
            return
        package = self.compat.get("install_package", "tensorrt")
        command = [str(self.env_python()), "-m", "pip", "install", "--upgrade", package]
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        logfile = self.logs_dir / ("export_tensorrt_install_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".log")
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            with logfile.open("w", encoding="utf-8") as stream:
                proc = subprocess.Popen(command, env=dict(self.process_env()),
                                        stdout=stream, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL, creationflags=flags)
        except OSError as exc:
            messagebox.showerror("TensorRT installation failed", str(exc), parent=self)
            return
        self.install_job = {"process": proc, "log": logfile, "kind": "tensorrt"}
        self.install_btn.configure(state="disabled")
        self.status.set("Installing TensorRT into selected Conda environment…")
        self.progress.pack(side="right", fill="x", expand=True, padx=(14, 0))
        self.progress.start(16)
        self._append("Installing " + package + ". Installation log: " + str(logfile))
        self._emit("TensorRT installation started: " + str(logfile))

    def _offer_wsl_setup(self):
        if not self.winfo_exists() or self.install_job:
            return
        if messagebox.askyesno(
                "LiteRT on Windows (WSL2)",
                "LiteRT export is available on Windows using WSL2, with a separate "
                "Linux Python 3.11 or 3.12 environment.\n\n"
                "Would you like to open the guided setup? Nothing is installed without your permission.",
                parent=self):
            self.setup_wsl()

    def setup_wsl(self):
        """Guide the operator; never elevate or install WSL without explicit action."""
        if not self.compat or self.compat.get("platform") != "win32":
            return
        wsl = self.compat.get("wsl") or {}
        phase = wsl.get("phase", "missing_wsl")
        if phase == "ready":
            messagebox.showinfo("WSL2 LiteRT ready", wsl.get("reason", "WSL is ready."), parent=self)
            return
        if phase in ("env_error", "unsupported_format", "distro_not_ready"):
            messagebox.showwarning(
                "WSL exporter diagnostic",
                "Studio could not confirm that the existing WSL exporter is ready.\n\n"
                + str(wsl.get("reason", "Unknown probe failure"))
                + "\n\nNo installation has been started. Check your working "
                  "WSL export environment before attempting repairs.", parent=self)
            return
        if phase in ("missing_env", "missing_packages") and wsl.get("distribution"):
            distro = wsl["distribution"]
            if messagebox.askyesno(
                    "Install LiteRT dependencies in WSL?",
                    f"Check result for {distro}:\n{wsl.get('reason', 'Missing dependencies')}\n\n"
                    "Run setup_yolo_export.sh in that distribution?\n\n"
                    "This script preserves a working Python 3.11/3.12 environment, "
                    "or creates Python 3.11 if missing. It installs uv if needed, "
                    "plus PyTorch, Ultralytics and LiteRT dependencies (no TensorFlow). "
                    "It may download large packages. "
                    "It does not modify Windows Conda. If Linux sudo is needed, "
                    "run setup_yolo_export.sh interactively from WSL instead.\n\n"
                    "Continue?", parent=self):
                self.install_wsl_environment(distro)
            return
        self._show_wsl_install_guide(wsl)

    def _show_wsl_install_guide(self, report):
        dialog = tk.Toplevel(self)
        dialog.title("Set up WSL2 for LiteRT export")
        dialog.transient(self.winfo_toplevel())
        dialog.resizable(False, False)
        frame = ttk.Frame(dialog, padding=22)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="LiteRT export through Windows Subsystem for Linux",
                  font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 12))
        if report.get("phase") == "wsl1":
            command = f'wsl --set-version {report.get("distribution", "Ubuntu")} 2'
            info = ("The selected Linux distribution uses WSL1.\n"
                    "Open PowerShell as Administrator and run:\n"
                    f"      {command}\n\n"
                    "Wait for conversion to finish; make sure hardware virtualization is enabled.\n"
                    "Then click Recheck, and use WSL / LiteRT setup to install the Python packages.")
        else:
            command = "wsl --install -d Ubuntu"
            info = ("1. Open PowerShell as Administrator and run:\n"
                    f"      {command}\n"
                    "2. Restart Windows if requested.\n"
                    "3. Open Ubuntu once, create your Linux username and password.\n"
                    "4. Return here and click Recheck environment.\n"
                    "5. Click WSL / LiteRT setup again to install the Linux Python packages.\n\n"
                    "The installation may require BIOS virtualization and internet access.\n"
                    "No Windows components are installed by this dialog.")
        ttk.Label(frame, text=info, justify="left", wraplength=580).pack(anchor="w", pady=(0, 12))
        ttk.Label(frame, text="Current status: " + report.get("reason", "WSL2 not available"),
                  style="Muted.TLabel", wraplength=580).pack(anchor="w", pady=(0, 14))
        actions = ttk.Frame(frame)
        actions.pack(fill="x")
        def copy_command():
            dialog.clipboard_clear()
            dialog.clipboard_append(command)
            dialog.update()
        ttk.Button(actions, text="Copy install command", command=copy_command).pack(side="left")
        ttk.Button(actions, text="Microsoft WSL instructions", command=lambda: webbrowser.open(
            "https://learn.microsoft.com/en-us/windows/wsl/install")).pack(side="left", padx=7)
        ttk.Button(actions, text="Recheck", command=lambda: (dialog.destroy(), self.check_compatibility())).pack(side="left")
        ttk.Button(actions, text="Close", command=dialog.destroy).pack(side="right")

    def install_wsl_environment(self, distro):
        if self.install_job or self.busy or self._probe_pending:
            return
        script = Path(__file__).with_name("setup_yolo_export.sh")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        logfile = self.logs_dir / ("export_wsl_setup_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".log")
        command = ["wsl.exe", "--distribution", distro, "--cd", str(script.parent),
                   "--exec", "/bin/bash", "./setup_yolo_export.sh"]
        try:
            with logfile.open("w", encoding="utf-8") as stream:
                proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL,
                                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except OSError as exc:
            messagebox.showerror("WSL setup failed", str(exc), parent=self)
            return
        self.install_job = {"process": proc, "log": logfile, "kind": "wsl"}
        self.wsl_setup_btn.configure(state="disabled")
        self.status.set("Installing LiteRT dependencies in WSL2 " + distro + "…")
        self.progress.pack(side="right", fill="x", expand=True, padx=(14, 0))
        self.progress.start(16)
        self._append("WSL package installation log: " + str(logfile))

    def _settings(self):
        return validate_export_settings({
            "model": self.model.get(), "output": self.output.get(),
            "model_name": self.model_name.get(),
            "format": self.fmt.get(), "imgsz": self.imgsz.get(),
            "batch": self.batch.get(), "device": self.device.get(),
            "precision": self.precision.get(), "dynamic": self.dynamic.get(),
            "simplify": self.simplify.get(), "nms": self.nms.get(),
            "opset": self.opset.get(), "workspace": self.workspace.get(),
            "data": self.data.get(), "fraction": self.fraction.get(),
            "wsl_distribution": (self.compat or {}).get("wsl", {}).get("distribution", ""),
            "export_python": self._export_python(),
        }, capabilities=self.compat)

    def start(self):
        if self.busy or self.install_job or self._probe_pending:
            return
        if self.can_start is not None and not self.can_start():
            messagebox.showwarning("Another operation is active",
                "Finish the current GPU/training/evaluation job before exporting.", parent=self)
            return
        try:
            values = self._settings()
        except (ValueError, OSError) as exc:
            messagebox.showerror("Invalid export settings", str(exc), parent=self)
            return
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        settings_file = self.logs_dir / f"model_export_{stamp}_settings.json"
        logfile = self.logs_dir / f"model_export_{stamp}.log"
        settings_file.write_text(json.dumps(values, indent=2), encoding="utf-8")
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        command = [str(self._export_python()), "-u", str(Path(__file__).with_name("run_model_export.py")), str(settings_file)]
        try:
            with logfile.open("w", encoding="utf-8") as stream:
                proc = subprocess.Popen(command, cwd=str(Path(__file__).parent),
                                        env=self._export_env(), stdin=subprocess.DEVNULL,
                                        stdout=stream, stderr=subprocess.STDOUT,
                                        creationflags=flags, start_new_session=(os.name != "nt"))
        except OSError as exc:
            messagebox.showerror("Could not start model export", str(exc), parent=self)
            return
        self.job = {"process": proc, "log": logfile, "offset": 0, "pending": "", "result": None,
                    "stopping": False, "format": values["format"],
                    "backend": values.get("backend", "native"),
                    "wsl_distribution": values.get("wsl_distribution"), "working_dir": None}
        self._last_result = None
        self.open_btn.configure(state="disabled")
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status.set("Exporting " + values["format"].upper() + " …")
        self.progress.pack(side="right", fill="x", expand=True, padx=(14, 0))
        self.progress.start(13)
        self._append('Selected export target: ' + FORMATS[values['format']][0] +
                     " · Ultralytics format=" + repr(values['ultralytics_format']))
        self._append('Output model name: ' +
                     (values['model_name'] + {'engine': '.engine', 'onnx': '.onnx', 'litert': '.tflite'}[values['format']]
                      if values['model_name'] else '(Ultralytics default)'))
        self._append('Export using Python: ' + self._export_python())
        self._append('Ultralytics auto-installation disabled for this job.')
        self._append("Starting native Ultralytics export. Full log: " + str(logfile))
        self._emit("Model export started: " + str(logfile))
        self._expanded_setup = False
        self.setup.pack_forget()
        self.monitor.pack_configure(fill="both", expand=True)
        self.log_text.configure(height=22)
        self.toggle_btn.pack(side="right", padx=8)
        self.toggle_btn.configure(text="Show settings")

    def stop(self):
        if not self.busy:
            return
        if not messagebox.askyesno("Stop export?", "Stop the current export process? The source model remains untouched.", parent=self):
            return
        job = self.job
        job["stopping"] = True
        proc = job["process"]
        try:
            if os.name == "nt" and job.get("backend") == "wsl" and job.get("working_dir"):
                # Cancel only this WSL export process, never all WSL sessions.
                pidfile = Path(job["working_dir"]) / "wsl_export.pid"
                if pidfile.is_file():
                    pid = pidfile.read_text(encoding="utf-8").strip()
                    if pid.isdecimal():
                        subprocess.run(["wsl.exe", "--distribution", job["wsl_distribution"],
                                        "--exec", "/bin/kill", "-TERM", pid],
                                       check=False, capture_output=True, timeout=7,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                               check=False, capture_output=True, timeout=15,
                               creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                os.killpg(proc.pid, signal.SIGTERM)
        except (OSError, ProcessLookupError, subprocess.TimeoutExpired) as exc:
            self._append("Stop request: " + str(exc))
        self.stop_btn.configure(state="disabled")
        self.status.set("Stopping export…")

    def _process_line(self, line):
        if line.startswith("STUDIO_EXPORT_WORKDIR="):
            if self.job:
                self.job["working_dir"] = line.split("=", 1)[1]
        elif line.startswith("STUDIO_EXPORT_STATUS="):
            self.status.set(line.split("=", 1)[1])
        elif line.startswith("STUDIO_EXPORT_RESULT="):
            try:
                self.job["result"] = json.loads(line.split("=", 1)[1])
            except ValueError:
                self._append("WARNING: invalid export summary event")
        elif line.strip():
            self._append(line[:2000])

    def poll(self):
        try:
            while True:
                kind, payload = self._probe_queue.get_nowait()
                self._probe_pending = False
                if kind == "probe":
                    self._apply_probe(payload)
                else:
                    self.status.set("Compatibility probe failed: " + payload[:150])
                    self._append("ERROR: " + payload)
                    self._emit("Model export compatibility probe failed: " + payload)
        except queue.Empty:
            pass
        if self.install_job and self.install_job["process"].poll() is not None:
            finished = self.install_job
            self.install_job = None
            self.progress.stop()
            self.progress.pack_forget()
            code = finished["process"].returncode
            kind = finished.get("kind", "tensorrt")
            label = "WSL LiteRT setup" if kind == "wsl" else "TensorRT installation"
            self._append(label + (" completed" if code == 0 else f" failed (exit {code})") +
                         ". Details: " + str(finished["log"]))
            if code != 0:
                messagebox.showerror(label + " failed",
                                     f"Setup exited with code {code}. See log:\n{finished['log']}\n\n"
                                     "If Linux sudo/input is required, run setup_yolo_export.sh "
                                     "from an interactive WSL terminal.", parent=self)
            self.check_compatibility()
        job = self.job
        if not job:
            return
        try:
            with job["log"].open(encoding="utf-8", errors="replace") as f:
                f.seek(job["offset"])
                chunk = f.read(60000)
                job["offset"] = f.tell()
            if chunk:
                lines = (job["pending"] + chunk).replace("\r", "\n").split("\n")
                job["pending"] = lines.pop()
                for line in lines:
                    self._process_line(line)
            code = job["process"].poll()
            if code is not None and job["log"].stat().st_size <= job["offset"]:
                if job["pending"].strip():
                    self._process_line(job["pending"])
                self.progress.stop()
                self.progress.pack_forget()
                self.stop_btn.configure(state="disabled")
                if not job["stopping"] and code == 0 and job["result"]:
                    self._last_result = job["result"]
                    self.open_btn.configure(state="normal")
                    self.status.set("Export completed · " + job["format"].upper())
                    if job['format'] == 'engine':
                        check = self._last_result.get('tensorrt_verification') or {}
                        if check.get('verified') is True:
                            self.status.set('TensorRT engine verified · synthetic inference passed')
                            self._append('Deployment check PASSED with TensorRT '
                                         + str(check.get('tensorrt', 'unknown')) + ' in '
                                         + str(check.get('python', 'selected Python')))
                    self._append("Saved exported model: " + self._last_result["exported_model"])
                    self._emit("Model export completed: " + self._last_result["exported_model"])
                else:
                    self.status.set("Export stopped" if job["stopping"] else f"Export failed (exit {code})")
                    self._emit(self.status.get() + " · " + str(job["log"]))
                self.job = None
                self.setup.pack(fill="both", expand=True, before=self.monitor)
                self.monitor.pack_configure(fill="x", expand=False)
                self.log_text.configure(height=3)
                self.toggle_btn.pack_forget()
                if self.compat and self.compat.get("formats", {}).get(self.fmt.get(), {}).get("available"):
                    self.start_btn.configure(state="normal")
        except OSError as exc:
            self._append("ERROR reading export log: " + str(exc))

    def open_results(self):
        if not self._last_result:
            return
        path = Path(self._last_result.get("output_directory", ""))
        if not path.is_dir():
            messagebox.showerror("Missing export folder", str(path), parent=self)
            return
        try:
            if os.name == "nt":
                os.startfile(str(path))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
        except OSError as exc:
            messagebox.showerror("Could not open export folder", str(exc), parent=self)
