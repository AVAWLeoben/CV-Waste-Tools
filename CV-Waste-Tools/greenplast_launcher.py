# -*- coding: utf-8 -*-
"""Cross-platform GreenPlast launcher for dataset splitting and annotation audit."""
from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import queue
import subprocess
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from greenplast_environment import (APP_DIR, EnvironmentManager, probe_env,
                                    cached_environment, save_cached_environment, CONFIG_FILE,
                                    env_python, process_environment,
                                    MANAGED_NAME, MANAGED_DISPLAY_NAME)

class Launcher:
    def __init__(self, prepared=None):
        self.root = tk.Tk()
        self.root.title("GreenPlast · Model Tools")
        self.root.geometry("790x545")
        self.root.minsize(680, 490)
        self.msgs = queue.Queue()
        self.conda = None
        self.env = None
        self.busy = False
        self.child_processes = {}
        self.build_ui()
        self.root.after(100, self.process_messages)
        if prepared:
            conda, prefix, details = prepared
            self.process_ready(conda, prefix, details)
        else:
            self.refresh(force=False)

    def process_ready(self, conda, prefix, details):
        """Use a checked environment prepared by the Linux shell bootstrap."""
        self.conda = conda
        self.env = prefix
        self.busy = False
        self.retry.configure(state="normal")
        self.train_button.configure(state="normal")
        self.audit_button.configure(state="normal")
        self.progress.configure(mode="determinate", maximum=100, value=100)
        self.status.set("Ready — choose a tool")
        self.env_text.set(
            f"{MANAGED_DISPLAY_NAME if self.env.name.lower() == MANAGED_NAME else self.env.name} "
            f"({self.env})  |  PyTorch {details.get('torch', '?')}  |  "
            f"CUDA: {'available' if details.get('cuda') else 'not available'}"
            + (f"  |  {details['gpu']}" if details.get('gpu') else "")
        )
        self.append_log("Using verified Conda environment: " + str(self.env))
        self.append_log("Persistent configuration: " + str(CONFIG_FILE))

    def build_ui(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        outer = ttk.Frame(self.root, padding=22)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(6, weight=1)
        ttk.Label(outer, text="GreenPlast  /  Model Tools",
                  font=("Segoe UI", 20, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(outer, text="Train and split datasets, or inspect YOLO annotations.",
                  foreground="#555555").grid(row=1, column=0, sticky="w", pady=(4, 18))

        buttons = ttk.Frame(outer)
        buttons.grid(row=2, column=0, sticky="ew")
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)
        self.train_button = ttk.Button(buttons, text="Open  ·  Split & Train",
                                       command=lambda: self.launch("train"), state="disabled")
        self.train_button.grid(row=0, column=0, sticky="ew", padx=(0, 8), ipady=16)
        self.audit_button = ttk.Button(buttons, text="Open  ·  Annotation Audit",
                                       command=lambda: self.launch("audit"), state="disabled")
        self.audit_button.grid(row=0, column=1, sticky="ew", padx=(8, 0), ipady=16)
        ttk.Label(buttons, text="Combine folders, make train/val/test lists, optionally train YOLO.",
                  wraplength=320, foreground="#555555").grid(row=1, column=0,
                                                           sticky="w", padx=(2, 14), pady=(8, 0))
        ttk.Label(buttons, text="Find class disagreements and suspicious boxes; export review CSV.",
                  wraplength=320, foreground="#555555").grid(row=1, column=1,
                                                           sticky="w", padx=(16, 0), pady=(8, 0))
        ttk.Separator(outer, orient="horizontal").grid(row=3, column=0, sticky="ew", pady=20)
        info = ttk.Frame(outer)
        info.grid(row=4, column=0, sticky="ew")
        info.columnconfigure(0, weight=1)
        self.status = tk.StringVar(value="Looking for Conda environments ...")
        self.env_text = tk.StringVar(value="Environment: checking")
        ttk.Label(info, textvariable=self.status, font=("Segoe UI", 10, "bold")).grid(
            row=0, column=0, sticky="w")
        ttk.Label(info, textvariable=self.env_text, foreground="#555555",
                  wraplength=660).grid(row=1, column=0, sticky="w", pady=(6, 8))
        self.progress = ttk.Progressbar(info, mode="indeterminate")
        self.progress.grid(row=2, column=0, sticky="ew")
        ttk.Label(outer, text="Setup and activity log", font=("Segoe UI", 10, "bold")).grid(
            row=5, column=0, sticky="w", pady=(16, 5))
        logframe = ttk.Frame(outer)
        logframe.grid(row=6, column=0, sticky="nsew")
        logframe.rowconfigure(0, weight=1)
        logframe.columnconfigure(0, weight=1)
        self.log_widget = tk.Text(logframe, height=6, wrap="word", state="disabled",
                                  font=("Consolas", 9), background="#f5f6f8")
        self.log_widget.grid(row=0, column=0, sticky="nsew")
        scr = ttk.Scrollbar(logframe, command=self.log_widget.yview)
        scr.grid(row=0, column=1, sticky="ns")
        self.log_widget.configure(yscrollcommand=scr.set)
        footer = ttk.Frame(outer)
        footer.grid(row=7, column=0, sticky="ew", pady=(12, 0))
        self.retry = ttk.Button(footer, text="Recheck / repair environment",
                                command=self.refresh)
        self.retry.pack(side="left")
        ttk.Button(footer, text="Open logs folder", command=self.open_logs_folder).pack(side="left", padx=(10, 0))
        ttk.Button(footer, text="Close", command=self.root.destroy).pack(side="right")

    def report(self, kind, value):
        self.msgs.put((kind, value))

    def refresh(self, force=True):
        if self.busy:
            return
        self.busy = True
        self.train_button.configure(state="disabled")
        self.audit_button.configure(state="disabled")
        self.retry.configure(state="disabled")
        self.status.set("Loading saved environment ..." if not force else "Rechecking Conda environment ...")
        self.env_text.set("Environment: checking")
        self.progress.configure(mode="indeterminate", value=0)
        self.progress.start(16)

        def worker():
            try:
                settings = EnvironmentManager(self.report).setup(force_recheck=force)
                self.report("ready", settings)
            except Exception as exc:
                self.report("error", str(exc))

        threading.Thread(target=worker, daemon=True, name="CondaEnvironmentSetup").start()

    def append_log(self, line):
        self.log_widget.configure(state="normal")
        self.log_widget.insert("end", line + "\n")
        # Don't retain thousands of pip-installation lines in the widget.
        if int(self.log_widget.index("end-1c").split(".")[0]) > 250:
            self.log_widget.delete("1.0", "75.0")
        self.log_widget.see("end")
        self.log_widget.configure(state="disabled")

    def process_messages(self):
        try:
            while True:
                kind, value = self.msgs.get_nowait()
                if kind == "log":
                    self.append_log("• " + str(value))
                elif kind == "detail":
                    # Display recent pip/conda command output in a scrolling log.
                    self.append_log(str(value))
                elif kind == "status":
                    self.status.set(str(value))
                elif kind == "ready":
                    self.conda, self.env, details = value
                    self.busy = False
                    self.progress.stop()
                    self.progress.configure(mode="determinate", maximum=100, value=100)
                    self.retry.configure(state="normal")
                    self.train_button.configure(state="normal")
                    self.audit_button.configure(state="normal")
                    self.status.set("Ready — choose a tool")
                    self.env_text.set(f"{MANAGED_DISPLAY_NAME if self.env.name.lower() == MANAGED_NAME else self.env.name} "
            f"({self.env})  |  PyTorch {details.get('torch', '?')}  |  "
                                      f"CUDA: {'available' if details.get('cuda') else 'not available'}"
                                      + (f"  |  {details['gpu']}" if details.get('gpu') else ""))
                elif kind == "error":
                    self.busy = False
                    self.progress.stop()
                    self.progress.configure(mode="determinate", maximum=100, value=0)
                    self.retry.configure(state="normal")
                    self.status.set("Environment setup failed")
                    self.env_text.set("Check the log and select Recheck / repair after resolving the problem.")
                    self.append_log("ERROR: " + str(value))
                    messagebox.showerror("GreenPlast setup failed", str(value), parent=self.root)
        except queue.Empty:
            pass
        self.poll_child_logs()
        if self.root.winfo_exists():
            self.root.after(250, self.process_messages)

    def open_logs_folder(self):
        folder = APP_DIR / "logs"
        folder.mkdir(parents=True, exist_ok=True)
        try:
            if os.name == "nt":
                os.startfile(str(folder))
            else:
                subprocess.Popen(["xdg-open", str(folder)],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Cannot open logs", f"{folder}\n\n{exc}", parent=self.root)

    def poll_child_logs(self):
        """Stream child output into the existing GUI, without creating consoles."""
        for pid, state in list(self.child_processes.items()):
            process = state["process"]
            try:
                with state["path"].open("r", encoding="utf-8", errors="replace") as stream:
                    stream.seek(state["offset"])
                    chunk = stream.read(16000)
                    state["offset"] = stream.tell()
                if chunk:
                    combined = (state["pending"] + chunk).replace("\r", "\n")
                    lines = combined.split("\n")
                    state["pending"] = lines.pop()
                    # Huge training logs remain complete on disk. Keep GUI responsive.
                    for line in lines[-35:]:
                        if line.strip():
                            self.append_log(f"[{state['name']}] {line[:600]}")
                exit_code = process.poll()
                # The process can exit while its last buffered log bytes are still
                # available; finish only when the stream has been consumed.
                remaining = state["path"].stat().st_size > state["offset"]
                if exit_code is not None and not remaining:
                    if state["pending"].strip():
                        self.append_log(f"[{state['name']}] {state['pending'][:600]}")
                    self.append_log(
                        f"{state['name'].title()} finished (exit {exit_code}). "
                        f"Full log: {state['path']}")
                    if exit_code != 0:
                        messagebox.showerror(
                            f"{state['name'].title()} failed",
                            f"The tool exited with code {exit_code}.\n\n"
                            f"See the launcher log or: {state['path']}", parent=self.root)
                    del self.child_processes[pid]
            except OSError as exc:
                self.append_log(f"Could not read {state['name']} log: {exc}")

    def launch(self, name):
        if not self.conda or not self.env:
            messagebox.showerror("Environment not ready", "Please wait for environment setup.")
            return
        logs = APP_DIR / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        logfile = logs / f"{name}_{stamp}.log"
        command = [str(env_python(self.env)), "-u", str(APP_DIR / "run_tool.py"), name]
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            with logfile.open("w", encoding="utf-8") as output:
                proc = subprocess.Popen(
                    command, cwd=str(APP_DIR), env=process_environment(self.env),
                    stdin=subprocess.DEVNULL, stdout=output,
                    stderr=subprocess.STDOUT, creationflags=flags)
        except OSError as exc:
            messagebox.showerror("Launch failed", str(exc), parent=self.root)
            return
        self.child_processes[proc.pid] = {
            "name": name, "path": logfile, "process": proc,
            "offset": 0, "pending": "",
        }
        self.append_log(f"Started {name} (PID {proc.pid}); no separate console. Log: {logfile}")

    def run(self):
        self.root.mainloop()


def main(argv=None):
    parser = argparse.ArgumentParser(description="GreenPlast tool launcher")
    parser.add_argument("--env-prefix", default=None, help="Previously verified Conda env")
    parser.add_argument("--conda-exe", default=None, help="Conda executable path")
    args = parser.parse_args(argv)
    prepared = None
    if args.env_prefix or args.conda_exe:
        if not args.env_prefix or not args.conda_exe:
            parser.error("--env-prefix and --conda-exe must both be specified")
        env = Path(args.env_prefix).resolve()
        cached = cached_environment(expected_conda=args.conda_exe,
                                    expected_prefix=env)
        if cached:
            prepared = cached
        else:
            # Prepared environment from Linux bootstrap is normally cached;
            # verify if cache is absent or no longer matches its files/GPU.
            details = probe_env(args.conda_exe, env, needs_cuda=False, timeout=480)
            if not details.get("ok"):
                parser.error("Prepared environment is not usable: " + str(details.get("error", details)))
            prepared = (args.conda_exe, env, details)
            try:
                save_cached_environment(*prepared)
            except OSError:
                pass
    Launcher(prepared=prepared).run()


if __name__ == "__main__":
    main()
