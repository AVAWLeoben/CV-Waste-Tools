# -*- coding: utf-8 -*-
"""CV Waste Studio: a single-window interface for the existing YOLO tooling.

Conda preparation works from the base interpreter, but all tool panels run
inside the verified environment. The heavy training work is isolated in one
headless child process and does not create another GUI.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from studio_branding import apply_window_icon, set_windows_app_id, sidebar_icon
from training_charts import LiveMetricsGraph
from training_log import BestEpochTracker
from training_results import TrainingResultsPanel
from studio_single_instance import (another_instance_running, claim_gui_instance,
                                    release_gui_instance, activate_existing_window)

from greenplast_environment import (
    APP_DIR, EnvironmentManager, probe_env, cached_environment,
    save_cached_environment, CONFIG_FILE, env_python,
    process_environment, MANAGED_NAME, MANAGED_DISPLAY_NAME,
)

BG = "#F3F6FB"
SURFACE = "#FFFFFF"
INK = "#19263B"
MUTED = "#60718A"
ACCENT = "#405DE6"
ACCENT_HOVER = "#304AC9"
NAV_BG = "#15243A"
NAV_MUTED = "#9DACBD"
BORDER = "#DAE2EF"
SUCCESS = "#1C8058"
DANGER = "#B53A43"
PAGE_TEXT = {
    "train": ("Datasets & training", "Create splits or train directly from an existing YOLO dataset."),
    "audit": ("Annotation audit", "Identify inconsistent labels and export review-ready CSV findings."),
    "recall": ("Precision / recall", "Evaluate recall at your required precision using reusable inference results."),
    "system": ("Environment & setup", "Verify the Conda environment, inspect GPU support and access logs."),
    "results": ("Training results", "Explore completed Ultralytics runs, training curves and validation plots."),
}


def _same_executable(a, b):
    try:
        return os.path.samefile(a, b)
    except (OSError, ValueError):
        return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


class Studio:
    def __init__(self, prepared=None, allow_multiple=False):
        self.allow_multiple = allow_multiple
        set_windows_app_id()
        self.root = tk.Tk()
        apply_window_icon(self.root)
        self.root.title("CV Waste Studio")
        self.root.geometry("1290x845")
        self.root.minsize(1010, 690)
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.messages = queue.Queue()
        self.env = None
        self.conda = None
        self.env_busy = False
        self.details = {}
        self.active = "train"
        self.ready = False
        self.pages = {}
        self.nav = {}
        self.logs = {key: deque(maxlen=1000) for key in PAGE_TEXT}
        self.session_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.activity_paths = {key: APP_DIR / "logs" / f"studio_{key}_{self.session_stamp}.log"
                               for key in ("system", "audit", "recall")}
        self.train_job = None
        self.best_epoch_tracker = BestEpochTracker()
        self.train_button = None
        self.settings_expanded = False
        self.audit_gui = None
        self.recall_gui = None
        self._style()
        self._layout()
        self.root.after(160, self._tick)
        if prepared:
            self.process_ready(*prepared)
        else:
            self.refresh(force=False)

    def _style(self):
        s = ttk.Style(self.root)
        try:
            s.theme_use("clam")
        except tk.TclError:
            pass
        s.configure(".", font=("Segoe UI", 10), foreground=INK, background=SURFACE)
        s.configure("TFrame", background=SURFACE)
        s.configure("TLabel", background=SURFACE, foreground=INK)
        s.configure("Muted.TLabel", foreground=MUTED)
        s.configure("TButton", background="#EAF0F8", foreground=INK,
                    padding=(14, 9), relief="flat", borderwidth=0)
        s.map("TButton", background=[("active", "#DEE8F7"), ("disabled", "#F1F3F7")],
              foreground=[("disabled", "#94A0AF")])
        s.configure("Accent.TButton", background=ACCENT, foreground="white",
                    padding=(16, 10), borderwidth=0)
        s.map("Accent.TButton", background=[("active", ACCENT_HOVER), ("disabled", "#B6C3F5")],
              foreground=[("disabled", "#F9FAFF")])
        s.configure("Danger.TButton", background="#FCECEE", foreground=DANGER,
                    padding=(14, 9), borderwidth=0)
        s.map("Danger.TButton", background=[("active", "#F9DEE1")])
        s.configure("TEntry", padding=(9, 7), fieldbackground="#FAFCFF",
                    bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER)
        s.configure("TCombobox", padding=(7, 6), fieldbackground="#FAFCFF",
                    bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER)
        s.map("TCombobox", fieldbackground=[("readonly", "#FAFCFF")],
              selectbackground=[("readonly", "#FAFCFF")],
              selectforeground=[("readonly", INK)])
        s.configure("TCheckbutton", background=SURFACE, foreground=INK)
        s.configure("TRadiobutton", background=SURFACE, foreground=INK)
        s.configure("TNotebook", background=SURFACE, borderwidth=0)
        s.configure("TNotebook.Tab", padding=(16, 11), font=("Segoe UI", 10, "bold"),
                    background="#ECF1F8", foreground=MUTED, borderwidth=0)
        s.map("TNotebook.Tab", background=[("selected", SURFACE), ("active", "#E5ECFA")],
              foreground=[("selected", ACCENT)])
        s.configure("TProgressbar", background=ACCENT, troughcolor="#E4EAF4", borderwidth=0)
        s.configure("TScrollbar", background="#CCD6E6", troughcolor=SURFACE, borderwidth=0)
        s.configure("TSeparator", background=BORDER)

    def _layout(self):
        self.outer = tk.Frame(self.root, bg=BG)
        self.outer.pack(fill="both", expand=True)
        sidebar = tk.Frame(self.outer, bg=NAV_BG, width=220)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        brand = tk.Frame(sidebar, bg=NAV_BG)
        brand.pack(fill="x", padx=16, pady=(26, 26))
        self.brand_icon = sidebar_icon(self.root)
        if self.brand_icon is not None:
            tk.Label(brand, image=self.brand_icon, bg=NAV_BG, bd=0).pack(
                side="left", padx=(0, 9))
        brand_text = tk.Frame(brand, bg=NAV_BG)
        brand_text.pack(side="left", fill="x")
        tk.Label(brand_text, text="CV / WASTE", bg=NAV_BG, fg="#8DA7FF",
                 font=("Segoe UI", 9, "bold"), anchor="w").pack(anchor="w")
        tk.Label(brand_text, text="STUDIO", bg=NAV_BG, fg="white",
                 font=("Segoe UI", 19, "bold"), anchor="w").pack(anchor="w")
        tk.Label(brand_text, text="Model operations", bg=NAV_BG, fg=NAV_MUTED,
                 font=("Segoe UI", 9), anchor="w").pack(anchor="w", pady=(2, 0))
        tk.Frame(sidebar, height=1, bg="#293A52").pack(fill="x", padx=19, pady=(0, 18))
        nav_items = (
            ("train", "01", "Datasets & training"),
            ("audit", "02", "Annotation audit"),
            ("recall", "03", "Precision / recall"),
            ("results", "04", "Training results"),
        )
        for key, number, label in nav_items:
            self._nav_button(sidebar, key, f"{number}   {label}")
        tk.Frame(sidebar, bg=NAV_BG).pack(fill="both", expand=True)
        tk.Frame(sidebar, height=1, bg="#293A52").pack(fill="x", padx=19, pady=(0, 14))
        self._nav_button(sidebar, "system", "⚙   Environment")
        self.nav_hint = tk.Label(sidebar, text="Checking environment…", bg=NAV_BG,
                                 fg=NAV_MUTED, wraplength=180, justify="left",
                                 font=("Segoe UI", 9), anchor="w")
        self.nav_hint.pack(fill="x", padx=24, pady=(16, 28))

        work = tk.Frame(self.outer, bg=BG)
        self.workspace = work
        work.pack(side="left", fill="both", expand=True)
        top = tk.Frame(work, bg=BG)
        top.pack(fill="x", padx=26, pady=(24, 15))
        self.title = tk.Label(top, font=("Segoe UI", 23, "bold"),
                              bg=BG, fg=INK, anchor="w")
        self.title.pack(anchor="w")
        self.subtitle = tk.Label(top, font=("Segoe UI", 10), bg=BG,
                                 fg=MUTED, anchor="w")
        self.subtitle.pack(anchor="w", pady=(4, 0))
        status_bar = tk.Frame(top, bg=BG)
        status_bar.place(relx=1.0, rely=0.05, anchor="ne")
        self.status_dot = tk.Label(status_bar, text="●", bg=BG, fg="#E69B36",
                                   font=("Segoe UI", 12))
        self.status_dot.pack(side="left", padx=(0, 5))
        self.status_label = tk.Label(status_bar, text="Checking", bg=BG, fg=MUTED,
                                     font=("Segoe UI", 9, "bold"))
        self.status_label.pack(side="left")
        card = tk.Frame(work, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1)
        self.page_card = card
        card.pack(fill="both", expand=True, padx=26, pady=(0, 12))
        self.content = tk.Frame(card, bg=SURFACE)
        self.content.pack(fill="both", expand=True)
        self.log_frame = tk.Frame(work, bg=SURFACE, highlightbackground=BORDER,
                                  highlightthickness=1)
        self.log_frame.pack(fill="x", padx=26, pady=(0, 18))
        toolbar = tk.Frame(self.log_frame, bg=SURFACE)
        toolbar.pack(fill="x", padx=16, pady=(9, 5))
        tk.Label(toolbar, text="ACTIVITY", bg=SURFACE, fg=INK,
                 font=("Segoe UI", 9, "bold")).pack(side="left")
        self.log_label = tk.Label(toolbar, text="· Split & train", bg=SURFACE,
                                  fg=MUTED, font=("Segoe UI", 9))
        self.log_label.pack(side="left", padx=(8, 0))
        # Only shown while a dataset job is running. The training form can be
        # restored temporarily without interrupting the job.
        self.monitor_tools = tk.Frame(self.log_frame, bg="#F4F7FD")
        self.monitor_label = tk.Label(self.monitor_tools, bg="#F4F7FD", fg=INK,
                                      text="", font=("Segoe UI", 10, "bold"),
                                      anchor="w")
        self.monitor_label.pack(side="left", fill="x", expand=True, padx=(12, 4))
        self.monitor_stop = ttk.Button(self.monitor_tools, text="Stop running job",
                                       style="Danger.TButton", command=self.stop_training)
        self.monitor_stop.pack(side="right", padx=(4, 10), pady=5)
        self.settings_toggle = ttk.Button(self.monitor_tools, text="Show settings",
                                          command=self.toggle_training_settings)
        self.settings_toggle.pack(side="right", pady=5)
        ttk.Button(toolbar, text="Clear view", command=self.clear_log).pack(side="right")
        self.graph_toggle = ttk.Button(toolbar, text="Show graphs", command=self.toggle_graph)
        self.graph_toggle.pack(side="right", padx=(0, 8))
        self.graph_visible = False
        self.log_toggle = ttk.Button(toolbar, text="Hide log", command=self.toggle_log)
        self.log_toggle.pack(side="right", padx=(0, 8))
        self.follow_log = tk.BooleanVar(value=True)
        ttk.Checkbutton(toolbar, text="Auto-scroll", variable=self.follow_log).pack(side="right", padx=8)
        ttk.Button(toolbar, text="Open logs", command=self.open_logs_folder).pack(side="right", padx=4)
        self.metric_frame = tk.Frame(self.log_frame, bg=SURFACE)
        self.metric_frame.pack(fill="x", padx=16, pady=(0,6))
        self.metric_summary = tk.Frame(self.metric_frame, bg=SURFACE)
        self.metric_summary.pack(fill="x")
        tk.Label(self.metric_summary, text="LATEST EPOCH · losses: lower is better · precision / recall / mAP: 0–1, higher is better",
                 bg=SURFACE, fg=MUTED, font=("Segoe UI",8), anchor="w", justify="left", wraplength=820).pack(fill="x",pady=(0,3))
        self.train_metrics = tk.StringVar(value="Training: waiting for epoch results")
        self.val_metrics = tk.StringVar(value="Validation: waiting for evaluation")
        tk.Label(self.metric_summary, textvariable=self.train_metrics, bg=SURFACE, fg="#2857A4",
                 anchor="w", justify="left", wraplength=820, font=("Segoe UI",9)).pack(fill="x")
        tk.Label(self.metric_summary, textvariable=self.val_metrics, bg=SURFACE, fg="#28704A",
                 anchor="w", justify="left", wraplength=820, font=("Segoe UI",9)).pack(fill="x")
        # The best checkpoint remains visible even when the latest-epoch
        # summary is hidden in graph-first monitoring mode.
        self.best_epoch_banner = tk.Frame(self.metric_frame, bg="#F0F5FF",
                                          highlightbackground="#B7CAF5",
                                          highlightthickness=1)
        self.best_epoch_banner.pack(fill="x", pady=(0, 5))
        self.best_heading = tk.StringVar()
        self.best_scores = tk.StringVar()
        self.best_losses = tk.StringVar()
        for variable, color, font in (
                (self.best_heading, "#204B9B", ("Segoe UI", 9, "bold")),
                (self.best_scores, INK, ("Segoe UI", 9)),
                (self.best_losses, MUTED, ("Segoe UI", 8))):
            tk.Label(self.best_epoch_banner, textvariable=variable, bg="#F0F5FF",
                     fg=color, font=font, anchor="w").pack(fill="x", padx=11, pady=1)
        self._refresh_best_epoch()
        self.progress_head = tk.Frame(self.metric_frame, bg=SURFACE)
        self.progress_head.pack(fill="x", pady=(7, 3))
        tk.Label(self.progress_head, text="ITERATION PROGRESS", bg=SURFACE, fg=MUTED,
                 font=("Segoe UI", 8, "bold")).pack(side="left")
        self.iteration_label = tk.StringVar(value="Waiting for training batches")
        tk.Label(self.progress_head, textvariable=self.iteration_label, bg=SURFACE,
                 fg=INK, font=("Segoe UI", 9)).pack(side="right")
        self.iteration_bar = ttk.Progressbar(self.metric_frame, mode="determinate", maximum=100)
        self.iteration_bar.pack(fill="x", pady=(0, 5))
        self.graph_frame = tk.Frame(self.metric_frame, bg=SURFACE)
        header = tk.Frame(self.graph_frame, bg=SURFACE)
        header.pack(fill="x", pady=(3, 4))
        tk.Label(header, text="LIVE TRAINING CURVES", bg=SURFACE, fg=MUTED,
                 font=("Segoe UI", 8, "bold")).pack(side="left")
        tk.Label(header, text="Click any chart to enlarge · Esc or × to close · updates each epoch",
                 bg=SURFACE, fg=MUTED, font=("Segoe UI", 8)).pack(side="right")
        self.training_graph = LiveMetricsGraph(self.graph_frame)
        self.training_graph.pack(fill="x")
        self.metric_frame.bind('<Configure>', lambda e: [w.configure(wraplength=max(100,e.width))
                               for w in self.metric_frame.winfo_children()
                               if isinstance(w, tk.Label)])
        text_frame = tk.Frame(self.log_frame, bg=SURFACE)
        text_frame.pack(fill="x", padx=15, pady=(0, 13))
        self.log_text_frame = text_frame
        self.log_text = tk.Text(text_frame, height=3, wrap="word", bd=0,
                                background="#F8FAFD", foreground="#31445D",
                                font=("Consolas", 9), state="disabled", padx=10, pady=9,
                                highlightthickness=0, selectbackground="#D9E4FB")
        self.log_text.pack(side="left", fill="both", expand=True)
        self.log_scroll = ttk.Scrollbar(text_frame, command=self.log_text.yview)
        self.log_scroll.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=self.log_scroll.set)
        # The active workflow controls how much room the form and log receive.
        # Training monitoring expands into the workspace, while initial setup
        # leaves nearly the entire height to the settings form.
        card.pack_forget()
        self.log_frame.pack_forget()
        self.log_text.tag_configure("error", foreground=DANGER)
        self.log_text.tag_configure("success", foreground=SUCCESS)
        self.log_text.tag_configure("warning", foreground="#916400")
        self.log_text.tag_configure("training", foreground="#2857A4")
        self.log_text.tag_configure("validation", foreground="#28704A")
        self.log_visible = True
        self.setup_page = tk.Frame(self.content, bg=SURFACE)
        self.pages["system"] = self.setup_page
        self._build_system()
        self._show("system")

    def _nav_button(self, parent, key, text):
        button = tk.Button(parent, text=text, anchor="w", bg=NAV_BG, fg="#CBD6E5",
                           activebackground="#273B5D", activeforeground="white",
                           relief="flat", bd=0, highlightthickness=0,
                           font=("Segoe UI", 10, "bold"), padx=20, pady=15,
                           cursor="hand2", command=lambda k=key: self.show(k))
        button.pack(fill="x", padx=12, pady=2)
        self.nav[key] = button

    def _build_system(self):
        p = self.setup_page
        ttk.Label(p, text="Managed Conda environment", font=("Segoe UI", 15, "bold")).pack(
            anchor="w", padx=24, pady=(26, 5))
        ttk.Label(p, text="All three tools use the same checked Python, PyTorch and CUDA stack.",
                  style="Muted.TLabel").pack(anchor="w", padx=24)
        self.env_info = tk.StringVar(value="Locating Conda and checking dependencies…")
        self.env_status = tk.StringVar(value="Checking…")
        ttk.Label(p, textvariable=self.env_status, font=("Segoe UI", 12, "bold")).pack(
            anchor="w", padx=24, pady=(25, 8))
        ttk.Label(p, textvariable=self.env_info, wraplength=750,
                  style="Muted.TLabel", justify="left").pack(anchor="w", padx=24)
        self.env_progress = ttk.Progressbar(p, mode="indeterminate")
        self.env_progress.pack(fill="x", padx=24, pady=(23, 20))
        actions = ttk.Frame(p)
        actions.pack(fill="x", padx=24)
        self.recheck_btn = ttk.Button(actions, text="Recheck / repair environment",
                                      style="Accent.TButton", command=self.refresh)
        self.recheck_btn.pack(side="left")
        ttk.Button(actions, text="Open log folder", command=self.open_logs_folder).pack(
            side="left", padx=10)
        ttk.Label(p, text="Environment activity and diagnostics appear in the shared log below.",
                  style="Muted.TLabel", wraplength=750).pack(anchor="w", padx=24, pady=24)

    def _show(self, key):
        for page in self.pages.values():
            page.pack_forget()
        if key not in self.pages:
            key = "system"
        self.pages[key].pack(fill="both", expand=True)
        self.active = key
        self._apply_workspace_layout()
        name, detail = PAGE_TEXT[key]
        self.title.configure(text="Training monitor" if self._monitoring() else name)
        self.subtitle.configure(text=("Live batch progress, epoch metrics and job output. "
                                      "Training settings are available on demand."
                                      if self._monitoring() else detail))
        for k, btn in self.nav.items():
            btn.configure(bg="#304C87" if k == key else NAV_BG,
                          fg="white" if k == key else "#CBD6E5")
        self.log_label.configure(text="· " + name)
        self._render_log()

    def _monitoring(self):
        """Only the datasets page becomes a full-screen live job monitor."""
        return self.active == "train" and self.train_job is not None

    def toggle_training_settings(self):
        if not self._monitoring():
            return
        self.settings_expanded = not self.settings_expanded
        self._apply_workspace_layout()

    def _apply_workspace_layout(self):
        """Reallocate space for setup, live monitoring and results.

        Repacking does not recreate widgets or modify form values, and is safe
        while the subprocess is still reporting batch progress.
        """
        monitoring = self._monitoring()
        self.page_card.pack_forget()
        self.log_frame.pack_forget()
        self.monitor_tools.pack_forget()
        self.metric_frame.pack_forget()
        self.graph_frame.pack_forget()
        self.log_text_frame.pack_forget()

        if self.active == "results":
            self.page_card.pack(fill="both", expand=True, padx=26, pady=(0, 18))
            self.page_card.pack_propagate(True)
            return

        if monitoring:
            if self.settings_expanded:
                # A genuinely useful settings view, not a few clipped rows:
                # give the form the available height and keep progress + stop
                # controls in a compact dock beneath it.
                self.log_frame.pack(side="bottom", fill="x", padx=26, pady=(0, 18))
                self.page_card.pack(side="top", fill="both", expand=True,
                                    padx=26, pady=(0, 8))
                self.page_card.pack_propagate(False)
            else:
                self.log_frame.pack(side="top", fill="both", expand=True,
                                    padx=26, pady=(0, 18))
            self.monitor_label.configure(text=self.train_state.get())
            self.settings_toggle.configure(text="Hide settings" if self.settings_expanded
                                           else "Show settings")
            self.monitor_tools.pack(side="top", fill="x", padx=15, pady=(0, 5))
            expand_metrics = self.graph_visible and not self.settings_expanded
            self.metric_frame.pack(side="top", fill="both" if expand_metrics else "x",
                                   expand=expand_metrics,
                                   padx=16, pady=(0, 6))
            self.training_graph.pack_configure(fill="both", expand=True)
            self.log_text.configure(height=2 if self.settings_expanded else 3)
        else:
            # Form-first layout: activity is a small, optional status dock.
            # Pack the dock first, at the bottom, to guarantee form visibility.
            self.log_frame.pack(side="bottom", fill="x", padx=26, pady=(0, 18))
            self.page_card.pack(side="top", fill="both", expand=True,
                                padx=26, pady=(0, 12))
            self.page_card.pack_propagate(False)
            self.log_text.configure(height=2 if self.active == "train" else 5)
            if self.active == "train" and self.graph_visible:
                self.metric_frame.pack(side="top", fill="x", padx=16, pady=(0, 6))
                self.training_graph.pack_configure(fill="x", expand=False)

        if self.active == "train" and self.graph_visible and not (
                monitoring and self.settings_expanded):
            self.metric_summary.pack_forget()
            self.graph_frame.pack(fill="both" if monitoring else "x",
                                  expand=monitoring, pady=(2, 5))
        elif self.active == "train" and monitoring:
            self.metric_summary.pack(fill="x", before=self.progress_head)

        if self.log_visible:
            expand_log = monitoring and not self.graph_visible and not self.settings_expanded
            self.log_text_frame.pack(side="bottom" if monitoring else "top",
                                     fill="both" if expand_log else "x",
                                     expand=expand_log, padx=15,
                                     pady=(0, 10 if monitoring else 13))

    def show(self, key):
        if key not in self.pages:
            self._show("system")
            if not self.ready:
                self.log_line("system", "Tools become available after environment setup.")
        else:
            self._show(key)

    def log_line(self, key, line):
        if key not in self.logs:
            key = "system"
        parts = str(line).splitlines() or [""]
        if key == "train":
            from training_log import readable_line
            parts = [readable_line(part) for part in parts]
            for part in parts:
                if part.startswith("Training |"):
                    self.train_metrics.set(part)
                elif part.startswith("Validation |"):
                    self.val_metrics.set(part)
        for part in parts:
            self.logs[key].append(part[:1400])
        if key in self.activity_paths:
            try:
                destination = self.activity_paths[key]
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("a", encoding="utf-8") as stream:
                    stream.write("".join(part + "\n" for part in parts))
            except OSError:
                pass  # The GUI must remain usable when logging is unavailable.
        if key == self.active:
            self._append_visible(parts)

    def _append_visible(self, lines):
        self.log_text.configure(state="normal")
        for line in lines:
            tag = "error" if ("ERROR" in line.upper() or "TRACEBACK" in line.upper() or "JOB FAILED" in line.upper()) else (
                "success" if ("COMPLETE" in line.upper() or "FINISHED" in line.upper()) else "")
            if not tag:
                if "WARNING" in line.upper(): tag="warning"
                elif line.startswith("Training |"): tag="training"
                elif line.startswith("Validation |"): tag="validation"
            self.log_text.insert("end", line[:1400] + "\n", tag)
        if int(self.log_text.index("end-1c").split(".")[0]) > 1100:
            self.log_text.delete("1.0", "130.0")
        if self.follow_log.get():
            self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _render_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
        self._append_visible(list(self.logs[self.active]))

    def clear_log(self):
        self.logs[self.active].clear()
        self._render_log()

    def set_graph_visibility(self, visible):
        self.graph_visible = visible
        if visible:
            self.metric_summary.pack_forget()
            self.log_visible = False
        else:
            self.log_visible = True
        self._apply_workspace_layout()
        self.log_toggle.configure(text="Hide log" if self.log_visible else "Show log")
        self.graph_toggle.configure(text="Hide graphs" if visible else "Show graphs")

    def toggle_graph(self):
        self.set_graph_visibility(not self.graph_visible)

    def toggle_log(self):
        self.log_visible = not self.log_visible
        self._apply_workspace_layout()
        self.log_toggle.configure(text="Hide log" if self.log_visible else "Show log")

    def report(self, kind, value):
        self.messages.put((kind, value))

    def refresh(self, force=True):
        if self.env_busy:
            return
        if (self.train_job and self.train_job["process"].poll() is None) or (
                self.audit_gui and self.audit_gui.busy) or (
                self.recall_gui and self.recall_gui.busy):
            messagebox.showwarning("Tasks running", "Finish ongoing tool operations before repairing the environment.", parent=self.root)
            return
        self.env_busy = True
        self.env_status.set("Verifying environment…")
        self.env_info.set("Checking Python, required packages and available GPU devices.")
        self.env_progress.configure(mode="indeterminate", value=0)
        self.env_progress.start(18)
        self.recheck_btn.configure(state="disabled")
        self.status_label.configure(text="Checking")
        self.status_dot.configure(fg="#E69B36")
        self.log_line("system", "Starting " + ("full repair/verification" if force else "cached environment check") + "…")

        def worker():
            try:
                self.report("ready", EnvironmentManager(self.report).setup(force_recheck=force))
            except Exception as exc:
                self.report("error", str(exc))
        threading.Thread(target=worker, daemon=True, name="EnvironmentSetup").start()

    def process_ready(self, conda, prefix, details):
        self.conda = conda
        self.env = Path(prefix)
        self.details = details
        self.env_busy = False
        self.env_progress.stop()
        self.env_progress.configure(mode="determinate", maximum=100, value=100)
        self.recheck_btn.configure(state="normal")
        self.env_status.set("Environment ready")
        self.env_info.set(
            f"{MANAGED_DISPLAY_NAME if self.env.name.lower() == MANAGED_NAME else self.env.name}\n"
            f"Python: {env_python(self.env)}\n"
            f"PyTorch: {details.get('torch', 'unknown')}    ·    "
            f"CUDA: {'available' if details.get('cuda') else 'not available'}"
            + (f"    ·    GPU: {details['gpu']}" if details.get("gpu") else "")
            + f"\nConfig: {CONFIG_FILE}"
        )
        self.log_line("system", f"Environment verified: {self.env}")
        self.status_dot.configure(fg=SUCCESS)
        self.status_label.configure(text="Ready")
        self.nav_hint.configure(text="Environment ready\n" + self.env.name)

        # The original Windows entry point uses Conda *base* to bootstrap.
        # Restart the UI inside the managed environment so the embedded audit
        # panel can use NumPy without depending on base's packages. Only the
        # managed UI is left running; there is no secondary tool window.
        python = env_python(self.env)
        if not _same_executable(sys.executable, str(python)):
            args = [str(python), "-u", str(APP_DIR / "greenplast_launcher.py"),
                    "--env-prefix", str(self.env), "--conda-exe", str(self.conda)]
            if self.allow_multiple:
                args.append("--new-instance")
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            try:
                subprocess.Popen(args, cwd=str(APP_DIR), env=process_environment(self.env),
                                 stdin=subprocess.DEVNULL, creationflags=flags)
            except OSError as exc:
                self.log_line("system", f"ERROR: Could not start the verified Python: {exc}")
                self.env_status.set("Could not open managed application")
                self._show("system")
                return
            self.root.destroy()
            return
        # Also protect direct launches from a managed Python without CLI flags.
        if not self.allow_multiple and not claim_gui_instance():
            activate_existing_window()
            self.root.destroy()
            return
        self._init_pages()

    def _scroll_host(self, parent):
        """Scrollable settings viewport: usable even on smaller displays."""
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)
        canvas = tk.Canvas(parent, background=SURFACE, bd=0, highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        vertical = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal = ttk.Scrollbar(parent, orient="horizontal", command=canvas.xview)
        horizontal.grid(row=1, column=0, sticky="ew")
        horizontal.grid_remove()
        canvas.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        host = ttk.Frame(canvas)
        window = canvas.create_window((0, 0), window=host, anchor="nw")

        def resize(_event=None):
            actual = max(1, canvas.winfo_width())
            natural = max(1, host.winfo_reqwidth())
            target = max(actual, natural)
            if int(canvas.itemcget(window, "width") or 0) != target:
                canvas.itemconfigure(window, width=target)
            if natural > actual + 3:
                horizontal.grid()
            else:
                horizontal.grid_remove()
            canvas.configure(scrollregion=canvas.bbox("all"))

        canvas.bind("<Configure>", resize)
        host.bind("<Configure>", resize)

        def wheel(event):
            if getattr(event, "num", None) == 4:
                steps = -1
            elif getattr(event, "num", None) == 5:
                steps = 1
            else:
                steps = -int(event.delta / 120) if event.delta else 0
            if steps:
                canvas.yview_scroll(steps, "units")

        def bind(_event):
            for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                self.root.bind_all(sequence, wheel)

        def unbind(_event):
            for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                self.root.unbind_all(sequence)

        host.bind("<Enter>", bind)
        host.bind("<Leave>", unbind)
        canvas.bind("<Enter>", bind)
        canvas.bind("<Leave>", unbind)
        return host

    def _init_pages(self):
        if self.ready:
            return
        try:
            from train_and_split_tool_GUI import _all_in_one_gui
            from audit_dataset_review_gui import AuditGUI
            from test_model_recall_at_precision import RecallTestGUI
            train = tk.Frame(self.content, bg=SURFACE)
            audit = tk.Frame(self.content, bg=SURFACE)
            recall = tk.Frame(self.content, bg=SURFACE)
            results = tk.Frame(self.content, bg=SURFACE)
            self.pages.update(train=train, audit=audit, recall=recall, results=results)
            train_controls = ttk.Frame(train, padding=(20, 12, 20, 0))
            train_controls.pack(fill="x")
            self.train_state = tk.StringVar(value="Ready · split datasets or train existing YOLO splits")
            ttk.Label(train_controls, textvariable=self.train_state,
                      style="Muted.TLabel").pack(side="left")
            self.stop_button = ttk.Button(train_controls, text="Stop running job",
                                          style="Danger.TButton", command=self.stop_training,
                                          state="disabled")
            self.stop_button.pack(side="right")
            _, self.train_button = _all_in_one_gui(
                train, self.start_training, on_preannotate=self.start_preannotation)
            self.train_button.configure(style="Accent.TButton")
            self.audit_gui = AuditGUI(self._scroll_host(audit), embedded=True,
                                      on_log=lambda message: self.log_line("audit", message))
            self.audit_gui.start_button.configure(style="Accent.TButton")
            self.recall_gui = RecallTestGUI(self._scroll_host(recall), embedded=True,
                                            on_log=lambda message: self.log_line("recall", message))
            self.recall_gui.run_button.configure(style="Accent.TButton")
            self.recall_gui.stop_button.configure(style="Danger.TButton")
            self.results_panel = TrainingResultsPanel(results, APP_DIR / "logs",
                on_log=lambda line: self.log_line("results", line))
            self.results_panel.pack(fill="both", expand=True)
        except Exception as exc:
            import traceback
            self.log_line("system", "ERROR loading tool panels:\n" + traceback.format_exc())
            self.env_status.set(f"Could not load tools: {exc}")
            self._show("system")
            return
        self.ready = True
        self._show("train")
        self.log_line("train", "Ready. Split, train an existing dataset, or pre-annotate images.")
        self.log_line("audit", "Ready. The audit reads annotations without modifying them.")
        self.log_line("recall", "Ready. Cached predictions can be reused for new precision thresholds.")
        self.log_line("results", "Select a recent training run or browse to an Ultralytics run folder.")

    def start_training(self, settings):
        self._start_job(settings, "train")

    def start_preannotation(self, settings):
        self._start_job(settings, "preannotate")

    def _start_job(self, settings, kind):
        if self.train_job is not None:
            messagebox.showwarning("Job in progress", "Another dataset job is running.", parent=self.root)
            return
        logdir = APP_DIR / "logs"
        logdir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        logfile = logdir / f"{kind}_{stamp}.log"
        config = logdir / f"{kind}_{stamp}_settings.json"
        submitted = dict(settings)
        cache_manifest = None
        if (kind == "train" and submitted.get("cache") == "disk" and
                submitted.get("cleanup_disk_cache", True)):
            cache_manifest = logdir / f"train_{stamp}_disk_cache_manifest.json"
            submitted["cache_cleanup_manifest"] = str(cache_manifest)
        config.write_text(json.dumps(submitted, indent=2, default=str), encoding="utf-8")
        worker = "run_preannotation_job.py" if kind == "preannotate" else "run_training_job.py"
        cmd = [str(env_python(self.env)), "-u", str(APP_DIR / worker), str(config)]
        flags = (subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0
        try:
            with logfile.open("w", encoding="utf-8") as stream:
                proc = subprocess.Popen(cmd, cwd=str(APP_DIR), env=process_environment(self.env),
                                        stdin=subprocess.DEVNULL, stdout=stream,
                                        stderr=subprocess.STDOUT, creationflags=flags,
                                        start_new_session=(os.name != "nt"))
        except OSError as exc:
            messagebox.showerror("Could not start training", str(exc), parent=self.root)
            return
        self.best_epoch_tracker.reset()
        self._refresh_best_epoch()
        self.train_metrics.set("Training: waiting for epoch results")
        self.val_metrics.set("Validation: waiting for evaluation")
        self.iteration_bar.configure(value=0)
        self.training_graph.reset()
        self.set_graph_visibility(kind == "train" and (
            bool(submitted.get("existing_data_yaml")) or
            bool(submitted.get("train_after_split", True))))
        self.iteration_label.set("Waiting for training batches" if kind == "train"
                                 else "Pre-annotation running")
        self.train_job = {"process": proc, "log": logfile, "offset": 0,
                          "pending": "", "stopping": False, "cache_manifest": cache_manifest,
                          "kind": kind, "settings": submitted, "run_dir": None}
        self.settings_expanded = False
        # With the form collapsed there is room for both curves and job text.
        self.log_visible = True
        self.log_toggle.configure(text="Hide log")
        self.train_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.train_state.set("Running " + (
            "pre-annotation" if kind == "preannotate" else
            "existing dataset training" if settings.get("existing_data_yaml") else
            "split + training" if settings["train_after_split"] else "split only"))
        if self.active == "train":
            self._show("train")
        self.log_line("train", "Started job " + str(proc.pid) + " · " + str(logfile))
        self.log_line("train", "Settings saved to " + str(config))

    def stop_training(self):
        job = self.train_job
        if not job or job["process"].poll() is not None:
            return
        if not messagebox.askyesno("Stop current job?", "Stop the active split/training job?\nThe existing output files will remain on disk.", parent=self.root):
            return
        job["stopping"] = True
        process = job["process"]
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               capture_output=True, timeout=15, check=False,
                               creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                os.killpg(process.pid, signal.SIGTERM)
        except (OSError, subprocess.TimeoutExpired, ProcessLookupError) as exc:
            self.log_line("train", "Stop request error: " + str(exc))
        self.train_state.set("Stopping job…")
        self.monitor_label.configure(text=self.train_state.get())
        self.stop_button.configure(state="disabled")
        self.log_line("train", "Stop requested. Output files remain untouched.")

    def _refresh_best_epoch(self):
        heading, scores, losses = self.best_epoch_tracker.display()
        self.best_heading.set(heading)
        self.best_scores.set(scores)
        self.best_losses.set(losses)

    def _training_output_line(self, line):
        from training_log import parse_epoch_metrics, parse_iteration_progress, parse_training_run_dir
        completed_run = parse_training_run_dir(line)
        if completed_run is not None:
            if self.train_job is not None:
                self.train_job["run_dir"] = completed_run
            return
        metrics = parse_epoch_metrics(line)
        if metrics is not None:
            self.training_graph.record(metrics)
            tracker = getattr(self, "best_epoch_tracker", None)
            if tracker is not None and tracker.record(metrics):
                self._refresh_best_epoch()
                self.log_line("train", "Best epoch updated: " + tracker.display()[0])
            return  # Keep structured graph events in job file, not the activity text.
        progress = parse_iteration_progress(line)
        if progress is not None:
            epoch, epochs, iteration, iterations = progress
            percent = 100 * iteration / iterations
            self.iteration_bar.configure(value=percent)
            suffix = " · validating / saving" if iteration == iterations else ""
            self.iteration_label.set(
                f"Epoch {epoch}/{epochs}  ·  {iteration}/{iterations} batches  ·  "
                f"{percent:.0f}%{suffix}")
        else:
            self.log_line("train", line)

    def _poll_training(self):
        job = self.train_job
        if not job:
            return
        proc = job["process"]
        try:
            with job["log"].open("r", encoding="utf-8", errors="replace") as stream:
                stream.seek(job["offset"])
                chunk = stream.read(48000)
                job["offset"] = stream.tell()
            if chunk:
                lines = (job["pending"] + chunk).replace("\r", "\n").split("\n")
                job["pending"] = lines.pop()
                for line in lines:
                    if line.strip():
                        self._training_output_line(line)
            returncode = proc.poll()
            if returncode is not None and job["log"].stat().st_size <= job["offset"]:
                manifest = job.get("cache_manifest")
                if manifest is not None and manifest.is_file():
                    try:
                        from train_and_split_tool_GUI import cleanup_disk_cache_manifest
                        count, errors = cleanup_disk_cache_manifest(manifest)
                        self.log_line("train", f"Post-stop disk cache cleanup: {count} .npy files removed.")
                        if errors:
                            self.log_line("train", f"WARNING: {len(errors)} cache files not removed; "
                                          f"manifest retained: {manifest}")
                    except (OSError, ValueError, KeyError) as exc:
                        self.log_line("train", f"WARNING: cache cleanup could not complete: {exc}")
                if job["pending"].strip():
                    self._training_output_line(job["pending"])
                if job["stopping"]:
                    label = "Job stopped"
                elif returncode == 0:
                    label = "Job completed successfully"
                else:
                    label = f"Job failed (exit code {returncode})"
                self.train_state.set(label)
                self.log_line("train", label + " · Full log: " + str(job["log"]))
                self.train_button.configure(state="normal")
                self.stop_button.configure(state="disabled")
                self.train_job = None
                self.settings_expanded = False
                self.set_graph_visibility(False)
                if not job["stopping"] and returncode == 0 and job.get("kind") == "train" and job.get("run_dir"):
                    if self.results_panel.load_run(job["run_dir"]):
                        self._show("results")
                        return
                if self.active == "train":
                    self._show("train")
        except OSError as exc:
            self.log_line("train", "Log read error: " + str(exc))

    def _tick(self):
        try:
            while True:
                kind, value = self.messages.get_nowait()
                if kind in ("log", "detail"):
                    self.log_line("system", str(value))
                elif kind == "status":
                    self.env_status.set(str(value))
                elif kind == "ready":
                    self.process_ready(*value)
                elif kind == "error":
                    self.env_busy = False
                    self.env_progress.stop()
                    self.recheck_btn.configure(state="normal")
                    self.env_status.set("Environment setup failed")
                    self.env_info.set("Review the activity log and retry setup after fixing the problem.")
                    self.nav_hint.configure(text="Environment setup failed")
                    self.status_dot.configure(fg=DANGER)
                    self.status_label.configure(text="Setup error")
                    self.log_line("system", "ERROR: " + str(value))
                    self._show("system")
                    messagebox.showerror("Environment setup failed", str(value), parent=self.root)
        except queue.Empty:
            pass
        self._poll_training()
        try:
            if self.root.winfo_exists():
                self.root.after(200, self._tick)
        except tk.TclError:
            pass

    def open_logs_folder(self):
        folder = APP_DIR / "logs"
        folder.mkdir(parents=True, exist_ok=True)
        try:
            if os.name == "nt":
                os.startfile(str(folder))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)], stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
        except OSError as exc:
            messagebox.showerror("Could not open logs", str(exc), parent=self.root)

    def on_close(self):
        active_train = self.train_job and self.train_job["process"].poll() is None
        active_audit = self.audit_gui and self.audit_gui.busy
        active_recall = self.recall_gui and self.recall_gui.busy
        if active_train and not messagebox.askyesno(
                "Training is still running",
                "Close the interface? Training continues in the background and its "
                "complete output is available in the saved log file.", parent=self.root):
            return
        if active_audit and not messagebox.askyesno(
                "Audit is running", "Cancel the audit safely and close the application?",
                parent=self.root):
            return
        if active_recall and not messagebox.askyesno(
                "Evaluation is running", "Close now? The evaluation will be interrupted.",
                parent=self.root):
            return
        if active_audit:
            self.audit_gui.cancel()
        if active_recall:
            self.recall_gui.cancel()
        if active_audit or active_recall:
            # Let cooperative cancellation complete before closing the GUI.
            self._wait_for_evaluations_shutdown()
        else:
            self.root.destroy()

    def _wait_for_evaluations_shutdown(self):
        if ((self.audit_gui and self.audit_gui.busy) or
                (self.recall_gui and self.recall_gui.busy)):
            self.root.after(250, self._wait_for_evaluations_shutdown)
        else:
            self.root.destroy()

    def run(self):
        # Action row remains in the same workspace as the training form.
        # The train button from the original module handles validated submit.
        self.root.mainloop()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Unified CV Waste Studio")
    parser.add_argument("--env-prefix", default=None)
    parser.add_argument("--conda-exe", default=None)
    parser.add_argument("--new-instance", action="store_true",
                        help="Open an additional Studio window rather than focusing the current one")
    args = parser.parse_args(argv)
    if args.env_prefix and not args.conda_exe or args.conda_exe and not args.env_prefix:
        parser.error("--env-prefix and --conda-exe must both be specified")
    # Explicit new-window launches bypass the Windows mutex, including across
    # the base-Conda-to-managed-Python handoff. Normal launches still focus.
    allow_multiple = (args.new_instance or
                      os.environ.get("CV_WASTE_STUDIO_NEW_INSTANCE") == "1")
    if not allow_multiple and another_instance_running():
        activate_existing_window()
        return
    try:
        prepared = None
        if args.env_prefix:
            if not allow_multiple and not claim_gui_instance():
                activate_existing_window()
                return
            env = Path(args.env_prefix).resolve()
            cached = cached_environment(expected_conda=args.conda_exe, expected_prefix=env)
            if cached:
                prepared = cached
            else:
                details = probe_env(args.conda_exe, env, needs_cuda=False, timeout=480)
                if not details.get("ok"):
                    parser.error("Managed environment is not usable: " + str(details.get("error", details)))
                prepared = (args.conda_exe, env, details)
                try:
                    save_cached_environment(*prepared)
                except OSError:
                    pass
        Studio(prepared, allow_multiple=allow_multiple).run()
    finally:
        release_gui_instance()


if __name__ == "__main__":
    main()
