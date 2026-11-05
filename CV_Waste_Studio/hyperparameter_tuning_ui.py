"""CV Waste Studio's native Ultralytics hyperparameter tuning workspace."""
from __future__ import annotations
import json
import os
from datetime import datetime
from pathlib import Path
import re
import signal
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from hyperparameter_tuning import PRESETS, RANGES, best_trial, create_run_name, read_trials, validate_settings
from tuning_refinement import (load_tuning_source, propose_refinement, saved_settings,
                               source_resume_status)
from ray_tune_support import read_ray_trials, INSTALL_PACKAGES
from tuning_charts import TuningCharts

ACCENT = '#405DE6'


def ray_install_command(python):
    """Install the optional Ray stack into Studio's selected Conda Python."""
    return [str(python), '-m', 'pip', 'install', '--no-input', *INSTALL_PACKAGES,
            'pydantic>=2,<2.12']


class HyperparameterTuningPanel(tk.Frame):
    def __init__(self, parent, *, env_python, process_env, logs_dir, can_start=None, on_log=None):
        super().__init__(parent,bg='white')
        self.env_python=env_python
        self.process_env=process_env
        self.logs_dir=Path(logs_dir)
        self.can_start=can_start
        self.on_log=on_log
        self.job=None
        self.folder=None
        self.trials=[]
        self._last_ndjson_size=None
        self._selected_trial=None
        self.current_engine='native'
        self.ray_available=False
        self.ray_probe=None
        self.ray_report=None
        self.ray_install_job=None
        self._ray_checked_python=None  # Probe on first HPT visit, not at Studio startup.
        self._ray_offer_install=True
        self._build()

    @property
    def busy(self):
        return self.job is not None and self.job['process'].poll() is None

    @property
    def installing(self):
        return self.ray_install_job is not None and self.ray_install_job['process'].poll() is None

    def on_show(self):
        """Lazy compatibility check: called when 06 Hyperparameter Tuning opens."""
        selected_python=str(self.env_python())
        if self._ray_checked_python != selected_python and not self.installing:
            self.check_ray_compatibility()

    def _row(self, parent, row, caption, var, browse=None, width=16):
        ttk.Label(parent,text=caption).grid(row=row,column=0,sticky='w',pady=4,padx=(0,14))
        ttk.Entry(parent,textvariable=var,width=width).grid(row=row,column=1,sticky='ew',pady=4)
        if browse:
            ttk.Button(parent,text='Browse…',command=browse).grid(row=row,column=2,sticky='w',padx=10)

    def _build(self):
        main=ttk.Frame(self,padding=(18,12))
        main.pack(fill='both',expand=True)
        heading=ttk.Frame(main)
        heading.pack(fill='x',pady=(0,3))
        ttk.Label(heading,text='YOLO hyperparameter tuning',font=('Segoe UI',14,'bold')).pack(side='left')
        self.stop_btn=ttk.Button(heading,text='Stop tuning',style='Danger.TButton',state='disabled',command=self.stop)
        self.stop_btn.pack(side='right')
        actions=ttk.Frame(main)
        actions.pack(fill='x',pady=(0,7))
        self.mode_btn=ttk.Button(actions,text='Show settings',command=self.toggle_mode)
        self.mode_btn.pack(side='right')
        self.mode_btn.pack_forget()
        self.resume_btn=ttk.Button(actions,text='Resume / extend',command=self.resume_previous,state='disabled')
        self.resume_btn.pack(side='right',padx=(0,8))
        ttk.Button(actions,text='Continue saved HPT…',command=self.continue_from_folder).pack(side='right',padx=(0,8))
        ttk.Button(actions,text='Open previous run',command=self.open_previous).pack(side='right',padx=(0,8))
        self.status=tk.StringVar(value='Ready · select your chosen architecture and tune on train/val only')
        ttk.Label(main,textvariable=self.status,style='Muted.TLabel').pack(fill='x',pady=(0,9))
        self.setup=ttk.Frame(main)
        self.setup.pack(fill='both',expand=True)
        # Search engine stays visible at the top of the setup page.
        engines=ttk.Frame(self.setup)
        engines.pack(fill='x',pady=(0,7))
        ttk.Label(engines,text='Search engine:',font=('Segoe UI',9,'bold')).pack(side='left',padx=(0,10))
        self.engine=tk.StringVar(value='native')
        ttk.Radiobutton(engines,text='Native genetic evolution',variable=self.engine,
                        value='native',command=self._update_engine).pack(side='left',padx=(0,12))
        self.ray_radio=ttk.Radiobutton(engines,text='Bayesian TPE + ASHA',variable=self.engine,
                       value='ray',state='disabled',command=self._update_engine)
        self.ray_radio.pack(side='left',padx=(0,10))
        self.check_ray_btn=ttk.Button(engines,text='Check compatibility',command=self.check_ray_compatibility)
        self.check_ray_btn.pack(side='right')
        self.install_ray_btn=ttk.Button(engines,text='Install Ray / Optuna…',state='disabled',
                                        command=self.install_ray_dependencies)
        self.install_ray_btn.pack(side='right',padx=(0,8))
        ray_settings=ttk.Frame(self.setup)
        ray_settings.pack(fill='x',pady=(0,5))
        self.compat_label=tk.StringVar(value='Ray availability: checked when you open this section.')
        ttk.Label(ray_settings,textvariable=self.compat_label,style='Muted.TLabel',
                  wraplength=640).pack(side='left',fill='x',expand=True)
        self.grace=tk.StringVar(value='10')
        self.grace_label=ttk.Label(ray_settings,text='ASHA min epochs:')
        self.grace_entry=ttk.Entry(ray_settings,textvariable=self.grace,width=5)
        canvas=tk.Canvas(self.setup,highlightthickness=0,bd=0,bg='white')
        scroll=ttk.Scrollbar(self.setup,orient='vertical',command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right',fill='y')
        canvas.pack(side='left',fill='both',expand=True)
        body=ttk.Frame(canvas,padding=(4,2,12,24))
        window=canvas.create_window((0,0),window=body,anchor='nw')
        body.bind('<Configure>',lambda _:canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>',lambda e:canvas.itemconfigure(window,width=e.width))
        def wheel(event):
            delta = -1 if event.num == 4 else (1 if event.num == 5 else -int(event.delta/120))
            canvas.yview_scroll(delta,'units')
        def wheel_on(_event):
            for sequence in ('<MouseWheel>','<Button-4>','<Button-5>'):
                canvas.bind_all(sequence,wheel)
        def wheel_off(_event):
            for sequence in ('<MouseWheel>','<Button-4>','<Button-5>'):
                canvas.unbind_all(sequence)
        canvas.bind('<Enter>',wheel_on)
        canvas.bind('<Leave>',wheel_off)
        body.columnconfigure(1,weight=1)
        self.dataset=tk.StringVar()
        self.model=tk.StringVar(value='yolo11l.pt')
        self.output=tk.StringVar()
        self.name=tk.StringVar()
        self.iterations=tk.StringVar(value='20')
        self.epochs=tk.StringVar(value='25')
        self.imgsz=tk.StringVar(value='640')
        self.batch=tk.StringVar(value='8')
        self.workers=tk.StringVar(value='8')
        self.cache=tk.StringVar(value='off')
        self.optimizer=tk.StringVar(value='AdamW')
        self.device=tk.StringVar(value='auto')
        self.preset=tk.StringVar(value='Balanced')
        ttk.Label(body,text='01  Architecture & dataset',font=('Segoe UI',12,'bold')).grid(row=0,column=0,columnspan=3,sticky='w',pady=(0,8))
        self._row(body,1,'Existing YOLO data.yaml',self.dataset,
                  browse=lambda:self._file(self.dataset,'Choose YOLO data.yaml',[('YAML','*.yaml *.yml')]))
        mrow=ttk.Frame(body)
        mrow.grid(row=2,column=1,sticky='ew',pady=4)
        ttk.Combobox(mrow,textvariable=self.model,values=(
            'yolov8n.pt','yolov8s.pt','yolov8m.pt','yolov8l.pt','yolov8x.pt',
            'yolo10n.pt','yolo10s.pt','yolo10m.pt','yolo10l.pt','yolo10x.pt',
            'yolo11n.pt','yolo11s.pt','yolo11m.pt','yolo11l.pt','yolo11x.pt',
            'yolo12n.pt','yolo12s.pt','yolo12m.pt','yolo12l.pt','yolo12x.pt',
            'yolo26n.pt','yolo26s.pt','yolo26m.pt','yolo26l.pt','yolo26x.pt'),
            width=22).pack(side='left')
        ttk.Button(mrow,text='Choose .pt weights',command=lambda:self._file(
            self.model,'Choose trained checkpoint',[('PyTorch weights','*.pt')])).pack(side='left',padx=10)
        ttk.Label(body,text='Initial model / weights').grid(row=2,column=0,sticky='w',pady=4)
        ttk.Label(body,text='Tune only on training/validation splits; reserve test for final model assessment.',
                  style='Muted.TLabel').grid(row=3,column=1,columnspan=2,sticky='w',pady=(0,9))
        ttk.Separator(body).grid(row=4,column=0,columnspan=3,sticky='ew',pady=10)
        ttk.Label(body,text='02  Tuning budget & training resources',font=('Segoe UI',12,'bold')).grid(
            row=5,column=0,columnspan=3,sticky='w',pady=(0,8))
        self._row(body,6,'Trials (iterations)',self.iterations)
        self._row(body,7,'Epochs per trial',self.epochs)
        self._row(body,8,'Batch size',self.batch)
        self._row(body,9,'Image size',self.imgsz)
        self._row(body,10,'Dataloader workers',self.workers)
        for row,label,var,choices in (
            (11,'Cache',self.cache,('off','disk','ram')),
            (12,'Optimizer',self.optimizer,('AdamW','SGD','Adam','NAdam','RAdam','RMSProp','auto')),
            (13,'Device',self.device,('auto','gpu','cpu'))):
            ttk.Label(body,text=label).grid(row=row,column=0,sticky='w',pady=4)
            ttk.Combobox(body,textvariable=var,values=choices,state='readonly',width=17).grid(
                row=row,column=1,sticky='w',pady=4)
        ttk.Label(body,text='Each trial retrains independently using the native Ultralytics tuner; '
                  'cache=off avoids .npy files and excessive shared RAM on Windows; disk cache is cleaned after tuning.',
                  style='Muted.TLabel',wraplength=660).grid(row=14,column=1,columnspan=2,sticky='w',pady=(4,8))
        ttk.Separator(body).grid(row=15,column=0,columnspan=3,sticky='ew',pady=10)
        ttk.Label(body,text='03  Search space',font=('Segoe UI',12,'bold')).grid(row=16,column=0,columnspan=3,sticky='w',pady=(0,8))
        picks=ttk.Frame(body)
        picks.grid(row=17,column=0,columnspan=3,sticky='w',pady=5)
        ttk.Label(picks,text='Preset').pack(side='left')
        ttk.Combobox(picks,textvariable=self.preset,values=tuple(PRESETS),state='readonly',width=20).pack(
            side='left',padx=(10,12))
        ttk.Button(picks,text='Apply preset',command=self.apply_preset).pack(side='left')
        ttk.Button(picks,text='Import best YAML…',command=self.import_best_yaml).pack(side='left',padx=(14,5))
        ttk.Label(picks,text='±').pack(side='left')
        self.search_width=tk.StringVar(value='30')
        ttk.Entry(picks,textvariable=self.search_width,width=4).pack(side='left',padx=(2,2))
        ttk.Label(picks,text='% search width').pack(side='left')
        sourcebar=ttk.Frame(body)
        sourcebar.grid(row=18,column=0,columnspan=3,sticky='ew',pady=(1,6))
        ttk.Button(sourcebar,text='Refine previous HPT folder…',command=self.refine_previous).pack(side='left')
        ttk.Button(sourcebar,text='Use previous .pt weights…',command=self.use_previous_checkpoint).pack(side='left',padx=(8,12))
        self.refine_message=tk.StringVar(value='Refine starts a NEW search. True resume needs the original run folder, not just a YAML or .pt checkpoint.')
        ttk.Label(sourcebar,textvariable=self.refine_message,foreground='#566884',
                  wraplength=460).pack(side='left',fill='x',expand=True)

        ttk.Label(body,text='Check parameters to tune; adjust bounds (must stay within native search limits). '
                  'Full native uses the selected engine’s default search space.',
                  style='Muted.TLabel',wraplength=725).grid(row=19,column=0,columnspan=3,sticky='w',pady=(0,5))
        table=ttk.Frame(body)
        table.grid(row=20,column=0,columnspan=3,sticky='w',pady=(0,12))
        ttk.Label(table,text='Tune',font=('Segoe UI',9,'bold')).grid(row=0,column=0,sticky='w',padx=6)
        ttk.Label(table,text='Parameter',font=('Segoe UI',9,'bold')).grid(row=0,column=1,sticky='w',padx=6)
        ttk.Label(table,text='Minimum',font=('Segoe UI',9,'bold')).grid(row=0,column=2,sticky='w',padx=6)
        ttk.Label(table,text='Maximum',font=('Segoe UI',9,'bold')).grid(row=0,column=3,sticky='w',padx=6)
        ttk.Label(table,text='Search boundary',font=('Segoe UI',9,'bold')).grid(row=0,column=4,sticky='w',padx=6)
        self.boundary_labels={}
        self.space_widgets={}
        for j,(key,(lo,hi)) in enumerate(RANGES.items(),1):
            enabled=tk.BooleanVar(value=key in PRESETS['Balanced'])
            low=tk.StringVar(value=f'{lo:g}')
            high=tk.StringVar(value=f'{hi:g}')
            ttk.Checkbutton(table,variable=enabled).grid(row=j,column=0,sticky='w',padx=6,pady=2)
            ttk.Label(table,text=key,width=19).grid(row=j,column=1,sticky='w',padx=6,pady=2)
            ttk.Entry(table,textvariable=low,width=15).grid(row=j,column=2,padx=6,pady=2)
            ttk.Entry(table,textvariable=high,width=15).grid(row=j,column=3,padx=6,pady=2)
            flag=tk.StringVar(value='')
            ttk.Label(table,textvariable=flag,foreground='#AD6A12',width=27).grid(row=j,column=4,sticky='w',padx=6)
            self.boundary_labels[key]=flag
            self.space_widgets[key]=(enabled,low,high)
        ttk.Separator(body).grid(row=21,column=0,columnspan=3,sticky='ew',pady=10)
        ttk.Label(body,text='04  Outputs',font=('Segoe UI',12,'bold')).grid(row=22,column=0,columnspan=3,sticky='w',pady=(0,8))
        self._row(body,23,'Tuning results root',self.output,
                  browse=lambda:self._folder(self.output))
        self._row(body,24,'Run name (optional)',self.name)
        ttk.Label(body,text='Each run gets its own folder with best_hyperparameters.yaml, '
                  'best.pt, native tune figures and full trial history.',style='Muted.TLabel',
                  wraplength=680).grid(row=25,column=1,columnspan=2,sticky='w',pady=(0,10))
        self.start_btn=ttk.Button(body,text='Start hyperparameter tuning',style='Accent.TButton',command=self.start)
        self.start_btn.grid(row=26,column=2,sticky='e',pady=8)

        self.monitor=ttk.Frame(main)
        summary=ttk.Frame(self.monitor)
        summary.pack(fill='x',pady=(0,10))
        self.headline=tk.StringVar(value='No trial results yet')
        self.best_label=tk.StringVar(value='Best validation fitness  —')
        self.best_hp=tk.StringVar(value='Choose a search space and begin tuning')
        ttk.Label(summary,textvariable=self.headline,font=('Segoe UI',12,'bold')).pack(anchor='w')
        ttk.Label(summary,textvariable=self.best_label,font=('Segoe UI',11,'bold'),foreground='#245A9F').pack(anchor='w',pady=(3,0))
        ttk.Label(summary,textvariable=self.best_hp,style='Muted.TLabel',wraplength=880).pack(anchor='w',pady=(2,0))
        self.trial_status=tk.StringVar(value='Waiting for trials')
        progress_title=ttk.Frame(self.monitor)
        progress_title.pack(fill='x',pady=(7,3))
        ttk.Label(progress_title,text='TUNING PROGRESS',font=('Segoe UI',9,'bold')).pack(side='left')
        ttk.Label(progress_title,textvariable=self.trial_status).pack(side='right')
        self.progress=ttk.Progressbar(self.monitor,mode='determinate',maximum=100)
        self.progress.pack(fill='x',pady=(0,6))
        self.epoch_status=tk.StringVar(value='Current trial: preparing')
        ttk.Label(self.monitor,textvariable=self.epoch_status,style='Muted.TLabel').pack(fill='x')
        self.epoch_progress=ttk.Progressbar(self.monitor,mode='determinate',maximum=100)
        self.epoch_progress.pack(fill='x',pady=(3,7))
        graph_head=ttk.Frame(self.monitor)
        graph_head.pack(fill='x')
        ttk.Label(graph_head,text='TRIAL PERFORMANCE & SEARCH HISTORY',font=('Segoe UI',9,'bold')).pack(side='left')
        self.log_btn=ttk.Button(graph_head,text='Show log',command=self.toggle_log)
        self.log_btn.pack(side='right',padx=(6,0))
        ttk.Label(graph_head,text='Hover for blue halo · click chart to expand · × to close',
                  style='Muted.TLabel').pack(side='right')
        choose=ttk.Frame(self.monitor)
        choose.pack(fill='x',pady=(5,4))
        ttk.Label(choose,text='Inspect parameter:').pack(side='left')
        self.param=tk.StringVar(value='lr0')
        ttk.Combobox(choose,textvariable=self.param,values=tuple(RANGES),state='readonly',width=17).pack(
            side='left',padx=(8,12))
        self.param.trace_add('write',lambda *_:self.charts.set_parameter(self.param.get()))
        ttk.Button(choose,text='Open results folder',command=self.open_results).pack(side='right')
        ttk.Button(choose,text='Copy best checkpoint',command=self.copy_best_checkpoint).pack(side='right',padx=(0,8))
        ttk.Button(choose,text='Export best YAML…',command=self.export_best_yaml).pack(side='right',padx=(0,8))
        ttk.Button(choose,text='Copy best YAML',command=self.copy_best_yaml).pack(side='right',padx=(0,8))
        self.charts=TuningCharts(self.monitor,height=220)
        self.charts.pack(fill='both',expand=True)
        lower=ttk.Frame(self.monitor)
        lower.pack(fill='both',expand=True,pady=(7,0))
        cols=('trial','fitness','map50','map5095','precision','recall','epochs','status','parameters')
        self.table=ttk.Treeview(lower,columns=cols,show='headings',height=4)
        titles={'trial':('Trial',52),'fitness':('Fitness',77),'map50':('mAP50',80),
                'map5095':('mAP50–95',87),'precision':('Precision',82),
                'recall':('Recall',82),'epochs':('Ep.',53),'status':('Status',88),
                'parameters':('Tuned hyperparameters',340)}
        for key,(label,width) in titles.items():
            self.table.heading(key,text=label)
            self.table.column(key,width=width,stretch=key=='parameters',anchor='w')
        self.table.pack(side='left',fill='both',expand=True)
        side=ttk.Scrollbar(lower,orient='vertical',command=self.table.yview)
        side.pack(side='right',fill='y')
        self.table.configure(yscrollcommand=side.set)
        self.table.bind('<<TreeviewSelect>>',self._table_select)
        self.detail=tk.StringVar(value='Select a trial to inspect its parameter values.')
        ttk.Label(self.monitor,textvariable=self.detail,style='Muted.TLabel',wraplength=900).pack(fill='x',pady=(4,5))
        self._log=tk.Text(self.monitor,bg='#F8FAFD',height=3,font=('Consolas',9),bd=0,wrap='word')
        self._log_visible=False
        self._log.configure(state='disabled')
        self.in_setup=True
        self._update_engine()

    def _file(self,var,title,filetypes):
        selected=filedialog.askopenfilename(parent=self,title=title,filetypes=filetypes)
        if selected:var.set(selected)

    def _folder(self,var):
        selected=filedialog.askdirectory(parent=self,title='Select tuning results directory')
        if selected:var.set(selected)

    def apply_preset(self):
        keys=set(PRESETS.get(self.preset.get(), ()))
        for key,(enabled,_,_) in self.space_widgets.items():
            enabled.set(key in keys)

    def _update_engine(self):
        ray = self.engine.get() == 'ray'
        if ray and not self.ray_available:
            self.engine.set('native')
            ray = False
        if ray:
            self.grace_entry.pack(side='right',padx=(5,0))
            self.grace_label.pack(side='right')
        else:
            self.grace_label.pack_forget()
            self.grace_entry.pack_forget()
        self.start_btn.configure(text='Start Bayesian TPE + ASHA' if ray else 'Start native genetic tuning')

    def check_ray_compatibility(self, *, offer_install=True):
        """Probe in the selected Conda interpreter, only after entering HPT.

        Ray, Torch and CUDA are never imported into the Tk process.  No
        packages are installed until the operator explicitly approves.
        """
        if (self.ray_probe is not None and self.ray_probe.poll() is None) or self.installing:
            return
        self._ray_checked_python=str(self.env_python())
        self._ray_offer_install=offer_install
        self.compat_label.set('Checking Ray + Optuna + Ultralytics in selected Conda environment…')
        self.ray_radio.configure(state='disabled')
        self.install_ray_btn.configure(state='disabled')
        self.check_ray_btn.configure(state='disabled')
        self.ray_available=False
        self.engine.set('native')
        self._update_engine()
        try:
            flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
            env=self.process_env()
            env.setdefault('KMP_DUPLICATE_LIB_OK','TRUE')
            self.ray_probe=subprocess.Popen(
                [str(self.env_python()), '-u', str(Path(__file__).with_name('probe_ray_tune.py'))],
                cwd=str(Path(__file__).parent), env=env,
                stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                creationflags=flags)
            self.after(250,self._finish_ray_probe)
        except (OSError,ValueError) as exc:
            self._ray_unavailable(str(exc), offer_install=offer_install)

    def _finish_ray_probe(self):
        if self.ray_probe is None:return
        if self.ray_probe.poll() is None:
            self.after(250,self._finish_ray_probe)
            return
        out,err=self.ray_probe.communicate()
        self.ray_probe=None
        self.check_ray_btn.configure(state='normal')
        try:
            report=json.loads(out.decode('utf-8','replace').strip().splitlines()[-1])
            if not isinstance(report,dict):raise ValueError('Invalid probe result')
        except (ValueError,IndexError) as exc:
            report={'available':False,'reason':err.decode('utf-8','replace')[-800:] or str(exc)}
        self.ray_report=report
        if report.get('available'):
            self.ray_available=True
            self.ray_radio.configure(state='normal')
            self.install_ray_btn.configure(state='disabled')
            versions=report.get('versions',{})
            self.compat_label.set('Ray ready · Ultralytics '+versions.get('ultralytics','?')+
                 ' · Ray '+versions.get('ray','?')+' · Optuna '+versions.get('optuna','?'))
            self._logline('Bayesian TPE + ASHA available in '+str(self.env_python()))
        else:
            self._ray_unavailable(report.get('reason','Unknown compatibility problem'),
                                  offer_install=self._ray_offer_install)

    def _ray_unavailable(self,reason, *, offer_install=True):
        self.ray_available=False
        self.ray_radio.configure(state='disabled')
        self.install_ray_btn.configure(state='normal')
        self.check_ray_btn.configure(state='normal')
        self.engine.set('native')
        self._update_engine()
        self.compat_label.set('Bayesian mode unavailable · '+str(reason)[:170])
        self._logline('Ray / Optuna compatibility check: '+str(reason)[:350])
        if offer_install and messagebox.askyesno('Optional Bayesian tuning dependencies',
            'Ray Tune is unavailable in the selected Conda environment.\n\n'+str(reason)[:350]+
            '\n\nInstall optional Ray Tune / Optuna dependencies now?\n\n'+
            'The installation will run in the background using:\n'+
            str(self.env_python())+'\n\n'+
            'It may update packages in that Conda environment. '
            'You can decline and continue using native genetic tuning.',parent=self):
            self.install_ray_dependencies(confirmed=True)

    def install_ray_dependencies(self, confirmed=False):
        """Install only after explicit permission, asynchronously and with a log."""
        if self.installing or (self.ray_probe is not None and self.ray_probe.poll() is None):
            return
        if self.busy or (self.can_start and not self.can_start()):
            messagebox.showwarning('Another job active',
                'Finish active Studio GPU jobs before modifying the selected Conda environment.',parent=self)
            return
        python=str(self.env_python())
        command=ray_install_command(python)
        if not confirmed and not messagebox.askyesno(
                'Install optional Bayesian tuning packages?',
                'Install Ray Tune, Optuna and compatible Pydantic in the selected Conda Python?\n\n'+
                ' '.join('"'+arg+'"' if ' ' in arg else arg for arg in command)+
                '\n\nThis will modify this Conda environment, not the system Python. '
                'It can update existing packages. Proceed?',parent=self):
            return
        self.logs_dir.mkdir(parents=True,exist_ok=True)
        logfile=self.logs_dir / ('hpt_ray_install_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.log')
        env=self.process_env()
        flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
        try:
            with logfile.open('w',encoding='utf-8') as output:
                proc=subprocess.Popen(command,cwd=str(Path(__file__).parent),env=env,
                    stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,
                    creationflags=flags)
        except (OSError,ValueError) as exc:
            self.compat_label.set('Ray installation could not start · '+str(exc)[:140])
            self._logline('Ray installation failed to start: '+str(exc))
            messagebox.showerror('Could not start package installation',str(exc),parent=self)
            return
        self.ray_install_job={'process':proc,'log':logfile,'offset':0}
        self.ray_radio.configure(state='disabled')
        self.install_ray_btn.configure(state='disabled')
        self.check_ray_btn.configure(state='disabled')
        self.start_btn.configure(state='disabled')
        self.compat_label.set('Installing Ray + Optuna in selected Conda environment…')
        self._logline('Installing optional Ray dependencies into: '+python)
        self._logline('Full installation log: '+str(logfile))
        self.after(500,self._poll_ray_install)

    def _poll_ray_install(self):
        job=self.ray_install_job
        if job is None or not self.winfo_exists():return
        try:
            with job['log'].open(encoding='utf-8',errors='replace') as f:
                f.seek(job['offset'])
                chunk=f.read(32768)
                job['offset']=f.tell()
            if chunk:
                # Progress is available in the permanent log; keep GUI concise.
                for line in chunk.splitlines()[-6:]:
                    if line.strip():self._logline('pip: '+line.strip()[:220])
        except OSError:
            pass
        code=job['process'].poll()
        if code is None:
            self.after(500,self._poll_ray_install)
            return
        self.ray_install_job=None
        self.check_ray_btn.configure(state='normal')
        self.start_btn.configure(state='normal')
        if code == 0:
            self._logline('Ray / Optuna installation completed. Rechecking compatibility…')
            self.compat_label.set('Packages installed. Verifying compatibility…')
            self.check_ray_compatibility(offer_install=False)
        else:
            self.ray_radio.configure(state='disabled')
            self.install_ray_btn.configure(state='normal')
            self.compat_label.set('Ray installation failed (exit '+str(code)+'). See '+str(job['log']))
            self._logline('Ray installation failed, exit '+str(code)+'; see '+str(job['log']))

    def import_best_yaml(self):
        """Refine from a standalone YAML; it cannot restore tuning history."""
        path=filedialog.askopenfilename(parent=self,title='Choose saved best_hyperparameters.yaml',
                        filetypes=[('Ultralytics YAML','*.yaml *.yml')])
        if path:
            self._apply_refinement_source(path)

    def refine_previous(self):
        """Start a NEW search in a narrower range centred on a prior winner."""
        if self.busy:
            messagebox.showinfo('Tuning is active','Stop the current search before changing its configuration.',parent=self)
            return
        path=filedialog.askdirectory(parent=self,title='Choose completed HPT results folder to refine')
        if path:
            self._apply_refinement_source(path)

    def _apply_refinement_source(self,path):
        try:
            source=load_tuning_source(path)
            width=float(self.search_width.get().strip().replace(',','.'))/100
            proposal=propose_refinement(source['hyperparameters'],source['previous_space'],width)
            if not proposal:
                raise ValueError('No supported search ranges could be generated.')
            for key,(enabled,low,high) in self.space_widgets.items():
                enabled.set(key in proposal)
                self.boundary_labels[key].set('')
                if key in proposal:
                    item=proposal[key]
                    low.set(f"{item['min']:.8g}")
                    high.set(f"{item['max']:.8g}")
                    self.boundary_labels[key].set(item['warning'])
            self.preset.set('Custom (YAML)' if Path(path).is_file() else 'Custom (refinement)')
            origin=source['folder']
            cfg=source['settings']
            if not (origin and cfg):
                # Standalone best YAMLs are commonly imported after another
                # run has been viewed. Never reuse the old run name by accident.
                from datetime import datetime
                base=re.sub(r'[^a-zA-Z0-9_-]','_',Path(path).stem)[:55]
                self.name.set(base+'_refined_'+datetime.now().strftime('%H%M%S'))
            if origin and cfg:
                # These settings remain editable. Importing from an arbitrary
                # YAML does NOT modify dataset or model paths.
                msg=(f"Use the original run's model, dataset and resource settings for the "
                     f"new refinement?\n\nSource: {origin.name}\n"
                     'The original run and its checkpoints will not be modified.')
                if messagebox.askyesno('Use original experiment settings?',msg,parent=self):
                    for field in ('model','data','epochs','imgsz','batch','workers',
                                  'cache','optimizer','device'):
                        widget={'data':'dataset'}.get(field,field)
                        var=getattr(self,widget,None)
                        if var is not None and field in cfg:
                            var.set(str(cfg[field]))
                    self.output.set(str(origin.parent))
                    from datetime import datetime
                    base=re.sub(r'[^a-zA-Z0-9_-]','_',origin.name)[:65]
                    self.name.set(base+'_refined_'+datetime.now().strftime('%H%M%S'))
                    # Engine deliberately stays user-selectable for cross-engine
                    # refinement (Bayesian -> native genetic, or vice versa).
            alerts=[k for k,item in proposal.items() if item['warning']]
            self.refine_message.set(f"Refining {len(proposal)} parameters from {Path(path).name}"
                +(f" · {len(alerts)} at search boundaries: {', '.join(alerts)}" if alerts else ' · no boundary hits'))
            self._logline('Refinement seed: '+str(path)+' | '+self.refine_message.get())
            self._mode(True)
            messagebox.showinfo('New refinement search ready',
                f"Imported {len(proposal)} editable search ranges (±{width*100:g}%).\n"
                +(f"Parameters near previous search limits: {', '.join(alerts)}.\n"
                  'Their bounds may need expanding; warnings appear in the table.\n' if alerts else '')+
                'This starts a NEW HPT experiment, not a resume. Choose engine and a new run name, review the ranges, then Start.',
                parent=self)
        except (OSError,ValueError) as exc:
            messagebox.showerror('Could not refine previous HPT',str(exc),parent=self)

    def use_previous_checkpoint(self):
        """Initialise a NEW tuning search from weights, not tuner state."""
        if self.busy:return
        filename=filedialog.askopenfilename(parent=self,title='Choose previous model checkpoint (.pt)',
                 filetypes=[('PyTorch checkpoint','*.pt')])
        if not filename:return
        if messagebox.askyesno('Use a previous checkpoint?',
            'This sets the initial model weights for a NEW tuning search. It does not resume the genetic / Bayesian search history. '
            'A checkpoint already trained on this data may change the meaning of the comparison.\n\nUse these weights?',parent=self):
            self.model.set(filename)
            self.refine_message.set('Initial model set to previous .pt weights · new search, not history resume')

    def _load_saved_config_to_form(self,cfg):
        for key in ('model','epochs','iterations','batch','imgsz','workers','cache','optimizer','device'):
            if key in cfg:
                getattr(self,key).set(str(cfg[key]))
        self.dataset.set(str(cfg.get('data','')))
        self.output.set(str(cfg.get('output_dir','')))
        self.name.set(str(cfg.get('name','')))
        space=cfg.get('space')
        for key,(enabled,low,high) in self.space_widgets.items():
            self.boundary_labels[key].set('')
            enabled.set(bool(space is not None and key in space))
            if space and key in space:
                low.set(str(space[key][0]))
                high.set(str(space[key][1]))
        self.preset.set('Custom (saved run)' if space is not None else 'Full native')
        if 'grace_period' in cfg:
            self.grace.set(str(cfg['grace_period']))
        mode=cfg.get('engine','native')
        if mode=='native' or self.ray_available:
            self.engine.set(mode)
            self._update_engine()

    def continue_from_folder(self):
        """Load an original experiment and prepare a genuine stateful resume."""
        if self.busy:return
        path=filedialog.askdirectory(parent=self,title='Choose ORIGINAL HPT folder (YAML alone cannot resume)')
        if not path:return
        try:
            folder=Path(path).resolve()
            ok,detail=source_resume_status(folder)
            if not ok:
                raise ValueError(detail)
            self.load_folder(folder)
            cfg=self._saved_run_settings()
            self._load_saved_config_to_form(cfg)
            if cfg.get('engine')=='ray':
                self.iterations.set(str(cfg['iterations']))
                if not self.ray_available:
                    raise ValueError('Ray mode is unavailable in the current Conda environment. Recheck compatibility first.')
                self.refine_message.set('Ray: resume unfinished trials only; use Refine for more trials or a different search space.')
            else:
                done=len(self.trials)
                initial=int(cfg['iterations'])
                self.iterations.set(str(initial if done<initial else done+10))
                self.refine_message.set(f'Native: {done} recorded trials. Edit TOTAL trial count to extend this exact experiment.')
            self.status.set('Original HPT loaded · '+detail)
            self._update_resume_button()
            self._mode(True)
            messagebox.showinfo('Original tuning run loaded',
                detail+'\n\n'+self.refine_message.get()+
                '\nUse Resume / extend to continue. Parameters and prior search bounds will be preserved.',parent=self)
        except (ValueError,OSError,KeyError) as exc:
            messagebox.showerror('Cannot continue saved HPT',str(exc),parent=self)

    def _update_resume_button(self):
        if not self.folder or self.busy:
            self.resume_btn.configure(state='disabled')
            return
        supported,_=source_resume_status(self.folder)
        supported=supported and (self.current_engine!='ray' or self.ray_available)
        self.resume_btn.configure(state='normal' if supported else 'disabled',
            text='Resume Ray trials' if self.current_engine=='ray' else 'Resume / extend')

    def settings(self):
        space={}
        for key,(enabled,lo,hi) in self.space_widgets.items():
            if enabled.get():
                space[key]=(lo.get().replace(',','.'),hi.get().replace(',','.'))
        return validate_settings({'data':self.dataset.get().strip().strip('"'),
            'model':self.model.get(),'output_dir':self.output.get().strip() or None,
            'name':self.name.get(),'iterations':self.iterations.get(),
            'epochs':self.epochs.get(),'batch':self.batch.get(),'imgsz':self.imgsz.get(),
            'workers':self.workers.get(),'cache':self.cache.get(),'optimizer':self.optimizer.get(),
            'device':self.device.get(),'space':space if space else None,
            'engine':self.engine.get(),'grace_period':self.grace.get()})

    def _mode(self,setup):
        self.setup.pack_forget()
        self.monitor.pack_forget()
        self.in_setup=setup
        if setup:self.setup.pack(fill='both',expand=True)
        else:self.monitor.pack(fill='both',expand=True)
        self.mode_btn.configure(text='Show monitor' if setup else 'Show settings')
        self.mode_btn.pack(side='right',padx=(0,8))

    def toggle_mode(self):
        self._mode(not self.in_setup)

    def toggle_log(self):
        self._log_visible=not self._log_visible
        if self._log_visible:
            self._log.pack(fill='x',pady=(5,0))
        else:
            self._log.pack_forget()
        self.log_btn.configure(text='Hide log' if self._log_visible else 'Show log')

    def _logline(self,line):
        line=str(line)
        self._log.configure(state='normal')
        self._log.insert('end',line[:400]+'\n')
        if int(self._log.index('end-1c').split('.')[0])>250:
            self._log.delete('1.0','30.0')
        self._log.see('end')
        self._log.configure(state='disabled')
        if self.on_log:self.on_log(line)

    def start(self):
        if self.busy or self.installing:return
        if self.can_start and not self.can_start():
            messagebox.showwarning('Another job active','Complete your current Studio training or comparison job first.',parent=self)
            return
        try:
            settings=self.settings()
            root=Path(settings['output_dir'])
            name=settings['name'] or create_run_name()
            folder=root/name
            if folder.exists():
                raise ValueError('Run folder already exists. Choose a new run name or load the previous run.')
            self.logs_dir.mkdir(parents=True,exist_ok=True)
            self._start_process(settings,folder)
        except (ValueError,TypeError,OSError) as exc:
            messagebox.showerror('Invalid tuning settings',str(exc),parent=self)

    def _start_process(self,settings,folder):
        settings=dict(settings,name=folder.name)
        stamp=folder.name
        if settings.get('resume'):
            from datetime import datetime
            stamp += '_resume_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        file=self.logs_dir/('tuning_'+stamp+'.log')
        config=self.logs_dir/('tuning_'+stamp+'.json')
        config.write_text(json.dumps(settings,indent=2),encoding='utf-8')
        env=self.process_env()
        env.setdefault('KMP_DUPLICATE_LIB_OK','TRUE')
        flags=(subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP) if os.name=='nt' else 0
        with file.open('w',encoding='utf-8') as out:
            p=subprocess.Popen([str(self.env_python()),'-u',str(Path(__file__).with_name('run_hyperparameter_tuning.py')),
                                str(config)],cwd=str(Path(__file__).parent),env=env,
                               stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,
                               creationflags=flags,start_new_session=os.name!='nt')
        self.folder=folder
        self.job={'process':p,'file':file,'offset':0,'pending':'','target':settings['iterations'],
                  'epochs':settings['epochs'],'stopping':False}
        self.trials=[]
        self._last_ndjson_size=None
        self.charts.update_trials([])
        self.progress.configure(value=0)
        self.epoch_progress.configure(value=0)
        self.epoch_status.set('Current trial: preparing')
        self._refresh_trials()
        self.stop_btn.configure(state='normal')
        self.resume_btn.configure(state='disabled')
        self.start_btn.configure(state='disabled')
        self.current_engine=settings.get('engine', 'native')
        self.status.set('Tuning running · '+('Ray Optuna TPE + ASHA' if self.current_engine=='ray' else 'native Ultralytics evolution'))
        self.headline.set('Tuning · loading dataset and starting trial 1')
        self._logline(('Bayesian TPE + ASHA' if self.current_engine=='ray' else 'Native evolution')+' tuning launched · '+str(folder))
        self._mode(False)

    def stop(self):
        if not self.busy:return
        if not messagebox.askyesno('Stop hyperparameter tuning?',
             'Stop the tuner and its current worker? Completed trial records and checkpoints remain on disk.',
             parent=self):return
        self.job['stopping']=True
        p=self.job['process']
        try:
            if os.name=='nt':
                subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True,timeout=25,
                               creationflags=subprocess.CREATE_NO_WINDOW,check=False)
            else:os.killpg(p.pid,signal.SIGTERM)
        except (OSError,subprocess.TimeoutExpired,ProcessLookupError) as exc:
            self._logline('Stop error: '+str(exc))
        self.status.set('Stopping tuning; completed trials remain saved')
        self.stop_btn.configure(state='disabled')

    def _refresh_trials(self):
        if not self.folder:return
        if self.current_engine == 'ray':
            # Ray appends per-trial epoch records rather than native NDJSON.
            paths=list(self.folder.glob('**/result.json')) or list(self.folder.glob('**/progress.csv'))
            size=(len(paths), sum(p.stat().st_size for p in paths if p.is_file()))
        else:
            ndjson=self.folder/'tune_results.ndjson'
            try:size=ndjson.stat().st_size
            except OSError:size=-1
        if size==self._last_ndjson_size:return
        self._last_ndjson_size=size
        self.trials=(read_ray_trials(self.folder) if self.current_engine=='ray' else read_trials(self.folder))
        self.charts.update_trials(self.trials)
        self.table.delete(*self.table.get_children())
        for r in self.trials:
            m=r['metrics']
            def metric(k):
                value=m.get(k)
                try:return f'{float(value):.4f}'
                except (ValueError,TypeError):return '—'
            hp=', '.join(f'{k}={v:.4g}' for k,v in list(r['hyperparameters'].items())[:6]
                         if isinstance(v,(int,float)))
            self.table.insert('','end',iid=str(r['iteration']),values=(
                r['iteration'],f"{r['fitness']:.5f}" if not r['failed'] else 'Failed',
                metric('metrics/mAP50(B)'),metric('metrics/mAP50-95(B)'),
                metric('metrics/precision(B)'),metric('metrics/recall(B)'),
                r.get('epochs','—') if self.current_engine=='ray' else '—',
                r.get('status','—') if self.current_engine=='ray' else 'Finished',hp))
        best=best_trial(self.trials)
        count=len(self.trials)
        target=self.job['target'] if self.job else max(count,1)
        self.progress.configure(value=min(100,100*count/max(1,target)))
        word='started/recorded' if self.current_engine=='ray' else 'completed'
        self.trial_status.set(f'{count} / {target} trials {word} · {100*count/max(1,target):.0f}%')
        if self.current_engine=='ray' and self.trials:
            row=self.trials[-1]
            ep=int(row.get('epochs',0))
            epochs=int(self.job.get('epochs',0)) if self.job else int(self.epochs.get() or 1)
            if epochs>0:
                self.epoch_progress.configure(value=100*min(ep,epochs)/epochs)
                self.epoch_status.set(f"Ray trial {row['trial_id']} · epoch {ep}/{epochs}")
        if best:
            self.best_label.set(f"BEST  ·  Trial {best['iteration']}  ·  Fitness {best['fitness']:.5f}")
            metrics=best['metrics']
            if 'metrics/mAP50-95(B)' in metrics:
                self.best_label.set(self.best_label.get()+f"  ·  mAP50–95 {float(metrics['metrics/mAP50-95(B)']):.4f}")
            pairs=sorted(best['hyperparameters'].items())
            self.best_hp.set('   '.join(f'{k}={v:.4g}' for k,v in pairs if isinstance(v,(int,float))))
        elif count:
            self.best_label.set('No successful trials yet · check worker log')
        self.headline.set(f'Completed {count} tuning trials · {sum(not r["failed"] for r in self.trials)} valid')

    def _table_select(self,_event=None):
        sel=self.table.selection()
        if not sel:return
        match=next((r for r in self.trials if str(r['iteration'])==sel[0]),None)
        if match:
            self.detail.set('Trial '+sel[0]+' · '+ '   '.join(
                f'{k}={v:.5g}' for k,v in sorted(match['hyperparameters'].items()) if isinstance(v,(int,float))))

    def poll(self):
        job=self.job
        if not job:return
        try:
            self._refresh_trials()
            with job['file'].open('r',encoding='utf-8',errors='replace') as stream:
                stream.seek(job['offset'])
                text=stream.read(100000)
                job['offset']=stream.tell()
            if text:
                lines=(job['pending']+text).replace('\r','\n').split('\n')
                job['pending']=lines.pop()
                for line in lines:
                    line=re.sub(r'\x1b\[[0-9;]*[A-Za-z]','',line).strip()
                    if not line:continue
                    if found:=re.search(r'Starting iteration (\d+)/(\d+)',line):
                        self.epoch_progress.configure(value=0)
                        self.epoch_status.set(f'Current trial {found[1]}/{found[2]} · loading model / dataset')
                        self.status.set(f'Native tuner · trial {found[1]} of {found[2]} in progress')
                        self.trial_status.set(f'Trial {found[1]}/{found[2]} · training and validation')
                    elif (epoch:=re.search(r'(?<!\d)(\d+)/(\d+)\s+[\d.]+G',line)):
                        step,total=int(epoch[1]),int(epoch[2])
                        if 0<total<=1000 and step<=total:
                            self.epoch_progress.configure(value=100*step/total)
                            self.epoch_status.set(f'Current trial · epoch {step}/{total} · {100*step/total:.0f}%')
                    elif line.startswith('STUDIO_TUNE_') or 'training failure for hyperparameter tuning' in line:
                        self._logline(line)
                    elif ('Error:' in line or 'Traceback' in line or 'CUDA out of memory' in line):
                        self._logline(line)
            code=job['process'].poll()
            if code is not None and job['file'].stat().st_size<=job['offset']:
                if job['pending']:
                    self._logline(job['pending'][:300])
                self._refresh_trials()
                # Forced Windows termination bypasses the worker's finally block.
                if job['stopping'] and self.folder:
                    manifest=self.folder.parent/(self.folder.name+'_disk_cache_manifest.json')
                    if manifest.is_file():
                        try:
                            from train_and_split_tool_GUI import cleanup_disk_cache_manifest
                            removed, errors=cleanup_disk_cache_manifest(manifest)
                            self._logline(f'Post-stop disk cache cleanup: {removed} images; {len(errors)} errors')
                        except (OSError, ValueError) as exc:
                            self._logline('WARNING: Could not complete disk cleanup: '+str(exc))
                if job['stopping']:
                    self.status.set('Tuning stopped · completed trials preserved')
                elif code==0 and best_trial(self.trials):
                    self.status.set('Tuning complete · best hyperparameters ready')
                    self.progress.configure(value=100)
                else:
                    self.status.set(f'Tuning failed (exit {code}) · inspect log')
                    if not self._log_visible:self.toggle_log()
                self.job=None
                self.start_btn.configure(state='normal')
                self.stop_btn.configure(state='disabled')
                self._update_resume_button()
                self._logline('Tuning log: '+str(job['file']))
        except OSError as exc:
            self._logline('Tuning log read error: '+str(exc))

    def load_folder(self,folder):
        folder=Path(folder)
        if not folder.is_dir():
            raise ValueError('Selected tuning folder does not exist.')
        saved=self._settings_for_folder(folder)
        engine=(saved or {}).get('engine','native')
        if engine=='ray' or any(folder.glob('**/result.json')) or any(folder.glob('**/progress.csv')):
            self.current_engine='ray'
        elif (folder/'tune_results.ndjson').is_file():
            self.current_engine='native'
        else:
            raise ValueError('No Ultralytics native or Ray Tune trial results found.')
        self.folder=folder
        self._last_ndjson_size=None
        self._refresh_trials()
        self.status.set(f'Previous tuning run loaded · {folder.name}')
        self._update_resume_button()
        self._mode(False)

    def _saved_run_settings(self):
        if not self.folder:return None
        return self._settings_for_folder(self.folder)

    @staticmethod
    def _settings_for_folder(folder):
        candidates=(folder.parent/(folder.name+'.studio_tuning_settings.json'),
                    folder/'studio_tuning_settings.json')
        for path in candidates:
            if path.is_file():
                try:
                    result=json.loads(path.read_text(encoding='utf-8'))
                    if isinstance(result,dict):return result
                except (OSError, ValueError):
                    pass
        return None

    def resume_previous(self):
        if self.busy:return
        if self.can_start and not self.can_start():
            messagebox.showwarning('GPU job active','Finish the active GPU job first.',parent=self)
            return
        saved=self._saved_run_settings()
        if not saved or not self.folder:
            messagebox.showinfo('No saved configuration',
                'Continue from the ORIGINAL HPT run folder. A best YAML is not enough to restore search history.',parent=self)
            return
        try:
            ok,detail=source_resume_status(self.folder)
            if not ok:raise ValueError(detail)
            if self.current_engine=='ray':
                if not self.ray_available:
                    raise ValueError('Bayesian mode is not available in the selected Conda environment.')
                # Ray restores only unfinished trials; its existing budget and
                # searcher state cannot be extended from this GUI.
                target=int(saved['iterations'])
            else:
                target=int(self.iterations.get())
                done=len(self.trials)
                if target<=done:
                    raise ValueError(f'Enter a total trial count greater than {done} in Show settings.')
            config=validate_settings({**saved,'resume':True,'iterations':target,
                                      'name':self.folder.name,'output_dir':str(self.folder.parent)})
            self._start_process(config,self.folder)
        except (TypeError,OSError,ValueError) as exc:
            messagebox.showerror('Cannot resume tuning',str(exc),parent=self)

    def open_previous(self):
        if self.busy:
            messagebox.showinfo('Tuning is active','Finish or stop this run before opening another.',parent=self)
            return
        path=filedialog.askdirectory(parent=self,title='Choose native Ultralytics tuning folder')
        if path:
            try:self.load_folder(path)
            except (ValueError,OSError) as exc:
                messagebox.showerror('Could not load tuning results',str(exc),parent=self)

    def open_results(self):
        if not self.folder or not self.folder.is_dir():return
        try:
            if os.name=='nt':os.startfile(str(self.folder))
            elif os.sys.platform=='darwin':subprocess.Popen(['open',str(self.folder)])
            else:subprocess.Popen(['xdg-open',str(self.folder)])
        except OSError as exc:messagebox.showerror('Could not open folder',str(exc),parent=self)

    def copy_best_yaml(self):
        if not self.folder:return
        path=self.folder/'best_hyperparameters.yaml'
        if not path.is_file():
            messagebox.showinfo('Not available yet','Best YAML is written by Ultralytics after a successful trial.',parent=self)
            return
        self.clipboard_clear()
        self.clipboard_append(str(path))
        self._logline('Copied: '+str(path))

    def export_best_yaml(self):
        """Copy Ultralytics' native tuning output into a reusable YAML file."""
        import shutil
        if not self.folder:
            return
        source = self.folder / 'best_hyperparameters.yaml'
        if not source.is_file():
            messagebox.showinfo('Not available yet',
                                'Best hyperparameters are saved after a successful trial.', parent=self)
            return
        destination = filedialog.asksaveasfilename(
            parent=self, title='Export best hyperparameters for training',
            defaultextension='.yaml', initialfile='best_hyperparameters.yaml',
            filetypes=[('YAML configuration', '*.yaml *.yml')])
        if destination:
            try:
                if Path(destination).resolve() != source.resolve():
                    shutil.copyfile(source, destination)
                self._logline('Exported best hyperparameters: ' + str(destination))
            except OSError as exc:
                messagebox.showerror('Export failed', str(exc), parent=self)

    def copy_best_checkpoint(self):
        if not self.folder:return
        path=self.folder/'weights'/'best.pt'
        if self.current_engine=='ray' and (self.folder/'studio_ray_best.json').is_file():
            record=json.loads((self.folder/'studio_ray_best.json').read_text(encoding='utf-8'))
            ray_path=Path(record.get('trial_path',''))
            if ray_path.is_dir():
                candidate=next(ray_path.glob('**/weights/best.pt'),None)
                if candidate is not None:path=candidate
        if not path.is_file():
            messagebox.showinfo('Not available yet',
                'Ultralytics saves the best checkpoint in weights/best.pt after a successful trial.',parent=self)
            return
        self.clipboard_clear()
        self.clipboard_append(str(path))
        self._logline('Copied: '+str(path))
