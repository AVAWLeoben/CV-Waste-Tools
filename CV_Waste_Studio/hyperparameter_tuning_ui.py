"""CV Waste Studio's native Ultralytics hyperparameter tuning workspace."""
from __future__ import annotations
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from hyperparameter_tuning import PRESETS, RANGES, best_trial, create_run_name, read_trials, validate_settings
from tuning_charts import TuningCharts

ACCENT = '#405DE6'


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
        self._last_ndjson_size=-1
        self._selected_trial=None
        self._build()

    @property
    def busy(self):
        return self.job is not None and self.job['process'].poll() is None

    def _row(self, parent, row, caption, var, browse=None, width=16):
        ttk.Label(parent,text=caption).grid(row=row,column=0,sticky='w',pady=4,padx=(0,14))
        ttk.Entry(parent,textvariable=var,width=width).grid(row=row,column=1,sticky='ew',pady=4)
        if browse:
            ttk.Button(parent,text='Browse…',command=browse).grid(row=row,column=2,sticky='w',padx=10)

    def _build(self):
        main=ttk.Frame(self,padding=(18,12))
        main.pack(fill='both',expand=True)
        heading=ttk.Frame(main)
        heading.pack(fill='x',pady=(0,8))
        ttk.Label(heading,text='Native YOLO hyperparameter tuning',font=('Segoe UI',14,'bold')).pack(side='left')
        ttk.Button(heading,text='Open previous run',command=self.open_previous).pack(side='right')
        self.resume_btn=ttk.Button(heading,text='Resume / extend',command=self.resume_previous,state='disabled')
        self.resume_btn.pack(side='right',padx=(0,8))
        self.stop_btn=ttk.Button(heading,text='Stop tuning',style='Danger.TButton',state='disabled',command=self.stop)
        self.stop_btn.pack(side='right',padx=(8,8))
        self.mode_btn=ttk.Button(heading,text='Show settings',command=self.toggle_mode)
        self.mode_btn.pack(side='right')
        self.mode_btn.pack_forget()
        self.status=tk.StringVar(value='Ready · select your chosen architecture and tune on train/val only')
        ttk.Label(main,textvariable=self.status,style='Muted.TLabel').pack(fill='x',pady=(0,9))
        self.setup=ttk.Frame(main)
        self.setup.pack(fill='both',expand=True)
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
                  'cache=off avoids .npy files and excessive shared RAM on Windows; disk cache is automatically cleaned after tuning.',
                  style='Muted.TLabel',wraplength=660).grid(row=14,column=1,columnspan=2,sticky='w',pady=(4,8))
        ttk.Separator(body).grid(row=15,column=0,columnspan=3,sticky='ew',pady=10)
        ttk.Label(body,text='03  Search space',font=('Segoe UI',12,'bold')).grid(row=16,column=0,columnspan=3,sticky='w',pady=(0,8))
        picks=ttk.Frame(body)
        picks.grid(row=17,column=0,columnspan=3,sticky='w',pady=5)
        ttk.Label(picks,text='Preset').pack(side='left')
        ttk.Combobox(picks,textvariable=self.preset,values=tuple(PRESETS),state='readonly',width=20).pack(
            side='left',padx=(10,12))
        ttk.Button(picks,text='Apply preset',command=self.apply_preset).pack(side='left')
        ttk.Label(body,text='Check parameters to tune; adjust bounds (must stay within native search limits). '
                  'Full native uses the installed Ultralytics default search space.',
                  style='Muted.TLabel',wraplength=725).grid(row=18,column=0,columnspan=3,sticky='w',pady=(0,5))
        table=ttk.Frame(body)
        table.grid(row=19,column=0,columnspan=3,sticky='w',pady=(0,12))
        ttk.Label(table,text='Tune',font=('Segoe UI',9,'bold')).grid(row=0,column=0,sticky='w',padx=6)
        ttk.Label(table,text='Parameter',font=('Segoe UI',9,'bold')).grid(row=0,column=1,sticky='w',padx=6)
        ttk.Label(table,text='Minimum',font=('Segoe UI',9,'bold')).grid(row=0,column=2,sticky='w',padx=6)
        ttk.Label(table,text='Maximum',font=('Segoe UI',9,'bold')).grid(row=0,column=3,sticky='w',padx=6)
        self.space_widgets={}
        for j,(key,(lo,hi)) in enumerate(RANGES.items(),1):
            enabled=tk.BooleanVar(value=key in PRESETS['Balanced'])
            low=tk.StringVar(value=f'{lo:g}')
            high=tk.StringVar(value=f'{hi:g}')
            ttk.Checkbutton(table,variable=enabled).grid(row=j,column=0,sticky='w',padx=6,pady=2)
            ttk.Label(table,text=key,width=19).grid(row=j,column=1,sticky='w',padx=6,pady=2)
            ttk.Entry(table,textvariable=low,width=15).grid(row=j,column=2,padx=6,pady=2)
            ttk.Entry(table,textvariable=high,width=15).grid(row=j,column=3,padx=6,pady=2)
            self.space_widgets[key]=(enabled,low,high)
        ttk.Separator(body).grid(row=20,column=0,columnspan=3,sticky='ew',pady=10)
        ttk.Label(body,text='04  Outputs',font=('Segoe UI',12,'bold')).grid(row=21,column=0,columnspan=3,sticky='w',pady=(0,8))
        self._row(body,22,'Tuning results root',self.output,
                  browse=lambda:self._folder(self.output))
        self._row(body,23,'Run name (optional)',self.name)
        ttk.Label(body,text='Each run gets its own folder with best_hyperparameters.yaml, '
                  'best.pt, native tune figures and full trial history.',style='Muted.TLabel',
                  wraplength=680).grid(row=24,column=1,columnspan=2,sticky='w',pady=(0,10))
        self.start_btn=ttk.Button(body,text='Start native hyperparameter tuning',style='Accent.TButton',command=self.start)
        self.start_btn.grid(row=25,column=2,sticky='e',pady=8)

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
        cols=('trial','fitness','map50','map5095','precision','recall','parameters')
        self.table=ttk.Treeview(lower,columns=cols,show='headings',height=4)
        titles={'trial':('Trial',52),'fitness':('Fitness',77),'map50':('mAP50',80),
                'map5095':('mAP50–95',87),'precision':('Precision',82),
                'recall':('Recall',82),'parameters':('Tuned hyperparameters',420)}
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

    def _file(self,var,title,filetypes):
        selected=filedialog.askopenfilename(parent=self,title=title,filetypes=filetypes)
        if selected:var.set(selected)

    def _folder(self,var):
        selected=filedialog.askdirectory(parent=self,title='Select tuning results directory')
        if selected:var.set(selected)

    def apply_preset(self):
        keys=set(PRESETS[self.preset.get()])
        for key,(enabled,_,_) in self.space_widgets.items():
            enabled.set(key in keys)

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
            'device':self.device.get(),'space':space if space else None})

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
        if self.busy:return
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
        self.job={'process':p,'file':file,'offset':0,'pending':'','target':settings['iterations'],'stopping':False}
        self.trials=[]
        self._last_ndjson_size=-1
        self.charts.update_trials([])
        self.progress.configure(value=0)
        self.epoch_progress.configure(value=0)
        self.epoch_status.set('Current trial: preparing')
        self._refresh_trials()
        self.stop_btn.configure(state='normal')
        self.resume_btn.configure(state='disabled')
        self.start_btn.configure(state='disabled')
        self.status.set('Tuning running · native Ultralytics trials')
        self.headline.set('Tuning · loading dataset and starting trial 1')
        self._logline('Native tuning launched · '+str(folder))
        self._mode(False)

    def stop(self):
        if not self.busy:return
        if not messagebox.askyesno('Stop hyperparameter tuning?',
             'Stop the native tuner and its current training worker? Completed trial records and best checkpoint remain on disk.',
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
        ndjson=self.folder/'tune_results.ndjson'
        try:size=ndjson.stat().st_size
        except OSError:size=-1
        if size==self._last_ndjson_size:return
        self._last_ndjson_size=size
        self.trials=read_trials(self.folder)
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
                metric('metrics/precision(B)'),metric('metrics/recall(B)'),hp))
        best=best_trial(self.trials)
        count=len(self.trials)
        target=self.job['target'] if self.job else max(count,1)
        self.progress.configure(value=min(100,100*count/max(1,target)))
        self.trial_status.set(f'{count} / {target} trials completed · {100*count/max(1,target):.0f}%')
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
                self.resume_btn.configure(state='normal' if self._saved_run_settings() else 'disabled')
                self._logline('Tuning log: '+str(job['file']))
        except OSError as exc:
            self._logline('Tuning log read error: '+str(exc))

    def load_folder(self,folder):
        folder=Path(folder)
        if not folder.is_dir() or not (folder/'tune_results.ndjson').is_file():
            raise ValueError('This folder has no native Ultralytics tune_results.ndjson')
        self.folder=folder
        self._last_ndjson_size=-1
        self._refresh_trials()
        self.status.set(f'Previous tuning run loaded · {folder.name}')
        self.resume_btn.configure(state='normal' if self._saved_run_settings() else 'disabled')
        self._mode(False)

    def _saved_run_settings(self):
        if not self.folder:return None
        candidates=(self.folder.parent/(self.folder.name+'.studio_tuning_settings.json'),
                    self.folder/'studio_tuning_settings.json')
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
            messagebox.showinfo('No saved configuration','Load a tuning run created by CV Waste Studio first.',parent=self)
            return
        try:
            target=int(self.iterations.get())
            done=len(self.trials)
            if target<=done:
                raise ValueError(f'Enter a trial count larger than {done} in Show settings to extend this run.')
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
        if not path.is_file():
            messagebox.showinfo('Not available yet',
                'Ultralytics saves the best checkpoint in weights/best.pt after a successful trial.',parent=self)
            return
        self.clipboard_clear()
        self.clipboard_append(str(path))
        self._logline('Copied: '+str(path))
