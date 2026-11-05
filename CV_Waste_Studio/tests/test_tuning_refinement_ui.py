"""GUI smoke tests for new-refinement vs actual continuation."""
import json
import os
from pathlib import Path
import sys
import tempfile
import tkinter as tk
import unittest
from unittest.mock import patch

from hyperparameter_tuning_ui import HyperparameterTuningPanel


class RefinementPanelTests(unittest.TestCase):
    def setUp(self):
        self.root=tk.Tk();self.root.withdraw()
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name)
        self.run=self.base/'HPT_OLD';self.run.mkdir()
        self.dataset=self.base/'data.yaml'
        self.dataset.write_text('train: images/train\nval: images/val\nnc: 2\nnames: [a,b]\n')
        self.cfg={'data':str(self.dataset),'model':'yolo11s.pt',
                  'epochs':15,'iterations':10,'imgsz':960,'batch':8,
                  'workers':8,'cache':'off','optimizer':'AdamW',
                  'device':'cpu','engine':'native','grace_period':8,
                  'space':{'lr0':[.0001,.004],'momentum':[.7,.98]},
                  'name':'HPT_OLD','output_dir':str(self.base)}
        (self.base/'HPT_OLD.studio_tuning_settings.json').write_text(json.dumps(self.cfg))
        (self.run/'best_hyperparameters.yaml').write_text('lr0: .00051\nmomentum: .7\n')
        (self.run/'tune_results.ndjson').write_text(json.dumps({
            'iteration':1,'fitness':.8,'metrics':{'metrics/mAP50-95(B)':.8},
            'hyperparameters':{'lr0':.00051,'momentum':.7}})+'\n')
        with patch.object(HyperparameterTuningPanel,'check_ray_compatibility',lambda self:None):
            self.panel=HyperparameterTuningPanel(self.root,
                env_python=lambda:sys.executable,
                process_env=lambda:os.environ.copy(),logs_dir=self.base)

    def tearDown(self):
        self.root.destroy()

    def test_refine_into_new_name_with_boundary_warning(self):
        self.panel.search_width.set('20')
        with patch('hyperparameter_tuning_ui.messagebox.askyesno',return_value=True),\
             patch('hyperparameter_tuning_ui.messagebox.showinfo'):
            self.panel._apply_refinement_source(self.run)
        self.assertIn('refined',self.panel.name.get())
        self.assertNotEqual(self.panel.name.get(),'HPT_OLD')
        self.assertEqual(self.panel.dataset.get(),str(self.dataset))
        self.assertEqual(self.panel.imgsz.get(),'960')
        self.assertTrue(self.panel.boundary_labels['momentum'].get())
        self.assertEqual(self.panel.engine.get(),'native')
        self.assertEqual(self.panel.settings()['resume'],False)

    def test_continue_from_folder_restores_original_and_increases_budget(self):
        with patch('hyperparameter_tuning_ui.filedialog.askdirectory',return_value=str(self.run)),\
             patch('hyperparameter_tuning_ui.messagebox.showinfo'):
            self.panel.continue_from_folder()
        self.assertEqual(self.panel.name.get(),'HPT_OLD')
        self.assertEqual(self.panel.iterations.get(),'10')
        self.assertEqual(self.panel.imgsz.get(),'960')
        self.assertNotIn('disabled',self.panel.resume_btn.state())
        calls=[]
        with patch.object(self.panel,'_start_process',lambda cfg,folder:calls.append((cfg,folder))):
            self.panel.resume_previous()
        self.assertTrue(calls[0][0]['resume'])
        self.assertEqual(calls[0][1],self.run)

if __name__=='__main__':unittest.main()
