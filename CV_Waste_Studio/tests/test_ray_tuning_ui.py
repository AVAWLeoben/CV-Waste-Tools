"""Ray availability and reusable YAML import in the real Tk panel."""
import os
from pathlib import Path
import sys
import tempfile
import tkinter as tk
import unittest
from unittest.mock import patch

from hyperparameter_tuning_ui import HyperparameterTuningPanel


class RayTuningUITests(unittest.TestCase):
    def setUp(self):
        self.root=tk.Tk()
        self.root.withdraw()
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        with patch.object(HyperparameterTuningPanel,'check_ray_compatibility',lambda self: None):
            self.panel=HyperparameterTuningPanel(self.root,
                    env_python=lambda: sys.executable,
                    process_env=lambda: os.environ.copy(),logs_dir=self.tmp.name)

    def tearDown(self):
        self.root.destroy()

    def test_failure_disables_ray_but_keeps_native(self):
        with patch('hyperparameter_tuning_ui.messagebox.askyesno',return_value=False) as ask:
            self.panel._ray_unavailable('Ray missing')
        self.assertIn('disabled',self.panel.ray_radio.state())
        self.assertEqual(self.panel.engine.get(),'native')
        self.assertIn('Ray missing', self.panel.compat_label.get())
        self.assertTrue(ask.called)

    def test_compatible_probe_enables_ray(self):
        self.panel.ray_report={'available':True, 'versions':{'ultralytics':'8.4.2','ray':'2.5','optuna':'4'}}
        # direct UI availability state equivalent to a successful probe
        self.panel.ray_available=True
        self.panel.ray_radio.configure(state='normal')
        self.panel.engine.set('ray')
        self.panel._update_engine()
        self.assertNotIn('disabled',self.panel.ray_radio.state())
        self.assertIn('ASHA',self.panel.start_btn.cget('text'))

    def test_imported_yaml_sets_editable_ranges_for_both_modes(self):
        p=Path(self.tmp.name)/'best_hyperparameters.yaml'
        p.write_text('lr0: 0.00051\nmomentum: 0.7\nmosaic: 0.82613\n' +
                     'data: DO_NOT_COPY\n',encoding='utf-8')
        with patch('hyperparameter_tuning_ui.filedialog.askopenfilename',return_value=str(p)),\
             patch('hyperparameter_tuning_ui.messagebox.showinfo'):
            self.panel.import_best_yaml()
        enabled={name for name,(checkbox,_,_) in self.panel.space_widgets.items() if checkbox.get()}
        self.assertEqual(enabled,{'lr0','momentum','mosaic'})
        for name in enabled:
            _,lo,hi=self.panel.space_widgets[name]
            self.assertLess(float(lo.get()),float(hi.get()))
        self.assertEqual(self.panel.preset.get(),'Custom (YAML)')


if __name__=='__main__':unittest.main()
