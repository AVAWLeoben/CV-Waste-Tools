"""GUI tests for explicitly selected external export environments."""
import os
from pathlib import Path
import sys
import tempfile
import tkinter as tk
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_export_ui import ModelExportPanel


class ExportEnvGuiTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root=tk.Tk(); self.root.withdraw()
        except tk.TclError as exc:
            self.skipTest(str(exc))
        self.addCleanup(self.root.destroy)
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.envroot=Path(self.tmp.name)
        self.managed=self.envroot/'managed/bin/python'
        self.target=self.envroot/'sensortc/bin/python'
        for p in (self.managed, self.target):
            p.parent.mkdir(parents=True);p.write_text('')
        self.mock_discovery=patch('model_export_ui.discover_export_environments',
            return_value=[{'label':'Studio managed','python':str(self.managed)},
                          {'label':'SenSoRTC · verified','python':str(self.target)}])
        self.mock_saved=patch('model_export_ui.load_selection',return_value='')
        with self.mock_discovery, self.mock_saved, \
             patch.object(ModelExportPanel, 'check_compatibility', lambda s:None):
            self.panel=ModelExportPanel(self.root,env_python=lambda:str(self.managed),
                              process_env=lambda:{'PYTHONPATH':'/bad','PATH':'/usr/bin'},
                              logs_dir=self.tmp.name)
        self.panel.pack(fill='both',expand=True)

    def test_selected_external_environment_used_and_saved(self):
        label=next(name for name in self.panel._env_options if 'SenSoRTC' in name)
        self.panel.export_env_choice.set(label)
        with patch('model_export_ui.save_selection') as save, \
             patch.object(self.panel,'check_compatibility') as probe:
            self.panel._environment_changed()
        self.assertEqual(self.panel._export_python(),str(self.target))
        self.assertNotIn('PYTHONPATH',self.panel._export_env())
        self.assertEqual(self.panel._export_env()['YOLO_AUTOINSTALL'],'False')
        save.assert_called_once_with(str(self.target))
        probe.assert_called_once()

    def test_no_tensor_rt_pip_prompt_for_external_jetson_environment(self):
        label=next(name for name in self.panel._env_options if 'SenSoRTC' in name)
        self.panel.export_env_choice.set(label)
        cap={'platform':'linux','jetson':True,'ultralytics':'8.4.174',
             'cuda':True,'gpu':'Jetson Thor','errors':{'tensorrt':'missing'},
             'formats':{'onnx':{'available':True,'reason':'Ready'},
                        'engine':{'available':False,'reason':'JetPack bindings unavailable'},
                        'litert':{'available':False,'reason':'ARM64 unsupported'}}}
        with patch.object(self.panel,'_offer_tensor_rt') as offer:
            self.panel._apply_probe(cap)
            self.panel.on_show()
            self.root.update_idletasks()
            offer.assert_not_called()
        self.assertIn('disabled',self.panel.install_btn.state())

if __name__=='__main__':unittest.main()
