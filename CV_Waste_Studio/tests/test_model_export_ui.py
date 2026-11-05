"""GUI checks for optional TensorRT and Linux-only LiteRT access."""
import os
from pathlib import Path
import sys
import tempfile
import tkinter as tk
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model_export_ui import ModelExportPanel


class ExportPanelTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root=tk.Tk();self.root.withdraw()
        except tk.TclError as exc:
            self.skipTest(f'Display unavailable: {exc}')
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        with patch.object(ModelExportPanel,'check_compatibility',lambda _:None):
            self.panel=ModelExportPanel(self.root,env_python=lambda:sys.executable,
                                        process_env=lambda:os.environ.copy(),logs_dir=self.tmp.name)
        self.panel.pack(fill='both',expand=True)

    def tearDown(self):
        if hasattr(self,'root'):
            self.root.destroy()

    def test_missing_engine_prompt_decline_and_format_disabled(self):
        report={'ultralytics':'test','cuda':True,'gpu':'RTX mock','install_package':'tensorrt-cu13',
                'errors':{'tensorrt':'missing'},'formats':{
                    'onnx':{'available':True,'argument':'onnx','reason':'Ready'},
                    'engine':{'available':False,'reason':'TensorRT missing'},
                    'litert':{'available':False,'reason':'Linux only'}}}
        with patch('model_export_ui.messagebox.askyesno',return_value=False) as box:
            self.panel._apply_probe(report)
            self.panel._offer_tensor_rt()
        self.assertNotIn('disabled',self.panel.radios['onnx'].state())
        self.assertIn('disabled',self.panel.radios['engine'].state())
        self.assertIn('disabled',self.panel.radios['litert'].state())
        self.assertNotIn('disabled',self.panel.install_btn.state())
        self.assertTrue(box.called)

    def test_wsl_ready_enables_litert_and_setup_button(self):
        report = {'platform': 'win32', 'ultralytics': '8.4.mock', 'cuda': False,
                  'wsl': {'available': True, 'phase': 'ready', 'distribution': 'Ubuntu',
                          'reason': 'Ready through WSL2'}, 'errors': {},
                  'formats': {
                    'onnx': {'available': True, 'argument': 'onnx', 'reason': 'Ready'},
                    'engine': {'available': False, 'reason': 'No CUDA'},
                    'litert': {'available': True, 'argument': 'litert', 'backend': 'wsl',
                               'reason': 'Ready through WSL2'}}}
        self.panel._apply_probe(report)
        self.assertNotIn('disabled', self.panel.radios['litert'].state())
        self.assertNotIn('disabled', self.panel.wsl_setup_btn.state())
        self.panel.fmt.set('litert')
        self.panel._format_changed()
        self.assertIn('WSL', self.panel.format_status['litert'].get())
        self.assertEqual(self.panel.device.get(), 'cpu')

    def test_wsl_missing_disables_litert_with_setup_button(self):
        report = {'platform': 'win32', 'ultralytics': '8.4.mock', 'cuda': False,
                  'wsl': {'available': False, 'phase': 'missing_wsl', 'reason': 'Install WSL'},
                  'errors': {}, 'formats': {
                    'onnx': {'available': True, 'argument': 'onnx', 'reason': 'Ready'},
                    'engine': {'available': False, 'reason': 'No CUDA'},
                    'litert': {'available': False, 'reason': 'Install WSL'}}}
        self.panel._apply_probe(report)
        self.assertIn('disabled', self.panel.radios['litert'].state())
        self.assertNotIn('disabled', self.panel.wsl_setup_btn.state())

    def test_no_automatic_wsl_install_offer_on_probe_error(self):
        report = {'platform': 'win32', 'ultralytics': '8.4.mock', 'cuda': False,
                  'wsl': {'available': False, 'phase': 'env_error', 'distribution': 'Ubuntu',
                          'reason': 'A previously working exporter failed a metadata check'},
                  'errors': {}, 'formats': {
                      'onnx': {'available': True, 'argument': 'onnx', 'reason': 'Ready'},
                      'engine': {'available': False, 'reason': 'No CUDA'},
                      'litert': {'available': False, 'reason': 'Probe error'}}}
        with patch.object(self.panel, '_offer_wsl_setup') as offer:
            self.panel._apply_probe(report)
            self.panel.on_show()
            self.root.update_idletasks()
            offer.assert_not_called()
        self.assertIn('env_error', self.panel.format_status['litert'].get())
        with patch('model_export_ui.messagebox.showwarning') as warning, \
             patch('model_export_ui.messagebox.askyesno') as install:
            self.panel.setup_wsl()
        warning.assert_called_once()
        install.assert_not_called()

    def test_ready_format_and_enabled_export(self):
        report={'ultralytics':'test','cuda':True,'gpu':'RTX mock',
                'errors':{},'formats':{
                    'onnx':{'available':True,'argument':'onnx','reason':'Ready'},
                    'engine':{'available':True,'argument':'engine','reason':'Ready'},
                    'litert':{'available':True,'argument':'litert','reason':'Ready'}}}
        self.panel._apply_probe(report)
        self.assertNotIn('disabled',self.panel.radios['litert'].state())
        self.panel.fmt.set('litert');self.panel._format_changed()
        self.assertEqual(self.panel.device.get(),'cpu')
        self.assertEqual(tuple(self.panel.precision_combo.cget('values')),('fp32','int8'))
        self.assertIn('disabled',self.panel.dynamic_check.state())
        self.assertNotIn('disabled',self.panel.start_btn.state())

if __name__=='__main__':unittest.main()
