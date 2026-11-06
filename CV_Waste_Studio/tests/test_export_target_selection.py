"""Regression tests: selected format must agree with summary and export settings."""
import os
from pathlib import Path
import sys
import tempfile
import tkinter as tk
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_export_ui import ModelExportPanel
from model_export import build_export_kwargs


def capability_report(*, engine=True, onnx=True):
    return {
        'platform': 'linux', 'ultralytics': '8.4.174', 'cuda': True,
        'gpu': 'Jetson Thor', 'jetson': True, 'errors': {},
        'formats': {
            'onnx': {'available': onnx, 'reason': 'Ready' if onnx else 'No ONNX exporter', 'argument': 'onnx'},
            'engine': {'available': engine, 'reason': 'Ready' if engine else 'TensorRT version mismatch', 'argument': 'engine'},
            'litert': {'available': False, 'reason': 'Linux ARM64 unsupported'},
        },
    }


class SelectedFormatTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
            self.root.withdraw()
        except tk.TclError as exc:
            self.skipTest(f'No Tk display: {exc}')
        self.addCleanup(self.root.destroy)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        with patch.object(ModelExportPanel, 'check_compatibility', lambda self: None):
            self.panel = ModelExportPanel(
                self.root, env_python=lambda: sys.executable,
                process_env=lambda: os.environ.copy(), logs_dir=self.temp.name,
            )
        self.panel.pack(fill='both', expand=True)

    def test_status_updates_immediately_after_selecting_engine(self):
        self.panel._apply_probe(capability_report())
        self.assertIn('Target: ONNX (ready)', self.panel.status.get())
        self.panel.fmt.set('engine')
        self.panel._format_changed()
        self.assertIn('Target: TensorRT Engine (ready)', self.panel.status.get())
        self.assertNotIn('Target: ONNX', self.panel.status.get())
        self.assertNotIn('disabled', self.panel.start_btn.state())

    def test_unavailable_engine_is_not_silently_changed_to_onnx(self):
        self.panel.fmt.set('engine')
        self.panel._apply_probe(capability_report(engine=False))
        self.assertEqual('engine', self.panel.fmt.get())
        self.assertIn('TensorRT Engine', self.panel.status.get())
        self.assertIn('TensorRT version mismatch', self.panel.status.get())
        self.assertIn('disabled', self.panel.start_btn.state())
        self.assertIn('disabled', self.panel.radios['engine'].state())

    def test_selected_engine_becomes_exportable_after_environment_recheck(self):
        self.panel.fmt.set('engine')
        self.panel._apply_probe(capability_report(engine=False))
        self.panel._apply_probe(capability_report(engine=True))
        self.assertEqual('engine', self.panel.fmt.get())
        self.assertIn('Target: TensorRT Engine (ready)', self.panel.status.get())
        self.assertNotIn('disabled', self.panel.start_btn.state())

    def test_switching_back_to_onnx_updates_target(self):
        self.panel._apply_probe(capability_report())
        self.panel.fmt.set('engine')
        self.panel._format_changed()
        self.panel.fmt.set('onnx')
        self.panel._format_changed()
        self.assertIn('Target: ONNX (ready)', self.panel.status.get())

    def test_export_settings_and_native_kwargs_use_selected_engine(self):
        self.panel._apply_probe(capability_report())
        self.panel.fmt.set('engine')
        self.panel._format_changed()
        model = Path(self.temp.name) / 'best.pt'
        model.write_bytes(b'fake model path, only settings are checked')
        self.panel.model.set(str(model))
        self.panel.imgsz.set('1280')
        self.panel.precision.set('fp16')
        settings = self.panel._settings()
        self.assertEqual(settings['format'], 'engine')
        self.assertEqual(settings['ultralytics_format'], 'engine')
        export_kwargs = build_export_kwargs(settings, supports_quantize=False)
        self.assertEqual(export_kwargs['format'], 'engine')
        self.assertEqual(export_kwargs['imgsz'], 1280)
        self.assertTrue(export_kwargs['half'])


if __name__ == '__main__':
    unittest.main()
