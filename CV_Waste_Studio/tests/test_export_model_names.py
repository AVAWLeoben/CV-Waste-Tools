"""Regression tests: explicit export filenames for ONNX, TensorRT and WSL LiteRT."""
from pathlib import Path
import json
import os
import sys
import tempfile
import types
import tkinter as tk
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_export import (normalize_export_model_name, requested_export_filename,
                          validate_export_settings, run_export_job)


class ModelNameValidationTests(unittest.TestCase):
    def test_blank_retains_ultralytics_default(self):
        for fmt in ('onnx', 'engine', 'litert'):
            self.assertEqual(normalize_export_model_name('', fmt), '')
            self.assertEqual(requested_export_filename('', fmt), '')

    def test_format_specific_suffixes_and_stems(self):
        for fmt, suffix in [('onnx', '.onnx'), ('engine', '.engine'), ('litert', '.tflite')]:
            self.assertEqual(normalize_export_model_name('Yolo11L_1280_fp16', fmt),
                             'Yolo11L_1280_fp16')
            self.assertEqual(normalize_export_model_name('Yolo11L_1280_fp16' + suffix, fmt),
                             'Yolo11L_1280_fp16')
            self.assertEqual(requested_export_filename('Yolo11L_1280_fp16', fmt),
                             'Yolo11L_1280_fp16' + suffix)
            self.assertEqual(requested_export_filename('model.v2', fmt), 'model.v2' + suffix)

    def test_reject_invalid_cross_platform_names(self):
        for name in ('../outside', r'C:\temp\evil', 'a/b', 'a\\b', 'x:y', 'bad*name',
                     'a?b', 'name.', 'CON', 'NUL.txt', 'LPT9', '.hidden', 'A' * 250,
                     'name\nwithnewline', '<bad>', 'other.onnx'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                normalize_export_model_name(name, 'engine')
        with self.assertRaisesRegex(ValueError, r'\.onnx'):
            normalize_export_model_name('named.onnx', 'litert')

    def test_settings_include_normalized_stem(self):
        with tempfile.TemporaryDirectory() as td:
            pt = Path(td) / 'best.pt'
            pt.write_bytes(b'test')
            for fmt, extension in [('onnx', '.onnx'), ('engine', '.engine')]:
                cfg = validate_export_settings({'model': str(pt), 'format': fmt,
                                                'model_name': 'WasteSort_1280' + extension})
                self.assertEqual(cfg['model_name'], 'WasteSort_1280')


class ModelNameNativeExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.checkpoint = self.dir / 'best.pt'
        self.checkpoint.write_bytes(b'checkpoint remains')

    def make_factory(self, ext, output_external=False):
        dest = self.dir / 'outside' / ('source' + ext)
        class FakeYOLO:
            def __init__(self, path):
                self.path = Path(path)
            def export(self, **kwargs):
                if output_external:
                    dest.parent.mkdir(exist_ok=True)
                    dest.write_bytes(b'exported-model')
                    return str(dest)
                out = self.path.with_suffix(ext)
                out.write_bytes(b'exported-model')
                return str(out)
        return FakeYOLO

    def test_native_custom_onnx_name(self):
        cfg = {'model': str(self.checkpoint), 'format': 'onnx', 'model_name': 'WasteModel_1280',
               'output': str(self.dir / 'exports')}
        summary = run_export_job(cfg, self.make_factory('.onnx'))
        artifact = Path(summary['exported_model'])
        self.assertEqual(artifact.name, 'WasteModel_1280.onnx')
        self.assertEqual(artifact.read_bytes(), b'exported-model')
        self.assertEqual(self.checkpoint.read_bytes(), b'checkpoint remains')
        self.assertEqual(list(Path(summary['output_directory']).glob('best.pt')), [])
        self.assertEqual(json.loads((artifact.parent / 'export_summary.json').read_text())['exported_model'], str(artifact))

    def test_external_path_is_copied_then_named(self):
        cfg = {'model': str(self.checkpoint), 'format': 'onnx', 'model_name': 'My.Named.Model.onnx',
               'output': str(self.dir / 'exports')}
        summary = run_export_job(cfg, self.make_factory('.onnx', True))
        self.assertEqual(Path(summary['exported_model']).name, 'My.Named.Model.onnx')
        self.assertEqual((self.dir / 'outside' / 'source.onnx').read_bytes(), b'exported-model')

    def test_blank_preserves_previous_name(self):
        cfg = {'model': str(self.checkpoint), 'format': 'onnx',
               'output': str(self.dir / 'exports')}
        summary = run_export_job(cfg, self.make_factory('.onnx'))
        self.assertEqual(Path(summary['exported_model']).name, 'best.onnx')

    def test_engine_verification_receives_renamed_path(self):
        cfg = {'model': str(self.checkpoint), 'format': 'engine',
               'model_name': 'SenSoRTC_Thor_1280_FP16',
               'output': str(self.dir / 'exports'), 'precision': 'fp16', 'imgsz': 1280}
        fake_ultralytics = types.ModuleType('ultralytics')
        fake_ultralytics.YOLO = self.make_factory('.engine')
        formats = {'formats': {'engine': {'available': True, 'argument': 'engine'}}}
        seen = []
        def verify(path, imgsz):
            seen.append((Path(path), imgsz))
            return {'verified': True, 'status': 'passed'}
        with patch.dict(sys.modules, {'ultralytics': fake_ultralytics}), \
             patch('model_export.probe_export_environment', return_value=formats), \
             patch('model_export.validate_engine_with_selected_runtime', side_effect=verify):
            summary = run_export_job(cfg)
        self.assertEqual(seen[0][0].name, 'SenSoRTC_Thor_1280_FP16.engine')
        self.assertTrue(seen[0][0].is_file())
        self.assertEqual(seen[0][1], 1280)
        self.assertEqual(summary['exported_model'], str(seen[0][0]))

    def test_existing_destination_never_overwritten(self):
        cfg = {'model': str(self.checkpoint), 'format': 'onnx',
               'model_name': 'Chosen_Name', 'output': str(self.dir / 'exports')}
        class FakeYOLO:
            def __init__(self, path):
                self.path = Path(path)
            def export(self, **kwargs):
                original = self.path.with_suffix('.onnx')
                original.write_bytes(b'result')
                (self.path.parent / 'Chosen_Name.onnx').write_bytes(b'keep')
                return str(original)
        with self.assertRaisesRegex(FileExistsError, 'already exists'):
            run_export_job(cfg, FakeYOLO)
        name = next((self.dir / 'exports').glob('*/Chosen_Name.onnx'))
        self.assertEqual(name.read_bytes(), b'keep')
        self.assertTrue(self.checkpoint.is_file())


class WslModelNamesTests(unittest.TestCase):
    def test_wsl_worker_renames_to_requested_name(self):
        from wsl_litert_export_worker import main
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            pt = root / 'best.pt'
            pt.write_bytes(b'original')
            req = root / 'request.json'
            req.write_text(json.dumps({'model': str(pt), 'output': str(root),
                            'format': 'litert', 'imgsz': 160, 'batch': 1,
                            'precision': 'fp32', 'model_name': 'Sorting_160.tflite'}))
            class FakeYOLO:
                def __init__(self, path): self.path = Path(path)
                def export(self, **kwargs):
                    path = self.path.with_suffix('.tflite')
                    path.write_bytes(b'lite output')
                    return str(path)
            fake_ultralytics = types.ModuleType('ultralytics')
            fake_cfg = types.ModuleType('ultralytics.cfg')
            fake_cfg.DEFAULT_CFG_DICT = {}
            with patch.dict(sys.modules, {'ultralytics': fake_ultralytics, 'ultralytics.cfg': fake_cfg}), \
                 patch.object(sys, 'argv', ['wsl_worker', str(req)]):
                summary = main(model_factory=FakeYOLO)
            self.assertEqual(summary['filename'], 'Sorting_160.tflite')
            self.assertEqual((root / 'Sorting_160.tflite').read_bytes(), b'lite output')
            self.assertEqual(pt.read_bytes(), b'original')

    def test_windows_bridge_passes_model_name(self):
        import wsl_litert
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            pt = root / 'best.pt'
            pt.write_bytes(b'original')
            fake_probe = lambda: {'available': True, 'argument': 'litert',
                                  'distribution': 'Ubuntu', 'distributions': ['Ubuntu']}
            class FakePopen:
                def __init__(self, *args, **kwargs): pass
                def wait(self):
                    folder = next((root / 'out').glob('export_*_litert'))
                    request = json.loads((folder / 'wsl_export_request.json').read_text())
                    assert request['model_name'] == 'Custom_160'
                    (folder / 'Custom_160.tflite').write_bytes(b'tflite')
                    (folder / 'wsl_export_result.json').write_text(json.dumps(
                        {'filename': 'Custom_160.tflite', 'ultralytics_arguments': {'format': 'litert'}}))
                    return 0
            with patch.object(wsl_litert, 'windows_to_wsl', return_value='/mnt/c/safe'):
                result = wsl_litert.run_wsl_litert_export(
                    {'format': 'litert', 'model': str(pt), 'imgsz': 160,
                     'output': str(root / 'out'), 'model_name': 'Custom_160'},
                    probe=fake_probe, popen=FakePopen)
            self.assertEqual(Path(result['exported_model']).name, 'Custom_160.tflite')
            self.assertEqual(pt.read_bytes(), b'original')


class GuiModelNameTests(unittest.TestCase):
    def test_gui_passes_model_name_to_export_configuration(self):
        from model_export_ui import ModelExportPanel
        try:
            root = tk.Tk()
            root.withdraw()
        except tk.TclError as exc:
            self.skipTest(str(exc))
        self.addCleanup(root.destroy)
        with tempfile.TemporaryDirectory() as d:
            with patch.object(ModelExportPanel, 'check_compatibility', lambda p: None):
                panel = ModelExportPanel(root, env_python=lambda: sys.executable,
                                         process_env=lambda: os.environ.copy(), logs_dir=d)
            cap = {'ultralytics': 'mock', 'cuda': True, 'gpu': 'Thor', 'formats': {
                'onnx': {'available': True, 'argument': 'onnx', 'reason': 'Ready'},
                'engine': {'available': True, 'argument': 'engine', 'reason': 'Ready'},
                'litert': {'available': False, 'reason': 'ARM64'}}}
            panel._apply_probe(cap)
            panel.fmt.set('engine')
            panel._format_changed()
            weights = Path(d) / 'best.pt'
            weights.write_bytes(b'weights')
            panel.model.set(str(weights))
            panel.model_name.set('Thor_1280_FP16.engine')
            self.assertEqual(panel._settings()['model_name'], 'Thor_1280_FP16')
            self.assertEqual(panel._settings()['format'], 'engine')
            panel.model_name.set('Wrong.onnx')
            with self.assertRaises(ValueError):
                panel._settings()


if __name__ == '__main__':
    unittest.main()
