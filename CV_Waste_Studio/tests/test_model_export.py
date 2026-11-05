"""Regression tests for native model export and platform/dependency policy."""
from pathlib import Path
import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_export import (probe_export_environment, parse_image_size,
                          validate_export_settings, build_export_kwargs, run_export_job)


class ModelExportValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.weights = self.root / 'best.pt'
        self.weights.write_bytes(b'Fake weights for tests')
        self.cap = {'formats': {
            'onnx': {'available': True, 'argument': 'onnx'},
            'engine': {'available': True, 'argument': 'engine'},
            'litert': {'available': True, 'argument': 'litert'},
        }}
        self.opts = {'model': str(self.weights), 'output': str(self.root / 'exports'),
                     'imgsz': '1088x640', 'batch': '2', 'precision': 'fp32', 'format': 'onnx'}

    def test_rectangular_and_square_imgsz(self):
        self.assertEqual(parse_image_size('640x1088'), (640,1088))
        self.assertEqual(parse_image_size('(1088, 640)'), (1088,640))
        self.assertEqual(parse_image_size(960), 960)
        for invalid in ('640 x q', '5', '1x1', '640x960x1024'):
            with self.assertRaises(ValueError):
                parse_image_size(invalid)

    def test_format_specific_arguments(self):
        opts = validate_export_settings(self.opts,self.cap)
        self.assertEqual(opts['imgsz'], (1088, 640))
        kwargs = build_export_kwargs(opts)
        self.assertEqual(kwargs['format'],'onnx')
        self.assertNotIn('workspace', kwargs)
        self.assertNotIn('data', kwargs)
        opts = validate_export_settings({**self.opts, 'format':'engine', 'precision':'fp16', 'workspace':'4'}, self.cap)
        self.assertEqual(opts['device'],'0')
        kwargs=build_export_kwargs(opts, supports_quantize=True)
        self.assertEqual(kwargs['quantize'], 16)
        self.assertEqual(kwargs['workspace'],4.0)
        self.assertNotIn('half',kwargs)
        self.assertTrue(build_export_kwargs(opts, supports_quantize=False)['half'])

    def test_litert_uses_cpu_and_legacy_tflite_alias(self):
        cap = dict(self.cap)
        cap['formats'] = dict(self.cap['formats'])
        cap['formats']['litert']={'available':True, 'argument':'tflite'}
        opts=validate_export_settings({**self.opts, 'format':'litert'},cap)
        self.assertEqual(build_export_kwargs(opts)['format'],'tflite')
        self.assertEqual(build_export_kwargs(opts)['device'],'cpu')
        self.assertNotIn('dynamic',build_export_kwargs(opts))
        for bad in ({'precision':'fp16'}, {'dynamic':True}):
            with self.assertRaises(ValueError):
                validate_export_settings({**self.opts, 'format':'litert', **bad},cap)

    def test_disabled_format_and_missing_calibration_rejected(self):
        cap=dict(self.cap)
        cap['formats']=dict(self.cap['formats'])
        cap['formats']['engine']={'available':False,'reason':'TensorRT missing'}
        with self.assertRaisesRegex(ValueError,'TensorRT missing'):
            validate_export_settings({**self.opts, 'format':'engine'},cap)
        with self.assertRaisesRegex(ValueError,'calibration data.yaml'):
            validate_export_settings({**self.opts, 'precision':'int8'},self.cap)

    def test_int8_uses_real_calibration_dataset(self):
        calib=self.root/'data.yaml'
        calib.write_text('train: test\nval: test\n')
        opts=validate_export_settings({**self.opts,'precision':'int8','data':str(calib),'fraction':'0.5'},self.cap)
        kw=build_export_kwargs(opts)
        self.assertEqual(kw['data'],str(calib))
        self.assertEqual(kw['fraction'],.5)

    def test_native_export_preserves_original_and_captures_folder(self):
        exported=[]
        class FakeYOLO:
            def __init__(self,path):
                exported.append(('load',Path(path)))
                self.path=Path(path)
            def export(self,**kwargs):
                exported.append(('export',kwargs))
                out=self.path.with_suffix('.onnx')
                out.write_bytes(b'onnx-output')
                return str(out)
        result=run_export_job(self.opts, FakeYOLO)
        folder=Path(result['output_directory'])
        self.assertEqual(self.weights.read_bytes(), b'Fake weights for tests')
        self.assertEqual(Path(result['exported_model']).read_bytes(), b'onnx-output')
        self.assertTrue((folder/'export_summary.json').exists())
        self.assertFalse((folder/'best.pt').exists())
        self.assertEqual(exported[-1][1]['imgsz'],(1088,640))
        self.assertEqual(exported[-1][1]['format'],'onnx')

    def test_failure_removes_private_copy_not_original(self):
        class FakeYOLO:
            def __init__(self,path):
                self.path=Path(path)
            def export(self,**kwargs):
                raise RuntimeError('simulated export failure')
        with self.assertRaisesRegex(RuntimeError,'simulated'):
            run_export_job(self.opts, FakeYOLO)
        self.assertTrue(self.weights.exists())
        self.assertEqual(list((self.root/'exports').rglob('*.pt')),[])


class ModelExportCompatibilityTests(unittest.TestCase):
    @staticmethod
    def fake_dependencies(formats=('onnx','engine','litert'), cuda=True, with_trt=True):
        ul=types.ModuleType('ultralytics');ul.__version__='8.4.test'
        engine=types.ModuleType('ultralytics.engine')
        exporter=types.ModuleType('ultralytics.engine.exporter')
        exporter.export_formats=lambda: {'Argument': list(formats)}
        torch=types.ModuleType('torch')
        torch.version=types.SimpleNamespace(cuda='13.0')
        torch.cuda=types.SimpleNamespace(is_available=lambda: cuda, get_device_name=lambda i:'Mock NVIDIA GPU')
        trt=types.ModuleType('tensorrt')
        trt.__version__='10.2'
        if with_trt:
            trt.Builder=lambda logger: object()
        return {'ultralytics':ul,'ultralytics.engine':engine,
                'ultralytics.engine.exporter':exporter,'torch':torch,'tensorrt':trt}

    def test_linux_litert_and_trt_ready(self):
        with patch.dict(sys.modules,self.fake_dependencies()):
            cap=probe_export_environment('linux','x86_64')
        self.assertTrue(all(cap['formats'][k]['available'] for k in ('onnx','engine','litert')))
        self.assertEqual(cap['install_package'],'tensorrt-cu13')

    def test_windows_disables_litert(self):
        with patch.dict(sys.modules,self.fake_dependencies()):
            cap=probe_export_environment('win32','AMD64')
        self.assertFalse(cap['formats']['litert']['available'])
        self.assertTrue(cap['formats']['engine']['available'])

    def test_missing_builder_disables_engine_not_onnx(self):
        with patch.dict(sys.modules,self.fake_dependencies(with_trt=False)):
            cap=probe_export_environment('linux','x86_64')
        self.assertFalse(cap['formats']['engine']['available'])
        self.assertTrue(cap['formats']['onnx']['available'])
        self.assertIn('builder',cap['formats']['engine']['reason'].lower())

    def test_legacy_tflite_fallback(self):
        with patch.dict(sys.modules,self.fake_dependencies(formats=('onnx','engine','tflite'))):
            cap=probe_export_environment('linux','x86_64')
        self.assertEqual(cap['formats']['litert']['argument'],'tflite')

    def test_no_cuda_disables_engine_even_with_tensorrt(self):
        with patch.dict(sys.modules,self.fake_dependencies(cuda=False)):
            cap=probe_export_environment('linux','x86_64')
        self.assertFalse(cap['formats']['engine']['available'])
        self.assertTrue(cap['formats']['litert']['available'])

if __name__=='__main__': unittest.main()
