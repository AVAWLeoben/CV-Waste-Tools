"""Regression tests for Jetson/SenSoRTC export isolation and verification."""
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_environment import discover_export_environments, make_export_process_env
from model_export import (tensorrt_versions_match, is_jetson_environment,
                          validate_engine_with_selected_runtime, run_export_job)


class ExportEnvironmentSafetyTests(unittest.TestCase):
    def test_timestamped_sensortc_env_and_conda_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            for folder in (home / '.local/share/SenSoRTC/envs/install-20261106-123223-1db6793b',
                           home / 'miniconda3/envs/CV-Waste'):
                (folder / 'bin').mkdir(parents=True)
                (folder / 'bin/python').write_text('fake executable')
            managed = str(home / 'miniconda3/envs/CV-Waste/bin/python')
            names = discover_export_environments(managed, home,
                         [str(home / 'miniconda3/envs/CV-Waste')])
            self.assertTrue(any('SenSoRTC' in e['label'] and
                                '123223' in e['python'] for e in names))
            self.assertEqual(len({x['python'] for x in names}), len(names))

    def test_external_worker_not_contaminated_by_studio_conda(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            a = path / 'studio/bin/python'; b = path / 'sensortc/bin/python'
            for file in (a, b):
                file.parent.mkdir(parents=True)
                file.touch()
            env = make_export_process_env({'PATH': '/usr/bin', 'PYTHONPATH': '/bad',
                                           'LD_LIBRARY_PATH': '/bad/lib',
                                           'CONDA_PREFIX': '/bad/conda'}, str(b), str(a))
            self.assertEqual(env['YOLO_AUTOINSTALL'], 'False')
            self.assertNotIn('PYTHONPATH', env)
            self.assertNotIn('LD_LIBRARY_PATH', env)
            self.assertNotIn('CONDA_PREFIX', env)
            self.assertTrue(env['PATH'].startswith(str(b.parent)))
            same = make_export_process_env({'PYTHONPATH': '/studio'}, str(a), str(a))
            self.assertEqual(same['PYTHONPATH'], '/studio')

    def test_jetsontensor_rt_version_matching(self):
        self.assertTrue(tensorrt_versions_match('10.13.3.9', '10.13.3.9-1'))
        self.assertFalse(tensorrt_versions_match('10.15.0.0', '10.13.3.9-1'))
        self.assertIsNone(tensorrt_versions_match(None, '10.13.3.9'))
        self.assertTrue(is_jetson_environment('linux','aarch64','NVIDIA Jetson Thor'))
        self.assertFalse(is_jetson_environment('linux','x86_64','NVIDIA 5090'))

    def test_post_export_verification_isolated_and_fail_closed(self):
        ok = {'verified':True, 'tensorrt':'10.13.3.9', 'test':'synthetic_inference'}
        with patch('model_export.subprocess.run', return_value=types.SimpleNamespace(
                    returncode=0, stdout='STUDIO_ENGINE_VERIFICATION='+json.dumps(ok),
                    stderr='')) as run:
            self.assertTrue(validate_engine_with_selected_runtime('test.engine', 1280)['verified'])
            self.assertEqual(run.call_args.kwargs['env']['YOLO_AUTOINSTALL'], 'False')
        with patch('model_export.subprocess.run', return_value=types.SimpleNamespace(
                    returncode=1, stdout='STUDIO_ENGINE_VERIFICATION='+json.dumps(
                        {'verified':False,'error':'Serialization version mismatch'}), stderr='')):
            result = validate_engine_with_selected_runtime('test.engine', 1280)
            self.assertFalse(result['verified'])
            self.assertIn('Serialization', result['error'])

    def test_unverified_engine_preserved_but_not_reported_successful(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            model=path/'best.pt'; model.write_bytes(b'pretend checkpoint')
            class FakeYOLO:
                def __init__(self, checkpoint): self.file=Path(checkpoint)
                def export(self, **kwargs):
                    engine=self.file.with_suffix('.engine'); engine.write_bytes(b'fake engine')
                    return str(engine)
            cap={'formats':{'engine':{'available':True,'argument':'engine'}}}
            request={'model':str(model),'format':'engine','output':str(path/'exports'),
                     'device':'0','imgsz':640}
            with patch.dict(sys.modules, {'ultralytics':types.SimpleNamespace(YOLO=FakeYOLO)}), \
                 patch('model_export.probe_export_environment', return_value=cap), \
                 patch('model_export.validate_engine_with_selected_runtime', return_value={
                     'verified':False,'error':'TensorRT serialized engine mismatch'}):
                with self.assertRaisesRegex(RuntimeError, 'validation failed'):
                    run_export_job(request)
            self.assertTrue(model.exists())
            metadata=list((path/'exports').rglob('export_summary.json'))
            self.assertEqual(len(metadata), 1)
            summary=json.loads(metadata[0].read_text())
            self.assertFalse(summary['tensorrt_verification']['verified'])
            self.assertTrue(Path(summary['exported_model']).exists())

    def test_verifier_loads_and_predicts_with_tensorrt_selected_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            engine=Path(directory)/'test.engine'; engine.write_bytes(b'engine')
            fake_trt=types.ModuleType('tensorrt'); fake_trt.__version__='10.13.3.9'
            fake_torch=types.ModuleType('torch')
            fake_torch.version=types.SimpleNamespace(cuda='13.0')
            fake_torch.cuda=types.SimpleNamespace(is_available=lambda:True,
                                                   get_device_name=lambda x:'Jetson Thor')
            fake_ultralytics=types.ModuleType('ultralytics')
            called=[]
            class FakeYOLO:
                def __init__(self, model, task=None): called.append(('load',model))
                def predict(self, **kwargs):
                    called.append(('predict',kwargs)); return [object()]
            fake_ultralytics.YOLO=FakeYOLO
            with patch.dict(sys.modules, {'tensorrt':fake_trt,'torch':fake_torch,
                                          'ultralytics':fake_ultralytics}):
                from verify_tensorrt_engine import verify_engine
                result=verify_engine(engine,[640,960])
            self.assertTrue(result['verified'])
            self.assertEqual(result['tensorrt'],'10.13.3.9')
            self.assertEqual(called[-1][1]['imgsz'],[640,960])

if __name__=='__main__':
    unittest.main()
