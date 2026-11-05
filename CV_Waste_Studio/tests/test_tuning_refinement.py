"""Safe cross-mode HPT refinement, boundary flags, and true resume safeguards."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from hyperparameter_tuning import validate_settings, tune_arguments
from tuning_refinement import (load_tuning_source, propose_refinement,
                               saved_settings, source_resume_status, ray_has_state)


class RefinementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.data = self.base/'data.yaml'
        self.data.write_text('train: images/train\nval: images/val\n'
                             'nc: 2\nnames: [food, other]\n',encoding='utf-8')
        self.cfg = dict(data=str(self.data),model='yolo11s.pt',epochs=10,
                        iterations=5,batch=8,workers=8,imgsz=960,cache='off',
                        device='cpu',optimizer='AdamW',grace_period=4,
                        space={'lr0': [0.0001,0.004], 'momentum':[0.7,0.98]},
                        output_dir=str(self.base),name='old',engine='native')
        self.run=self.base/'old'
        self.run.mkdir()
        (self.run/'best_hyperparameters.yaml').write_text(
            '# saved by HPT\nlr0: 0.00051\nmomentum: 0.7\nmosaic: 0.82613\n'
            'data: malicious.yaml\nmodel: another.pt\n',encoding='utf-8')
        (self.run/'tune_results.ndjson').write_text(json.dumps({
            'iteration': 1,'fitness':.7,'hyperparameters':{'lr0':.00051},
            'metrics':{'metrics/mAP50-95(B)':.7}})+'\n',encoding='utf-8')
        (self.base/'old.studio_tuning_settings.json').write_text(json.dumps(self.cfg),encoding='utf-8')

    def test_load_folder_and_yml_ignore_untrusted_old_paths(self):
        for origin in (self.run,self.run/'best_hyperparameters.yaml'):
            with self.subTest(path=origin):
                source=load_tuning_source(origin)
                self.assertEqual(source['hyperparameters']['lr0'],.00051)
                self.assertEqual(source['settings']['model'],'yolo11s.pt')
                self.assertNotIn('data',source['hyperparameters'])
                self.assertNotIn('model',source['hyperparameters'])
                self.assertEqual(source['previous_space']['momentum'],[.7,.98])

    def test_refinement_flags_boundary_and_does_not_mutate_source(self):
        before=(self.run/'best_hyperparameters.yaml').read_bytes()
        source=load_tuning_source(self.run)
        proposals=propose_refinement(source['hyperparameters'],source['previous_space'],.20)
        self.assertIn('Near lower bound',proposals['momentum']['warning'])
        self.assertEqual(proposals['momentum']['min'],.7)
        self.assertGreater(proposals['momentum']['max'],.7)
        self.assertLess(proposals['lr0']['max'],.004)
        self.assertEqual((self.run/'best_hyperparameters.yaml').read_bytes(),before)

    def test_old_search_boundary_is_explored_outward(self):
        proposal=propose_refinement({'momentum':.81}, {'momentum':(.80,.95)}, width=.01)
        self.assertIn('lower',proposal['momentum']['warning'])
        self.assertLess(proposal['momentum']['min'],.8)

    def test_yaml_alone_not_full_resume(self):
        other=self.base/'standalone.yaml'
        other.write_text('lr0: 0.0006\n')
        source=load_tuning_source(other)
        self.assertIsNone(source['folder'])
        self.assertFalse(source['settings'])
        self.assertFalse(source_resume_status(self.base/'not_a_run')[0])

    def test_resume_requires_same_original_engine_and_configuration(self):
        original=validate_settings({**self.cfg,'resume':True,'iterations':10})
        self.assertTrue(original['resume'])
        for changed in ({'model':'yolov8s.pt'}, {'imgsz':640},
                        {'space': {'lr0':(1e-5,.001)}}, {'engine':'ray'}):
            with self.subTest(changed=changed),self.assertRaises(ValueError):
                validate_settings({**self.cfg,'resume':True,'iterations':10,**changed})
        self.assertTrue(source_resume_status(self.run)[0])

    def test_ray_resume_requires_original_ray_state(self):
        cfg=dict(self.cfg,engine='ray')
        (self.base/'old.studio_tuning_settings.json').write_text(json.dumps(cfg),encoding='utf-8')
        self.assertFalse(source_resume_status(self.run)[0])
        with self.assertRaises(ValueError):
            validate_settings(dict(cfg,resume=True))
        (self.run/'tuner.pkl').write_bytes(b'fake for disk validation')
        (self.run/'experiment_state-001.json').write_text('{}')
        self.assertTrue(ray_has_state(self.run))
        result=validate_settings(dict(cfg,resume=True))
        kwargs=tune_arguments(result,'old')
        self.assertTrue(kwargs['use_ray'])
        self.assertTrue(kwargs['resume'])
        self.assertTrue(kwargs['exist_ok'])
        self.assertEqual(kwargs['iterations'],5)
        with self.assertRaises(ValueError):
            validate_settings(dict(cfg,resume=True,iterations=6))

    def test_ray_resume_worker_checks_real_tuner_restore(self):
        import types
        from unittest.mock import patch
        from hyperparameter_tuning import run_native_tuning
        cfg=dict(self.cfg,engine='ray')
        (self.base/'old.studio_tuning_settings.json').write_text(json.dumps(cfg))
        (self.run/'tuner.pkl').write_bytes(b'original Ray state')
        (self.run/'experiment_state-001.json').write_text('{}')
        settings=validate_settings(dict(cfg,resume=True))
        calls=[]
        run_folder=self.run
        class FakeYOLO:
            def __init__(self, model): pass
            def tune(self,**kwargs):
                calls.append(kwargs)
                trial=run_folder/'ray_trial'
                trial.mkdir(exist_ok=True)
                (trial/'result.json').write_text(json.dumps({
                    'epoch':10, 'metrics/mAP50-95(B)':.74,
                    'config/lr0':.00051})+'\n')
                return [types.SimpleNamespace(config={'lr0':.00051},
                    metrics={'metrics/mAP50-95(B)':.74},error=None,path=str(trial))]
        fake=types.ModuleType('ultralytics'); fake.YOLO=FakeYOLO
        ray=types.ModuleType('ray'); ray.tune=types.ModuleType('ray.tune')
        ray.tune.uniform=lambda a,b:('uniform',a,b)
        ray.tune.Tuner=types.SimpleNamespace(can_restore=lambda folder:folder==str(self.run))
        ray.is_initialized=lambda:False
        fake_torch=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda:False))
        with patch.dict(sys.modules,{'ultralytics':fake,'ray':ray,'ray.tune':ray.tune,'torch':fake_torch}), \
             patch('ray_tune_support.dependency_report',return_value={'available':True}):
            result=run_native_tuning(settings)
        self.assertEqual(result,self.run)
        self.assertTrue(calls[0]['resume'])
        self.assertTrue(calls[0]['exist_ok'])
        self.assertEqual(calls[0]['iterations'],cfg['iterations'])

    def test_missing_source_yaml_can_use_trial_history(self):
        (self.run/'best_hyperparameters.yaml').unlink()
        source=load_tuning_source(self.run)
        self.assertEqual(source['hyperparameters']['lr0'],.00051)
        self.assertEqual(source['winner']['iteration'],1)


if __name__=='__main__':unittest.main()

# Tests above remain the basis for both modes. The Ray restore state must be
# verified a second time *inside* the worker; a disk-file check in the GUI
# alone is insufficient for a different Conda/Ray version.
