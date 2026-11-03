"""Optional Ray integration must not change the native tuner or require Ray to test."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from hyperparameter_tuning import validate_settings, tune_arguments, run_native_tuning
from ray_tune_support import read_ray_trials, export_ray_best, ray_space


class RayTuningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        dataset = self.root / 'data.yaml'
        dataset.write_text('train: images/train\nval: images/val\n' +
                           'nc: 2\nnames: [food, other]\n', encoding='utf-8')
        self.cfg = {'data': str(dataset), 'model':'yolo11s.pt',
                    'iterations': 3, 'epochs': 10, 'grace_period': 4,
                    'batch': 8, 'imgsz': 1088, 'workers':8, 'device':'cpu',
                    'output_dir': str(self.root / 'runs'), 'name':'raytest',
                    'space': {'lr0': (0.0001, 0.005)}, 'engine':'ray'}

    def test_ray_kwargs_preserve_image_size_and_asynchronous_scheduler(self):
        settings = validate_settings(self.cfg)
        args = tune_arguments(settings, settings['name'])
        self.assertEqual(args['imgsz'], 1088)
        self.assertEqual(args['grace_period'], 4)
        self.assertEqual(args['search_alg'], 'optuna')
        self.assertTrue(args['use_ray'])
        self.assertEqual(args['gpu_per_trial'], 0)
        self.assertIs(args['cache'], False)
        self.assertNotIn('resume', args)

    def test_invalid_ray_settings_and_no_resume(self):
        for extra in ({'grace_period': 0}, {'grace_period': 11},
                      {'engine':'foo'}, {'resume':True}, {'imgsz': 0}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                validate_settings({**self.cfg, **extra})

    def test_native_mode_still_available(self):
        config=validate_settings({**self.cfg, 'engine':'native'})
        args=tune_arguments(config,'test')
        self.assertNotIn('use_ray',args)
        self.assertNotIn('search_alg',args)
        self.assertNotIn('grace_period',args)

    def test_ray_trial_reader_tolerates_partial_json_line(self):
        folder=self.root/'ray'
        one=folder/'train_a'
        one.mkdir(parents=True)
        (one/'result.json').write_text(json.dumps({
            'epoch': 5, 'metrics/mAP50-95(B)': .65,
            'metrics/mAP50(B)': .9, 'config/lr0':.001,
        })+'\n{ "unfinished":')
        row=read_ray_trials(folder)[0]
        self.assertEqual(row['fitness'], .65)
        self.assertEqual(row['epochs'], 5)
        self.assertAlmostEqual(row['hyperparameters']['lr0'], .001)

    def test_ray_model_tune_is_native_and_best_yaml_interoperable(self):
        settings=validate_settings(self.cfg)
        calls=[]
        class FakeYOLO:
            def __init__(self, model):calls.append(('init', model))
            def tune(self, **kwargs):
                calls.append(('tune',kwargs))
                folder=Path(kwargs['project'])/kwargs['name']
                trial=folder/'ray_trial_a'
                trial.mkdir(parents=True)
                (trial/'result.json').write_text(json.dumps({
                    'epoch':7,'metrics/mAP50-95(B)': .8,'config/lr0': .001})+'\n')
                return [types.SimpleNamespace(config={'lr0': .001, 'data': 'do not copy'},
                    metrics={'metrics/mAP50-95(B)': .8}, error=None, path=str(trial))]
        fake_ultralytics=types.ModuleType('ultralytics')
        fake_ultralytics.YOLO=FakeYOLO
        fake_ray=types.ModuleType('ray')
        fake_ray_tune=types.ModuleType('ray.tune')
        fake_ray_tune.uniform=lambda a,b: ('uniform',a,b)
        fake_ray.tune=fake_ray_tune
        fake_torch=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda:False))
        with patch.dict(sys.modules,{'ultralytics':fake_ultralytics,
                                    'ray':fake_ray,'ray.tune':fake_ray_tune,'torch':fake_torch}), \
             patch('ray_tune_support.dependency_report',return_value={'available':True}):
            folder=run_native_tuning(settings)
        self.assertEqual(len(calls),2)
        args=calls[1][1]
        self.assertTrue(args['use_ray'])
        self.assertEqual(args['space'],{'lr0':('uniform',0.0001,0.005)})
        self.assertEqual(args['gpu_per_trial'],0)
        contents=(folder/'best_hyperparameters.yaml').read_text()
        self.assertIn('lr0:',contents)
        self.assertNotIn('do not copy',contents)
        self.assertTrue((folder/'studio_ray_best.json').exists())
        self.assertEqual(len(read_ray_trials(folder)),1)


if __name__=='__main__':unittest.main()
