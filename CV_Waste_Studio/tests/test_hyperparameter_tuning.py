"""Native tuning orchestration and Ultralytics output compatibility."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from hyperparameter_tuning import RANGES, PRESETS, read_trials, best_trial, tune_arguments, validate_settings


class HyperparameterTuningTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.dataset=self.root/'data.yaml'
        self.dataset.write_text('train: images/train\nval: images/val\nnc: 2\nnames: [a, b]\n')
        self.cfg={'data':str(self.dataset),'model':'yolo11l.pt','iterations':3,'epochs':2,
                  'batch':4,'imgsz':640,'workers':8,'cache':'off','device':'cpu',
                  'output_dir':str(self.root/'results'),'space':{'lr0':(1e-5,.005)}}

    def tearDown(self):
        self.tmp.cleanup()

    def test_settings_and_cache_translation(self):
        config=validate_settings(self.cfg)
        args=tune_arguments(config,'tune_a')
        self.assertIs(args['cache'],False)
        self.assertEqual(args['workers'],8)
        self.assertEqual(args['space'],{'lr0':(1e-5,.005)})
        self.assertFalse(args['resume'])
        self.assertTrue(args['val'])

    def test_invalid_bounds_trials_and_workers(self):
        for change in ({'space':{'lr0':(.01,.0001)}}, {'workers':-1},
                       {'iterations':0},{'cache':'nonsense'},{'space':{'foo':(0.,1.)}}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                validate_settings({**self.cfg,**change})

    def test_native_space_and_resume_require_explicit_results(self):
        cfg=validate_settings({**self.cfg,'space':None})
        self.assertNotIn('space',tune_arguments(cfg,'tune'))
        with self.assertRaises(ValueError):
            validate_settings({**self.cfg,'resume':True,'name':'old'})
        folder=self.root/'results'/'old'
        folder.mkdir(parents=True)
        (folder/'tune_results.ndjson').write_text('{}\n')
        self.assertTrue(validate_settings({**self.cfg,'resume':True,'name':'old'})['resume'])

    def test_read_native_ndjson_and_partial_last_line(self):
        folder=self.root/'my_run'
        folder.mkdir()
        rows=[{'iteration':1,'fitness':.45,'hyperparameters':{'lr0':.004},
               'datasets':{'data':{'fitness':.45,'metrics/mAP50-95(B)':.45}}},
              {'iteration':2,'fitness':.52,'hyperparameters':{'lr0':.003},
               'datasets':{'data':{'fitness':.52,'metrics/mAP50-95(B)':.52}}},
              {'iteration':3,'fitness':0.0,'hyperparameters':{'lr0':.009},
               'datasets':{'data':{'fitness':0.0}}}]
        text='\n'.join(json.dumps(p) for p in rows)+'\n{"iteration":'
        (folder/'tune_results.ndjson').write_text(text)
        history=read_trials(folder)
        self.assertEqual(len(history),3)
        self.assertEqual(best_trial(history)['iteration'],2)
        self.assertTrue(history[-1]['failed']) # tuner failure fallback carries only fitness=0


    def test_disk_cache_cleanup_preserves_preexisting_files(self):
        from hyperparameter_tuning import run_native_tuning
        images=self.root/'images'/'train'
        images.mkdir(parents=True)
        for stem in ('old','new'):
            (images/(stem+'.jpg')).write_bytes(b'fake')
        cache_old=images/'old.npy'
        cache_new=images/'new.npy'
        cache_old.write_bytes(b'preexisting')
        self.dataset.write_text('train: images/train\nval: images/train\nnc: 2\nnames: [a, b]\n')
        config=validate_settings({**self.cfg,'cache':'disk'})
        class FakeYOLO:
            def __init__(self,name):pass
            def tune(self,**kwargs):
                path=Path(kwargs['project'])/kwargs['name']
                path.mkdir(parents=True)
                cache_new.write_bytes(b'generated')
                record={'iteration':1,'fitness':.63,'hyperparameters':{'lr0':.004},
                        'datasets':{'data':{'fitness':.63,'metrics/mAP50-95(B)':.63}}}
                (path/'tune_results.ndjson').write_text(json.dumps(record)+'\n')
        fake=types.ModuleType('ultralytics')
        fake.YOLO=FakeYOLO
        with patch.dict(sys.modules,{'ultralytics':fake,'torch':types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda:False))}):
            run_native_tuning(config)
        self.assertEqual(cache_old.read_bytes(),b'preexisting')
        self.assertFalse(cache_new.exists())

    def test_native_model_tune_not_custom_trainer(self):
        from hyperparameter_tuning import run_native_tuning
        config=validate_settings(self.cfg)
        calls=[]
        class FakeYOLO:
            def __init__(self,name):
                calls.append(('init',name))
            def tune(self,**kwargs):
                calls.append(('tune',kwargs))
                folder=Path(kwargs['project'])/kwargs['name']
                folder.mkdir(parents=True)
                trial={'iteration':1,'fitness':.6,'hyperparameters':{'lr0':.004},
                       'datasets':{'data':{'fitness':.6,'metrics/mAP50-95(B)':.6}}}
                (folder/'tune_results.ndjson').write_text(json.dumps(trial)+'\n')
        fake=types.ModuleType('ultralytics')
        fake.YOLO=FakeYOLO
        with patch.dict(sys.modules,{'ultralytics':fake,'torch':types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda:False))}):
            path=run_native_tuning(config)
        self.assertEqual(len(calls),2)
        self.assertEqual(calls[1][0],'tune')
        self.assertEqual(calls[1][1]['iterations'],3)
        self.assertIs(calls[1][1]['cache'],False)
        self.assertTrue((path/'studio_tuning_settings.json').is_file())


if __name__=='__main__': unittest.main()
