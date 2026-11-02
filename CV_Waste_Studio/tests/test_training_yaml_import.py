"""Regression tests for train-only Ultralytics YAML import."""
from pathlib import Path
import json
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_yaml_import import parse_training_yaml, validate_training_overrides
import train_and_split_tool_GUI as backend
import run_training_job


class UltralyticsTrainingYAMLTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.file = self.root / 'args.yaml'

    def test_full_args_yaml_applies_advanced_and_protects_run_paths(self):
        self.file.write_text('''
model: old-model.pt
data: old-data.yaml
project: old-project
name: old-run
save_dir: old-output
mode: train
epochs: 80
imgsz: 1088
batch: 8
workers: 6
cache: false
cls: 3.5
optimizer: AdamW
lr0: 0.0003
lrf: 0.2
warmup_epochs: 3
weight_decay: 0.0005
rect: true
amp: true
mosaic: 0.35
hsv_s: 0.45
agnostic_nms: true
''', encoding='utf-8')
        config = parse_training_yaml(self.file)
        self.assertEqual(config['visible']['imgsz'], 1088)
        self.assertEqual(config['visible']['cache'], 'off')
        self.assertEqual(config['visible']['cls'], 3.5)
        self.assertEqual(config['advanced']['lr0'], 0.0003)
        self.assertEqual(config['advanced']['optimizer'], 'AdamW')
        self.assertTrue(config['advanced']['rect'])
        self.assertEqual(config['augmentation'], {'mosaic': 0.35, 'hsv_s': 0.45})
        self.assertIn('data', config['skipped'])
        self.assertIn('model', config['skipped'])
        self.assertIn('agnostic_nms', config['skipped'])

    def test_best_hyperparameters_yaml(self):
        self.file.write_text('lr0: 0.00087\nlrf: 0.21\nmomentum: 0.91\ncls: 1.4\n'
                             'hsv_h: 0.025\nclose_mosaic: 7\n', encoding='utf-8')
        config = parse_training_yaml(self.file)
        self.assertEqual(config['visible'], {'cls': 1.4})
        self.assertEqual(config['augmentation']['close_mosaic'], 7)
        self.assertIn('momentum', config['advanced'])

    def test_rejects_unsupported_direct_overrides_and_malformed_yaml(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            validate_training_overrides({'data': 'unrelated.yaml'})
        with self.assertRaisesRegex(ValueError, 'optimizer'):
            validate_training_overrides({'optimizer': []})
        with self.assertRaisesRegex(ValueError, 'rect'):
            validate_training_overrides({'rect': 'yes'})
        self.file.write_text('!!python/object/apply:os.system ["echo unsafe"]')
        with self.assertRaisesRegex(ValueError, 'Cannot read'):
            parse_training_yaml(self.file)

    def test_backend_from_imported_yaml_preserves_source_split_and_passes_extra_kwargs(self):
        self.file.write_text('''
model: bad-or-old-model.pt
data: old.yaml
cls: 3.0
imgsz: 960
cache: false
workers: 8
lr0: 0.0002
optimizer: AdamW
rect: true
hsv_h: 0.02
''', encoding='utf-8')
        imported = parse_training_yaml(self.file)
        yaml = self.root / 'data.yaml'
        yaml.write_text('train: images/train\nval: images/val\nnames: [plastic, metal]\n', encoding='utf-8')
        original = yaml.read_bytes()
        received = []
        class FakeYOLO:
            def __init__(self, file):
                received.append(file)
            def train(self, **kwargs):
                received.append(kwargs)
                return 'trained'
        fake_ultra = types.ModuleType('ultralytics')
        fake_ultra.YOLO = FakeYOLO
        values = imported['visible']
        job = {'existing_data_yaml': str(yaml), 'train_after_split': True,
               'version': '11', 'size': 'n', 'project': str(self.root / 'new-results'),
               'run_name': 'new-run', 'epochs': 5, 'batch': 4,
               'imgsz': values['imgsz'], 'workers': values['workers'],
               'cls_weight': values['cls'], 'cache': values['cache'],
               'augmentation': imported['augmentation'],
               'training_overrides': imported['advanced'],
               'device': 'cpu', 'compact_progress': False}
        # Same on-disk JSON protocol used by the real job launcher.
        settings_json = self.root / 'settings.json'
        settings_json.write_text(json.dumps(job), encoding='utf-8')
        with patch.object(backend, 'autosplit_dataset', side_effect=AssertionError('split called')), \
             patch.object(backend, '_ensure_gpu_pytorch', return_value=None), \
             patch.object(backend, '_load_dependency', return_value=None), \
             patch.object(backend, '_select_training_device', return_value='cpu'), \
             patch.dict(sys.modules, {'ultralytics': fake_ultra, 'torch': types.ModuleType('torch')}):
            run_training_job.main([str(settings_json)])
        self.assertEqual(received[0], 'yolo11n.pt')
        train_args = received[1]
        self.assertEqual(train_args['data'], str(yaml))
        self.assertEqual(train_args['project'], str(self.root / 'new-results'))
        self.assertEqual(train_args['name'], 'new-run')
        self.assertEqual(train_args['imgsz'], 960)
        self.assertIs(train_args['cache'], False)
        self.assertEqual(train_args['workers'], 8)
        self.assertEqual(train_args['cls'], 3.0)
        self.assertEqual(train_args['lr0'], 0.0002)
        self.assertEqual(train_args['optimizer'], 'AdamW')
        self.assertTrue(train_args['rect'])
        self.assertEqual(train_args['hsv_h'], 0.02)
        self.assertEqual(yaml.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
