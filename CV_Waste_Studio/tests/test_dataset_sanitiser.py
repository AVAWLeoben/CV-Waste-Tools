"""Tests for safe dataset sanitisation, with mocked YOLO inference."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dataset_sanitiser as sanitise


class SanitiserTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        self.root = self.home / 'dataset'
        (self.root / 'images' / 'train').mkdir(parents=True)
        (self.root / 'labels' / 'train').mkdir(parents=True)
        for name in ('keep', 'empty', 'missing'):
            (self.root / 'images' / 'train' / f'{name}.jpg').write_bytes(b'jpg_' + name.encode())
        (self.root / 'labels' / 'train' / 'keep.txt').write_text('1 0.5 0.5 0.4 0.4\n')
        (self.root / 'labels' / 'train' / 'empty.txt').write_text(' \n\t')
        self.plan = self.home / 'plan.json'
        self.copy = self.home / 'cleaned'
        self.settings = dict(source=str(self.root), method='labels', action='copy', output=str(self.copy))

    def test_label_preview_and_copy_preserve_original(self):
        (self.root / 'train.txt').write_text(str(self.root / 'images' / 'train' / 'empty.jpg')+'\n' +
                                             str(self.root / 'images' / 'train' / 'keep.jpg')+'\n')
        (self.root / 'data.yaml').write_text(f'path: {self.root}\ntrain: train.txt\nnames: [plastic]\n')
        result = sanitise.scan_dataset(self.settings, self.plan)
        self.assertEqual(result['total'], 3)
        self.assertEqual([r['image'] for r in result['remove']], ['images/train/empty.jpg'])
        self.assertEqual(result['missing_labels'], 1)
        self.assertTrue((self.root/'images/train/empty.jpg').exists())
        applied = sanitise.apply_plan(self.plan)
        self.assertEqual(applied['removed'], 1)
        self.assertFalse((self.copy/'images/train/empty.jpg').exists())
        self.assertFalse((self.copy/'labels/train/empty.txt').exists())
        self.assertTrue((self.copy/'images/train/keep.jpg').exists())
        self.assertTrue((self.copy/'images/train/missing.jpg').exists())
        self.assertTrue((self.root/'images/train/empty.jpg').exists())
        self.assertTrue((self.root/'labels/train/empty.txt').exists())
        self.assertEqual((self.copy/'train.txt').read_text().strip(),
                         str(self.copy/'images/train/keep.jpg'))
        self.assertIn(str(self.copy), (self.copy/'data.yaml').read_text())
        self.assertNotIn(str(self.root), (self.copy/'data.yaml').read_text())

    def test_empty_missing_opt_in(self):
        self.settings['include_missing'] = True
        plan = sanitise.scan_dataset(self.settings, self.plan)
        self.assertEqual(len(plan['remove']), 2)
        self.assertEqual([r['image'] for r in plan['remove']],
                         ['images/train/empty.jpg', 'images/train/missing.jpg'])

    def test_inplace_removes_both_and_updates_manifest(self):
        self.settings['action'] = 'inplace'
        self.settings.pop('output')
        (self.root/'val.txt').write_text('images/train/empty.jpg\nimages/train/keep.jpg\n')
        sanitise.scan_dataset(self.settings, self.plan)
        result = sanitise.apply_plan(self.plan)
        self.assertEqual(result['removed'], 1)
        self.assertFalse((self.root/'images/train/empty.jpg').exists())
        self.assertFalse((self.root/'labels/train/empty.txt').exists())
        self.assertTrue((self.root/'images/train/keep.jpg').exists())
        self.assertEqual((self.root/'val.txt').read_text(), 'images/train/keep.jpg\n')
        self.assertTrue(self.plan.with_name('plan_delete_audit.json').exists())

    def test_modified_file_aborts_before_delete(self):
        self.settings['action'] = 'inplace'
        sanitise.scan_dataset(self.settings, self.plan)
        (self.root/'images/train/empty.jpg').write_bytes(b'new data')
        with self.assertRaisesRegex(ValueError, 'changed since preview'):
            sanitise.apply_plan(self.plan)
        self.assertTrue((self.root/'labels/train/empty.txt').exists())

    def test_new_label_after_missing_label_preview_aborts(self):
        self.settings['action'] = 'inplace'
        self.settings['include_missing'] = True
        sanitise.scan_dataset(self.settings, self.plan)
        (self.root/'labels/train/missing.txt').write_text('0 0.5 0.5 0.1 0.1\n')
        with self.assertRaisesRegex(ValueError, 'label appeared'):
            sanitise.apply_plan(self.plan)
        self.assertTrue((self.root/'images/train/empty.jpg').exists())

    def test_missing_images_aborts_before_delete(self):
        self.settings['action'] = 'inplace'
        sanitise.scan_dataset(self.settings, self.plan)
        (self.root/'images/train/extra.png').write_bytes(b'x')
        with self.assertRaisesRegex(ValueError, 'images changed'):
            sanitise.apply_plan(self.plan)

    def test_empty_source_not_accepted(self):
        with self.assertRaisesRegex(ValueError, 'Select a dataset'):
            sanitise.validate_settings({**self.settings, 'source': ''})

    def test_output_safety(self):
        for candidate in (self.root, self.root/'new', self.home):
            with self.subTest(candidate=candidate):
                with self.assertRaises((ValueError, FileExistsError)):
                    sanitise.validate_settings({**self.settings, 'output':str(candidate)})

    def test_duplicate_stem_rejected(self):
        (self.root/'images/train/empty.png').write_bytes(b'png')
        with self.assertRaisesRegex(ValueError, 'share the label'):
            sanitise.scan_dataset(self.settings, self.plan)

    def test_nested_split_structure(self):
        nested = self.home/'nested'
        (nested/'train/images').mkdir(parents=True)
        (nested/'train/labels').mkdir(parents=True)
        (nested/'train/images/x.jpg').write_bytes(b'img')
        (nested/'train/labels/x.txt').write_text('')
        plan = sanitise.scan_dataset({**self.settings, 'source':str(nested), 'output':str(self.home/'nested_clean')}, self.plan)
        self.assertEqual(plan['remove'][0]['label'], 'train/labels/x.txt')

    def test_mixed_layout(self):
        mixed = self.home/'mixed'
        mixed.mkdir()
        (mixed/'a.jpg').write_bytes(b'image')
        (mixed/'a.txt').write_text('\n')
        plan = sanitise.scan_dataset({**self.settings, 'source':str(mixed), 'output':str(self.home/'mixed_clean')}, self.plan)
        self.assertEqual(len(plan['remove']), 1)
        sanitise.apply_plan(self.plan)
        self.assertFalse((self.home/'mixed_clean/a.jpg').exists())

    def test_symlink_label_rejected(self):
        (self.root/'labels/train/missing.txt').symlink_to(self.root/'labels/train/keep.txt')
        with self.assertRaisesRegex(ValueError, 'symbolic link'):
            sanitise.scan_dataset(self.settings, self.plan)

    def test_model_scans_boxes_not_labels_and_uses_batched_yolo(self):
        model = self.home/'best.pt'
        model.write_bytes(b'fake')
        calls = []
        class YOLO:
            task = 'segment'
            def __init__(self, path):
                calls.append(('loaded', path))
            def predict(self, source, **kwargs):
                calls.append(('predict', source, kwargs))
                return [types.SimpleNamespace(boxes=[] if 'empty' in path else [object()]) for path in source]
        mod = types.ModuleType('ultralytics')
        mod.YOLO = YOLO
        settings = {**self.settings, 'method':'model', 'model':str(model), 'batch':2, 'imgsz':1280,
                    'conf':0.4, 'device':'cpu'}
        with patch.dict(sys.modules, {'ultralytics':mod}):
            plan = sanitise.scan_dataset(settings, self.plan)
        self.assertEqual(len(plan['remove']), 1)
        self.assertEqual(plan['remove'][0]['image'], 'images/train/empty.jpg')
        self.assertEqual(calls[1][2]['imgsz'], 1280)
        self.assertEqual(calls[1][2]['conf'], 0.4)
        self.assertEqual(calls[1][2]['device'], 'cpu')
        self.assertEqual(calls[1][2]['batch'], 2)

    def test_model_missing_boxes_aborts_scan(self):
        model = self.home/'best.pt'
        model.write_bytes(b'fake')
        class YOLO:
            task = 'classify'
            def __init__(self, path): pass
        mod = types.ModuleType('ultralytics')
        mod.YOLO = YOLO
        with patch.dict(sys.modules, {'ultralytics':mod}):
            with self.assertRaisesRegex(ValueError, 'detection/segmentation'):
                sanitise.scan_dataset({**self.settings, 'method':'model', 'model':str(model)}, self.plan)
        self.assertFalse(self.plan.exists())

    def test_worker_round_trip_copy(self):
        import subprocess
        worker = Path(sanitise.__file__).with_name('run_dataset_sanitiser.py')
        scan_job = self.home/'scan_job.json'
        scan_job.write_text(json.dumps({'phase':'scan','plan':str(self.plan),'settings':self.settings}))
        result = subprocess.run([sys.executable, str(worker), str(scan_job)],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('SCAN_COMPLETE', result.stdout)
        apply_job = self.home/'apply_job.json'
        apply_job.write_text(json.dumps({'phase':'apply','plan':str(self.plan)}))
        result = subprocess.run([sys.executable, str(worker), str(apply_job)],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('APPLY_COMPLETE', result.stdout)
        self.assertTrue((self.root/'images/train/empty.jpg').exists())
        self.assertFalse((self.copy/'images/train/empty.jpg').exists())

    def test_cannot_overwrite_cleaned_dataset(self):
        sanitise.scan_dataset(self.settings, self.plan)
        self.copy.mkdir()
        with self.assertRaises(FileExistsError):
            sanitise.apply_plan(self.plan)


if __name__ == '__main__':
    unittest.main()
