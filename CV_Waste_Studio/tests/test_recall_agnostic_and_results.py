"""Tests for Recall@Precision class-agnostic NMS and results-folder shortcut."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_model_recall_at_precision as recall


class RecallAgnosticNms(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / 'best.pt'
        self.model.write_bytes(b'mock model')
        self.image = self.root / 'image.jpg'
        self.image.write_bytes(b'mock image')
        self.settings = dict(model=str(self.model), mode='Image folder', source=str(self.root),
                             output=str(self.root / 'results'), split='test', precision=.95,
                             food_class=0, imgsz=640, batch=4, nms_iou=.7,
                             match_iou=.5, max_det=300, device='CPU', reuse_cache=True,
                             save_plots=False)

    def test_agnostic_nms_changes_cache_identity_but_false_preserves_existing_cache_key(self):
        old_identity = dict(cache_version=recall.CACHE_VERSION,
                            model=recall._normal_path(self.model.resolve()),
                            model_size=self.model.stat().st_size,
                            model_mtime_ns=self.model.stat().st_mtime_ns,
                            imgsz=640, conf=recall.CONF_FLOOR, nms_iou=.7, max_det=300)
        hash_value = hashlib.sha256()
        hash_value.update(json.dumps(old_identity, sort_keys=True).encode('utf-8'))
        image_stat = self.image.stat()
        hash_value.update(recall._normal_path(self.image).encode('utf-8'))
        hash_value.update(f'|{image_stat.st_size}|{image_stat.st_mtime_ns}\n'.encode('ascii'))
        prior_cache_key = hash_value.hexdigest()[:24]
        self.assertEqual(recall._cache_identity(self.settings, [self.image]), prior_cache_key)
        self.assertEqual(recall._cache_identity({**self.settings, 'agnostic_nms': False}, [self.image]), prior_cache_key)
        self.assertNotEqual(recall._cache_identity({**self.settings, 'agnostic_nms': True}, [self.image]), prior_cache_key)

    def test_ultralytics_predict_receives_agnostic_nms_and_metadata_records_it(self):
        captured = []
        class FakeYOLO:
            def __init__(self, path):
                self.names = {0: 'food', 1: 'metal'}
            def predict(self, **kwargs):
                captured.append(kwargs)
                yield types.SimpleNamespace(path=str(image), boxes=None)

        image = self.image
        with patch.dict(sys.modules, {'ultralytics': types.SimpleNamespace(YOLO=FakeYOLO)}), \
             patch.object(recall, 'choose_device', return_value='cpu'):
            for value in (False, True):
                settings = {**self.settings, 'agnostic_nms': value}
                cache = self.root / f'detections_{value}.npz'
                meta_path = self.root / f'detections_{value}.json'
                result, metadata = recall.predict_and_cache(
                    settings, [self.image], cache, meta_path, lambda *_: None)
                self.assertEqual(captured[-1]['agnostic_nms'], value)
                self.assertEqual(metadata['agnostic_nms'], value)
                self.assertEqual(json.loads(meta_path.read_text())['agnostic_nms'], value)
                self.assertTrue(cache.is_file())

    def test_open_results_folder_checks_path_and_invokes_system_file_manager(self):
        folder = self.root / 'precision_recall_test'
        folder.mkdir()
        with patch.object(recall.sys, 'platform', 'linux'), \
             patch.object(recall.subprocess, 'Popen') as popen:
            recall.open_results_folder(folder)
            popen.assert_called_once_with(['xdg-open', str(folder.resolve())])
        with patch.object(recall.sys, 'platform', 'win32'), \
             patch.object(recall.os, 'startfile', create=True) as startfile:
            recall.open_results_folder(folder)
            startfile.assert_called_once_with(str(folder.resolve()))
        with self.assertRaises(FileNotFoundError):
            recall.open_results_folder(self.root / 'missing')


@unittest.skipUnless(os.getenv('DISPLAY') or os.name == 'nt', 'GUI test needs display')
class RecallResultsGUI(unittest.TestCase):
    def test_checkbox_and_results_folder_button_lifecycle(self):
        import tkinter as tk
        try:
            root = tk.Tk()
        except tk.TclError:
            self.skipTest('No accessible Tk display')
        root.withdraw()
        self.addCleanup(root.destroy)
        gui = recall.RecallTestGUI(root, embedded=True)
        self.assertFalse(gui.agnostic_nms_var.get())
        self.assertEqual(str(gui.open_results_button.cget('state')), 'disabled')
        gui.agnostic_nms_var.set(True)
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / 'test.pt'
            model.write_bytes(b'model')
            gui.model_var.set(str(model))
            gui.source_var.set(str(Path(tmp)))
            gui.mode_var.set('Image folder')
            self.assertIs(gui._settings()['agnostic_nms'], True)
        # Test the output workflow without spawning a worker.
        folder = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: folder.rmdir())
        summary = dict(output_directory=str(folder), target_reached=True,
                       class_name='food', confidence_threshold=.82162,
                       achieved_precision=.9501, recall_at_precision=.8134,
                       true_positives=2493, false_positives=131, false_negatives=572)
        gui.messages.put(('done', summary))
        gui._poll()
        self.assertEqual(gui.last_results_dir, str(folder))
        self.assertEqual(str(gui.open_results_button.cget('state')), 'normal')
        with patch.object(recall, 'open_results_folder') as open_mock:
            gui._open_results_folder()
            open_mock.assert_called_once_with(str(folder))
        # Launching a new evaluation must disable the previous result link.
        with patch.object(gui, '_settings', return_value={'model': 'dummy'}), \
             patch.object(recall, 'evaluate', return_value=None):
            gui._start(False)
            self.assertIsNone(gui.last_results_dir)
            self.assertEqual(str(gui.open_results_button.cget('state')), 'disabled')
            gui.messages.put(('cancelled', 'Stopped'))
            gui._poll()
        self.assertEqual(str(gui.open_results_button.cget('state')), 'disabled')


if __name__ == '__main__':
    unittest.main()
