"""Recall@Precision cancellation regression checks (no GPU needed)."""
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_model_recall_at_precision as recall


class EvaluationCancellation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / 'best.pt'
        self.model.write_bytes(b'mock model file')
        self.images = [self.root / 'images' / f'frame{i}.jpg' for i in range(2)]
        self.images[0].parent.mkdir()
        for image in self.images:
            image.write_bytes(b'mock image')
        self.settings = dict(
            model=str(self.model), mode='Image folder', source=str(self.images[0].parent),
            output=str(self.root / 'results'), split='test', precision=.95, food_class=0,
            imgsz=640, batch=8, nms_iou=.7, match_iou=.5, max_det=300,
            device='CPU', reuse_cache=True, save_plots=False)

    def test_stop_during_inference_does_not_save_partial_cache(self):
        stop = threading.Event()
        cache = self.root / 'results' / 'prediction_cache' / 'test.npz'
        meta = cache.with_suffix('.json')

        class FakeYOLO:
            def __init__(self, _model):
                self.names = ['object']

            def predict(self, **_kwargs):
                yield types.SimpleNamespace(path=str(self_images[0]), boxes=None)
                stop.set()
                yield types.SimpleNamespace(path=str(self_images[1]), boxes=None)

        self_images = self.images
        with patch.dict(sys.modules, {'ultralytics': types.SimpleNamespace(YOLO=FakeYOLO)}), \
             patch.object(recall, 'choose_device', return_value='cpu'):
            with self.assertRaises(recall.EvaluationCancelled):
                recall.predict_and_cache(self.settings, self.images, cache, meta,
                                         lambda *_: None, cancel_event=stop)
        self.assertFalse(cache.exists())
        self.assertFalse(meta.exists())

    def test_cancel_cached_evaluation_keeps_existing_prediction_cache(self):
        stop = threading.Event()
        identity = recall._cache_identity(self.settings, self.images[:1])
        cache_dir = Path(self.settings['output']) / 'prediction_cache'
        cache_dir.mkdir(parents=True)
        cache = cache_dir / f'detections_{identity}.npz'
        meta = cache_dir / f'detections_{identity}.json'
        np.savez_compressed(cache, image_index=np.array([], dtype=np.int32),
                            classes=np.array([], dtype=np.int32),
                            confidence=np.array([], dtype=np.float32),
                            xyxy=np.empty((0, 4), dtype=np.float32))
        meta.write_text(json.dumps(dict(cache_version=recall.CACHE_VERSION,
                                        image_count=1, class_names={})), encoding='utf-8')
        cache_before = cache.read_bytes()
        meta_before = meta.read_bytes()

        def stop_during_labels(_path):
            stop.set()
            return np.asarray([[0, .2, .2, .8, .8]], dtype=np.float32)

        with patch.object(recall, 'collect_images', return_value=self.images[:1]), \
             patch.object(recall, '_parse_yolo_label', side_effect=stop_during_labels):
            with self.assertRaises(recall.EvaluationCancelled):
                recall.evaluate(self.settings, lambda *_: None, cancel_event=stop)
        self.assertEqual(cache.read_bytes(), cache_before)
        self.assertEqual(meta.read_bytes(), meta_before)
        self.assertFalse(list(Path(self.settings['output']).glob('precision_recall_*')))

    def test_stopping_during_matching_interrupts_before_output(self):
        event = threading.Event()
        gt = [np.array([[0, .1, .1, .9, .9]], dtype=np.float32)] * 3
        prediction = dict(classes=np.array([0, 0, 0]),
                          confidence=np.array([.95, .9, .8]),
                          image_index=np.array([0, 1, 2]),
                          xyxy=np.array([[.1, .1, .9, .9]] * 3))
        def stop_after_first(*args):
            event.set()
            return np.asarray([1.0])
        with patch.object(recall, 'box_iou', side_effect=stop_after_first):
            with self.assertRaises(recall.EvaluationCancelled):
                recall.compute_curve(prediction, gt, 0, .95, .5, cancel_event=event)

    def test_cancelled_before_start_creates_no_output(self):
        stop = threading.Event()
        stop.set()
        with self.assertRaises(recall.EvaluationCancelled):
            recall.evaluate(self.settings, lambda *_: None, cancel_event=stop)
        self.assertFalse(Path(self.settings['output']).exists())


@unittest.skipUnless(os.getenv('DISPLAY') or os.name == 'nt', 'GUI test needs display')
class RecallGUIStopButton(unittest.TestCase):
    def test_stop_button_changes_state_and_restores_form(self):
        import tkinter as tk
        try:
            root = tk.Tk()
        except tk.TclError:
            self.skipTest('No accessible Tk display')
        root.withdraw()
        self.addCleanup(root.destroy)
        gui = recall.RecallTestGUI(root, embedded=True)
        gui.model_var.set(str(Path(__file__).resolve()))
        gui.source_var.set(str(Path(__file__).parent))
        gui.mode_var.set('Image folder')
        # Bypass settings validation; the background worker will wait for Stop.
        with patch.object(gui, '_settings', return_value={'model': 'fake'}), \
             patch.object(recall, 'evaluate', side_effect=self._wait_for_cancel):
            gui._start(False)
            self.assertTrue(gui.busy)
            self.assertEqual(str(gui.stop_button.cget('state')), 'normal')
            gui.cancel()
            self.assertTrue(gui.cancel_event.is_set())
            self.assertEqual(str(gui.stop_button.cget('state')), 'disabled')
            for _ in range(120):
                root.update()
                if not gui.busy:
                    break
                time.sleep(.01)
            self.assertFalse(gui.busy)
            self.assertEqual(gui.status_var.get(), 'Evaluation stopped.')
            self.assertEqual(str(gui.run_button.cget('state')), 'normal')

    @staticmethod
    def _wait_for_cancel(_settings, _emit, *, cache_only=False, cancel_event=None):
        cancel_event.wait(2)
        recall._check_cancelled(cancel_event)


if __name__ == '__main__':
    unittest.main()
