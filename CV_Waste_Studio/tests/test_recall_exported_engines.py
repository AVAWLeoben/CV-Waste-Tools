"""Exported TensorRT/ONNX Recall@Precision support: independent of Jetson hardware."""
from __future__ import annotations

import io
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_model_recall_at_precision as recall
import run_recall_evaluation as worker


class ExportedModelRecallTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = self.root / 'sorting_1280.engine'
        self.model.write_bytes(b'fake TensorRT engine')
        self.python = Path(sys.executable).resolve()
        self.settings = {'model': str(self.model), 'imgsz': 1280, 'batch': 1,
                         'device': 'Auto', 'evaluation_python': str(self.python)}

    def write_export_summary(self, *, imgsz=1280, batch=1, dynamic=False):
        (self.root / 'export_summary.json').write_text(json.dumps({
            'exported_model': str(self.model),
            'settings': {'imgsz': imgsz, 'batch': batch, 'dynamic': dynamic},
        }))

    def test_supported_model_extensions(self):
        self.assertEqual(recall.MODEL_EXTENSIONS, {'.pt', '.engine', '.onnx'})

    def test_recognizes_export_metadata_only_for_same_engine(self):
        self.write_export_summary()
        self.assertEqual(recall.engine_export_settings(self.model)['imgsz'], 1280)
        self.assertEqual(recall.engine_export_settings(self.root / 'other.engine'), {})
        self.assertEqual(recall.engine_export_settings(self.root / 'best.pt'), {})

    def test_requires_matching_static_engine_size_and_batch(self):
        self.write_export_summary()
        recall.validate_engine_input_settings(self.settings)
        with self.assertRaisesRegex(ValueError, 'imgsz=1280'):
            recall.validate_engine_input_settings({**self.settings, 'imgsz': 640})
        with self.assertRaisesRegex(ValueError, 'batch=1'):
            recall.validate_engine_input_settings({**self.settings, 'batch': 8})
        with self.assertRaisesRegex(ValueError, 'NVIDIA GPU'):
            recall.validate_engine_input_settings({**self.settings, 'device': 'CPU'})

    def test_dynamic_engine_accepts_other_settings(self):
        self.write_export_summary(imgsz=[640, 1088], batch=4, dynamic=True)
        recall.validate_engine_input_settings({**self.settings, 'imgsz': 1280, 'batch': 1})

    def test_rectangular_static_engine_input(self):
        self.write_export_summary(imgsz=[640, 1088])
        recall.validate_engine_input_settings({**self.settings, 'imgsz': (640, 1088)})
        with self.assertRaises(ValueError):
            recall.validate_engine_input_settings({**self.settings, 'imgsz': 1088})

    def test_cache_identity_separates_engine_interpreters_not_pt(self):
        image = self.root / 'one.jpg'
        image.write_bytes(b'123')
        base = {**self.settings, 'nms_iou': .7, 'max_det': 300}
        a = recall._cache_identity(base, [image])
        b = recall._cache_identity({**base, 'evaluation_python': '/other/bin/python'}, [image])
        self.assertNotEqual(a, b)
        pt = self.root / 'best.pt'
        pt.write_bytes(b'123')
        a = recall._cache_identity({**base, 'model': str(pt)}, [image])
        b = recall._cache_identity({**base, 'model': str(pt), 'evaluation_python': '/other/bin/python'}, [image])
        self.assertEqual(a, b)

    def test_process_launch_uses_selection_disables_auto_install_and_handles_events(self):
        events = []
        summary = {'target_precision': .95, 'achieved_precision': .97}
        class FakeProc:
            def __init__(self):
                self.stdout = io.StringIO('STUDIO_RECALL_EVENT=' + json.dumps({
                    'kind': 'log', 'value': 'Inference running'}) + '\nSTUDIO_RECALL_EVENT=' + json.dumps({
                    'kind': 'done', 'value': summary}) + '\n')
            def wait(self): return 0
            def poll(self): return 0
        def start(args, **kwargs):
            self.assertEqual(Path(args[0]), self.python)
            self.assertEqual(args[1], '-u')
            self.assertEqual(Path(args[2]).name, 'run_recall_evaluation.py')
            self.assertEqual(kwargs['env']['YOLO_AUTOINSTALL'], 'False')
            self.assertEqual(kwargs['env']['YOLO_AUTOINSTALL'], 'False')
            settings = json.loads(Path(args[3]).read_text())
            self.assertEqual(settings['model'], str(self.model))
            return FakeProc()
        with patch.object(recall.subprocess, 'Popen', side_effect=start):
            result = recall.run_engine_evaluation(self.settings, lambda *event: events.append(event))
        self.assertEqual(result, summary)
        self.assertIn(('log', 'Inference running'), events)
        self.assertIn(('done', summary), events)

    def test_process_error_surfaces_real_runtime_exception(self):
        events = []
        class FakeProc:
            def __init__(self):
                self.stdout = io.StringIO('STUDIO_RECALL_EVENT=' + json.dumps({
                    'kind': 'error', 'value': 'TensorRT serialization version mismatch'}) + '\n')
            def wait(self): return 1
            def poll(self): return 1
        with patch.object(recall.subprocess, 'Popen', return_value=FakeProc()):
            with self.assertRaisesRegex(RuntimeError, 'serialization version mismatch'):
                recall.run_engine_evaluation(self.settings, lambda *event: events.append(event))
        self.assertNotIn('done', [e[0] for e in events])

    def test_stop_marker_is_sent_while_engine_has_no_stdout(self):
        event = threading.Event()
        marker_seen = threading.Event()
        class FakeProc:
            def __init__(self, cancel_file):
                self.cancel_file = cancel_file
                self.stdout = self
            def __iter__(self):
                event.set()
                for _ in range(90):
                    if self.cancel_file.is_file():
                        marker_seen.set()
                        break
                    time.sleep(.02)
                return iter(())
            def wait(self): return 2
            def poll(self): return 2
            def terminate(self): pass
            def close(self): pass
        def start(args, **kwargs):
            return FakeProc(Path(args[-1]))
        with patch.object(recall.subprocess, 'Popen', side_effect=start):
            with self.assertRaises(recall.EvaluationCancelled):
                recall.run_engine_evaluation(self.settings, lambda *_: None, cancel_event=event)
        self.assertTrue(marker_seen.is_set(), 'Cancel marker must appear even without child stdout')

    def test_real_worker_round_trip_from_existing_prediction_cache(self):
        """Real child Python imports the worker, evaluates GT, returns JSON and saves report."""
        import numpy as np
        image_folder = self.root / 'images'
        label_folder = self.root / 'labels'
        image_folder.mkdir()
        label_folder.mkdir()
        image = image_folder / 'frame.jpg'
        image.write_bytes(b'cached-only - image never decoded')
        (label_folder / 'frame.txt').write_text('0 0.5 0.5 0.5 0.5\n')
        settings = dict(self.settings, mode='Image folder', source=str(image_folder),
                        split='test', food_class=0, precision=.95, match_iou=.5,
                        nms_iou=.7, max_det=300, reuse_cache=True, save_plots=False,
                        output=str(self.root / 'results'), agnostic_nms=False)
        digest = recall._cache_identity(settings, [image.resolve()])
        folder = Path(settings['output']) / 'prediction_cache'
        folder.mkdir(parents=True)
        np.savez_compressed(folder / f'detections_{digest}.npz',
                            image_index=np.array([0], np.int32), classes=np.array([0], np.int32),
                            confidence=np.array([.99], np.float32),
                            xyxy=np.array([[.25, .25, .75, .75]], np.float32))
        (folder / f'detections_{digest}.json').write_text(json.dumps({
            'cache_version': recall.CACHE_VERSION, 'image_count': 1,
            'class_names': {'0': 'waste'}, 'device': '0', 'inference_seconds': 3.5,
        }))
        events = []
        returned = recall.run_engine_evaluation(settings, lambda *event: events.append(event))
        self.assertTrue(returned['target_reached'])
        self.assertEqual(returned['recall_at_precision'], 1.0)
        self.assertTrue(returned['used_cached_predictions'])
        self.assertTrue(Path(returned['curve_csv']).is_file())
        self.assertTrue(Path(returned['output_directory'], 'operating_point_summary.json').is_file())
        self.assertEqual(events[-1][0], 'done')

    def test_headless_worker_serializes_done_and_cancel(self):
        req = self.root / 'request.json'
        req.write_text(json.dumps(self.settings))
        cancel = self.root / 'cancel'
        with patch.object(recall, 'evaluate', side_effect=lambda _s, emit, **kw: emit('done', {'ok': True})):
            with patch('run_recall_evaluation._event') as emit:
                self.assertEqual(worker.main([str(req), '--cancel-file', str(cancel)]), 0)
                emit.assert_any_call('done', {'ok': True})
        self.assertFalse(worker.FileCancellation(cancel).is_set())
        cancel.touch()
        self.assertTrue(worker.FileCancellation(cancel).is_set())


@unittest.skipUnless(os.getenv('DISPLAY') or os.name == 'nt', 'Tk display required')
class ExportedModelRecallGUI(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        self.root = tk.Tk()
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        self.gui = recall.RecallTestGUI(self.root, embedded=True)

    def test_engine_is_accepted_and_runtime_is_shown(self):
        with tempfile.TemporaryDirectory() as d:
            model = Path(d) / 'model.engine'
            model.write_bytes(b'fake')
            self.gui.model_var.set(str(model))
            self.gui.mode_var.set('Image folder')
            self.gui.source_var.set(d)
            self.gui.batch_var.set('1')
            self.assertIn('TensorRT runtime', self.gui.engine_runtime_var.get())
            self.assertEqual(self.gui._settings()['model'], str(model))
            self.assertEqual(self.gui._settings()['batch'], 1)
            self.assertEqual(self.gui._settings()['evaluation_python'], recall.load_selection() or sys.executable)

    def test_browse_engine_populates_settings_from_export_summary(self):
        with tempfile.TemporaryDirectory() as d:
            model = Path(d) / 'named.engine'
            model.write_bytes(b'fake')
            (model.parent / 'export_summary.json').write_text(json.dumps({
                'exported_model': str(model),
                'settings': {'imgsz': 1280, 'batch': 1, 'dynamic': False},
            }))
            with patch.object(recall.filedialog, 'askopenfilename', return_value=str(model)):
                self.gui._choose_model()
            self.assertEqual(self.gui.imgsz_var.get(), '1280')
            self.assertEqual(self.gui.batch_var.get(), '1')

    def test_engine_job_routes_outside_gui_process(self):
        with patch.object(self.gui, '_settings', return_value={'model': '/tmp/example.engine'}), \
             patch.object(recall, 'run_engine_evaluation', side_effect=lambda _s, emit, **kw: emit('cancelled', 'stop')) as run:
            self.gui._start(False)
            for _ in range(130):
                self.root.update()
                if not self.gui.busy:
                    break
                time.sleep(.01)
            run.assert_called_once()
            self.assertFalse(self.gui.busy)
