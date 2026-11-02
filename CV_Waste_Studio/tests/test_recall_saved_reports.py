"""Persistent Recall@Precision results, summary PNG, and GUI reopening."""
from pathlib import Path
import csv
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_model_recall_at_precision as recall
from recall_report import (REPORT_NAME, load_evaluation, saved_evaluations,
                           save_evaluation_report)


class RecallSavedResults(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.dest = self.root / "Recall@Precision_Test"
        self.dest.mkdir()
        self.model = self.root / "best.pt"
        self.model.write_bytes(b"mock model")

    def _example(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "food_precision_recall_by_confidence.csv").open("w", newline="") as out:
            writer = csv.writer(out)
            writer.writerow(["confidence", "precision", "recall", "TP", "FP", "FN"])
            writer.writerows([[.01, .20, .95, 100, 400, 5], [.3, .76, .88, 90, 25, 15],
                              [.82, .95, .81, 85, 4, 20], [.99, 1., .12, 12, 0, 93]])
        summary = dict(model=str(self.model), dataset="dataset.yaml", split="test",
                       class_id=0, class_name="Waste", image_count=1137,
                       target_precision=.95, target_reached=True,
                       confidence_threshold=.82162, achieved_precision=.9501,
                       recall_at_precision=.8134, true_positives=2493,
                       false_positives=131, false_negatives=572,
                       iou_match_threshold=.5, nms_iou_threshold=.7,
                       output_directory=str(directory))
        (directory / "operating_point_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return summary

    def test_saved_results_sort_and_ignore_incomplete_directories(self):
        old = self.dest / "precision_recall_20261101_100000_000000"
        new = self.dest / "precision_recall_20261102_100000_000000"
        self._example(old)
        self._example(new)
        (self.dest / "precision_recall_20261103_100000_000000").mkdir()
        self.assertEqual(saved_evaluations(self.dest), [new, old])
        self.assertEqual(saved_evaluations(self.root / "does_not_exist"), [])
        loaded = load_evaluation(new)
        self.assertEqual(loaded["class_name"], "Waste")
        self.assertEqual(loaded["output_directory"], str(new.resolve()))
        self.assertEqual(load_evaluation(new / "operating_point_summary.json")["class_id"], 0)
        with self.assertRaises(FileNotFoundError):
            load_evaluation(self.dest)

    def test_report_has_expected_dimensions_and_reopens_without_inference(self):
        from PIL import Image
        folder = self.dest / "precision_recall_20261102_100000_000000"
        summary = self._example(folder)
        report = save_evaluation_report(load_evaluation(folder))
        self.assertEqual(report.name, REPORT_NAME)
        with Image.open(report) as image:
            self.assertEqual(image.size, (1440, 900))
            self.assertEqual(image.format, "PNG")
        self.assertGreater(report.stat().st_size, 20000)

    def test_evaluation_automatically_saves_report_and_json_reference(self):
        images = [self.root / "images" / "001.jpg"]
        images[0].parent.mkdir()
        images[0].write_bytes(b"fake")
        prediction = dict(image_index=np.array([0, 0], dtype=np.int32),
                          classes=np.array([0, 0], dtype=np.int32),
                          confidence=np.array([.95, .5], dtype=np.float32),
                          xyxy=np.asarray([[.1, .1, .5, .5], [.5, .5, .8, .8]], dtype=np.float32))
        settings = dict(model=str(self.model), mode="Image folder", source=str(images[0].parent),
                        output=str(self.dest), split="test", precision=.95, food_class=0,
                        imgsz=640, batch=8, match_iou=.5, nms_iou=.7, max_det=300,
                        save_plots=False, reuse_cache=False)
        emitted = []
        with patch.object(recall, "collect_images", return_value=images), \
             patch.object(recall, "predict_and_cache", return_value=(prediction,
                         {"class_names": {"0": "Waste"}, "device": "cpu", "inference_seconds": 0.1})), \
             patch.object(recall, "ground_truth", return_value=[np.asarray([[0, .1, .1, .5, .5]], dtype=np.float32)]):
            summary = recall.evaluate(settings, lambda kind, value: emitted.append((kind, value)))
        report = Path(summary["report_png"])
        self.assertTrue(report.is_file())
        self.assertEqual(report.parent, Path(summary["output_directory"]))
        saved = load_evaluation(report.parent)
        self.assertEqual(saved["report_png"], str(report))
        self.assertTrue(any("Saved evaluation graphic" in value for kind, value in emitted if kind == "log"))


@unittest.skipUnless(os.getenv("DISPLAY") or os.name == "nt", "GUI test needs display")
class RecallSavedResultsGUI(unittest.TestCase):
    def test_reopens_existing_evaluation_without_starting_inference(self):
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "precision_recall_20261102_100000_000000"
            folder.mkdir()
            summary = dict(output_directory=str(folder), target_reached=True,
                           target_precision=.95, class_name="Metal", class_id=1,
                           confidence_threshold=.78, achieved_precision=.953,
                           recall_at_precision=.76, true_positives=22,
                           false_positives=1, false_negatives=7)
            (folder / "operating_point_summary.json").write_text(json.dumps(summary))
            gui = recall.RecallTestGUI(root, embedded=True)
            gui.output_var.set(directory)
            gui._refresh_previous_runs()
            self.assertEqual(gui.recent_choice.current(), 0)
            with patch.object(recall, "evaluate") as predict, \
                 patch.object(recall, "save_evaluation_report", side_effect=lambda x: (folder / REPORT_NAME).touch()):
                gui._load_selected_result()
                predict.assert_not_called()
            self.assertEqual(gui.last_results_dir, str(folder.resolve()))
            self.assertIn("Metal", gui.result_var.get())
            self.assertEqual(str(gui.open_results_button.cget("state")), "normal")
            self.assertEqual(str(gui.view_report_button.cget("state")), "normal")
            # The existing file's metric payload was not changed by reopening.
            self.assertNotIn("report_png", json.loads((folder / "operating_point_summary.json").read_text()))
