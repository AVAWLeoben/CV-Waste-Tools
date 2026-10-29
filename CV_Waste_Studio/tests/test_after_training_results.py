"""Tests of finished-run analysis without Ultralytics, GPU, or live GUI."""
from pathlib import Path
import json
import tempfile
import unittest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_results import load_run, discover_runs
from training_log import format_training_run_dir, parse_training_run_dir

HEADERS = ("epoch, time, train/box_loss, train/cls_loss, train/dfl_loss, "
           "metrics/precision(B), metrics/recall(B), metrics/mAP50(B), "
           "metrics/mAP50-95(B), val/box_loss, val/cls_loss, val/dfl_loss\n")
ROWS = ("1,6,1.3,2.6,0.8,0.2,0.1,0.3,0.2,1.5,2.7,1.2\n"
        "2,7,1.1,2.2,0.7,0.6,0.55,0.65,0.5,1.2,2.3,1.0\n"
        "3,8,0.9,2.0,0.6,0.55,0.51,0.60,0.48,1.0,2.1,0.9\n")

class ResultsTests(unittest.TestCase):
    def test_read_native_csv_and_find_peak(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "results.csv").write_text(HEADERS + ROWS, encoding="utf-8")
            data = load_run(folder)
            self.assertEqual(len(data["epochs"]), 3)
            self.assertEqual(data["peak"]["epoch"], 2)
            self.assertEqual(data["latest"]["epoch"], 3)
            self.assertEqual(data["epochs"][1]["metrics"]["metrics/mAP50-95(B)"], 0.5)
            self.assertEqual(data["epochs"][1]["metrics"]["train/box_loss"], 1.1)

    def test_missing_csv_is_not_faked(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_run(tmp)

    def test_partial_rows_and_nan_do_not_break_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "results.csv"
            f.write_text(HEADERS + ROWS + "4,9,NaN,,0.5,0.65,0.61,,0.52,,,,\n"
                         "5,9,bad,,,NaN,,,,,,,\n", encoding="utf-8")
            data = load_run(tmp)
            self.assertEqual(len(data["epochs"]), 4)
            self.assertEqual(data["latest"]["epoch"], 4)
            self.assertNotIn("train/box_loss", data["latest"]["metrics"])

    def test_run_folder_marker_roundtrip(self):
        folder = r"C:\Users\Public\Gerald Koinig\GreenPlast\training_runs\YOLO11l2"
        self.assertEqual(parse_training_run_dir(format_training_run_dir(folder)), folder)
        self.assertIsNone(parse_training_run_dir('STUDIO_TRAIN_RUN_DIR ""'))
        self.assertIsNone(parse_training_run_dir('STUDIO_TRAIN_RUN_DIR garbage'))

    def test_find_previous_run_from_saved_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            logs, project = home / "logs", home / "training_runs"
            logs.mkdir(); (project / "YOLO11l").mkdir(parents=True)
            (project / "YOLO11l" / "results.csv").write_text(HEADERS + ROWS)
            (logs / "train_20261029_123456_settings.json").write_text(json.dumps({"project": str(project)}))
            self.assertEqual(discover_runs(logs), [project / "YOLO11l"])

if __name__ == "__main__":
    unittest.main()

class CompletionNavigationTests(unittest.TestCase):
    def test_successful_training_opens_exact_run_from_worker_event(self):
        from greenplast_launcher import Studio
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as tmp:
            logfile = Path(tmp) / 'train.log'
            folder = str(Path(tmp) / 'training_runs' / 'experiment2')
            logfile.write_text(format_training_run_dir(folder) + '\n', encoding='utf-8')
            class Dummy:
                _training_output_line = Studio._training_output_line
                def __init__(self):
                    self.train_job = dict(log=logfile, offset=0, pending='',
                         process=type('Proc', (), {'poll': lambda self: 0})(),
                         stopping=False, cache_manifest=None, kind='train', run_dir=None)
                    self.train_button = Mock()
                    self.stop_button = Mock()
                    self.train_state = Mock()
                    self.results_panel = Mock()
                    self.results_panel.load_run.return_value = True
                    self.iteration_bar = Mock()
                    self.log_line = Mock()
                    self._show = Mock()
                    self.set_graph_visibility = Mock()
                    self.active = 'train'
            d = Dummy()
            Studio._poll_training(d)
            d.results_panel.load_run.assert_called_once_with(folder)
            d._show.assert_called_once_with('results')
            d.set_graph_visibility.assert_called_once_with(False)
            self.assertIsNone(d.train_job)

    def test_failed_training_does_not_navigate_to_results(self):
        from greenplast_launcher import Studio
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as tmp:
            logfile = Path(tmp) / 'failed.log'
            logfile.write_text('training failed\n')
            class Dummy:
                _training_output_line = Studio._training_output_line
                def __init__(self):
                    self.train_job = dict(log=logfile, offset=0, pending='',
                         process=type('Proc', (), {'poll': lambda self: 1})(),
                         stopping=False, cache_manifest=None, kind='train', run_dir=None)
                    self.train_button = Mock()
                    self.stop_button = Mock()
                    self.train_state = Mock()
                    self.results_panel = Mock()
                    self.log_line = Mock()
                    self._show = Mock()
                    self.set_graph_visibility = Mock()
                    self.active = 'train'
            d = Dummy()
            Studio._poll_training(d)
            d._show.assert_called_once_with('train')
            d.set_graph_visibility.assert_called_once_with(False)
