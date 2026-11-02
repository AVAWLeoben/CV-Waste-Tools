"""Model comparison contract tests; all inference/training is mocked."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from model_comparison_worker import (validate_settings, parse_event, EVENT_PREFIX,
                                     _train_config, _save_summary, run_comparison)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "data.yaml"
        self.data.write_text("path: .\ntrain: train/images\nval: val/images\ntest: test/images\nnames: [plastic, metal]\n",
                             encoding="utf-8")
        self.config = {"existing_data_yaml": str(self.data),
                       "models": [{"version": "11", "size": "n"},
                                  {"version": "11", "size": "s"}],
                       "output_dir": str(self.root / "comparisons"),
                       "batch": 16, "eval_batch": 4, "workers": 8,
                       "cache": "disk", "imgsz": 1280, "epochs": 2,
                       "cleanup_disk_cache": True}

    def test_requires_test_split(self):
        self.data.write_text("train: images/train\nval: images/val\nnames: [object]\n")
        with self.assertRaisesRegex(ValueError, "test"):
            validate_settings(self.config)

    def test_rejects_bad_model_selection(self):
        for models in ([], [{"version": "11", "size": "n"}],
                       [{"version": "11", "size": "bad"}, {"version": "11", "size": "n"}]):
            with self.subTest(models=models), self.assertRaises(ValueError):
                validate_settings({**self.config, "models": models})

    def test_invalid_worker_and_batch_rejected(self):
        for key, value in (("workers", -1), ("workers", True), ("batch", 0),
                           ("eval_batch", 0), ("imgsz", -1)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_settings({**self.config, key: value})

    def test_pretrained_and_custom_weights_selection(self):
        weights = self.root / "best.pt"
        weights.write_bytes(b"mock")
        cfg = validate_settings({**self.config, "models": [
            {"model_path": str(weights)}, {"version": "8", "size": "m"}]})
        self.assertEqual(cfg["models"][0]["model_path"], str(weights))
        self.assertEqual(cfg["models"][1]["label"], "yolov8m")
        self.assertEqual(cfg["workers"], 8)
        self.assertEqual(cfg["cache"], "disk")

    def test_train_config_reuses_native_training_api_settings(self):
        cfg = validate_settings(self.config)
        train = _train_config(cfg, cfg["models"][0], self.root / "output", "name", 8)
        self.assertEqual(train["existing_data_yaml"], str(self.data))
        self.assertIsNone(train["dataset_paths"])
        self.assertEqual(train["workers"], 8)
        self.assertEqual(train["batch"], 8)
        self.assertEqual(train["cache"], "disk")
        self.assertTrue(train["cleanup_disk_cache"])

    def test_event_parsing_handles_malformed(self):
        self.assertIsNone(parse_event("plain Ultralytics log"))
        self.assertIsNone(parse_event(EVENT_PREFIX + "not json"))
        event = parse_event(EVENT_PREFIX + '{"kind":"result","index":1}')
        self.assertEqual(event["index"], 1)

    def test_save_summary_keeps_partial_runs(self):
        report_dir = self.root / "report"
        report_dir.mkdir()
        cfg = validate_settings(self.config)
        records = [{"model": "yolo11n", "status": "completed", "map50_95": .7}]
        _save_summary(report_dir, records, "in_progress", cfg)
        self.assertEqual(json.loads((report_dir / "comparison.json").read_text())["models"], records)
        self.assertIn("yolo11n", (report_dir / "comparison.csv").read_text(encoding="utf-8-sig"))

    def test_mock_two_model_sequential_training_then_test(self):
        calls = []
        def simulate(cmd, logfile, update=None):
            calls.append((Path(cmd[2]).name, logfile))
            cfg = json.loads(Path(cmd[3]).read_text(encoding="utf-8"))
            logfile.write_text("Mock worker ran\n")
            if Path(cmd[2]).name == "run_training_job.py":
                run = Path(cfg["project"]) / cfg["run_name"]
                (run / "weights").mkdir(parents=True)
                (run / "weights" / "best.pt").write_bytes(b"mock")
                (run / "results.csv").write_text("epoch,metrics/mAP50-95(B)\n1,0.7\n")
                if update:
                    update("batch", epoch=1, epochs=2, iteration=2, iterations=4)
                return 0, {"run_dir": str(run), "oom": False}
            Path(cfg["result_json"]).write_text(json.dumps({
                "precision": .9, "recall": .8, "map50": .85, "map50_95": .71,
                "save_dir": str(self.root / "test")}), encoding="utf-8")
            return 0, {"oom": False}
        with mock.patch("model_comparison_worker._run_child", side_effect=simulate), \
             mock.patch("model_comparison_worker.emit") as events:
            status = run_comparison(self.config)
        self.assertEqual(status, "complete")
        self.assertEqual([c[0] for c in calls], ["run_training_job.py", "run_model_comparison_eval.py",
                                                   "run_training_job.py", "run_model_comparison_eval.py"])
        results_events = [c.kwargs["result"] for c in events.call_args_list
                          if c.args and c.args[0] == "result"]
        self.assertEqual(len(results_events), 2)
        self.assertTrue(all(x["map50_95"] == .71 for x in results_events))
        report = next((self.root / "comparisons").glob("Comparison_*/comparison.json"))
        self.assertEqual(json.loads(report.read_text())["status"], "complete")

    def test_oom_retries_at_reduced_batch(self):
        batches = []
        def fake_child(cmd, log, update=None):
            cfg = json.loads(Path(cmd[3]).read_text())
            if cmd[2].endswith("run_training_job.py"):
                batches.append(cfg["batch"])
                if len(batches) == 1:
                    return 1, {"run_dir": None, "oom": True}
                run = Path(cfg["project"]) / cfg["run_name"]
                (run / "weights").mkdir(parents=True)
                (run / "weights" / "best.pt").write_bytes(b"mock")
                (run / "results.csv").write_text("epoch,metrics/mAP50-95(B)\n1,0.4\n")
                return 0, {"run_dir": str(run), "oom": False}
            Path(cfg["result_json"]).write_text(json.dumps({"precision": .6, "recall": .7,
                "map50": .75, "map50_95": .55, "save_dir": "test"}))
            return 0, {"oom": False}
        with mock.patch("model_comparison_worker._run_child", side_effect=fake_child), \
             mock.patch("model_comparison_worker.emit"):
            self.assertEqual(run_comparison(self.config), "complete")
        self.assertEqual(batches, [16, 8, 16])


if __name__ == "__main__":
    unittest.main()

class NativeEvaluationTests(unittest.TestCase):
    def test_test_split_invokes_ultralytics_val_and_records_metrics(self):
        import sys
        import types
        from run_model_comparison_eval import evaluate
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = types.SimpleNamespace(box=types.SimpleNamespace(
                mp=.91, mr=.82, map50=.95, map=.73), save_dir=root / "eval")
            model = mock.MagicMock()
            model.val.return_value = result
            fake_ultra = types.ModuleType("ultralytics")
            fake_ultra.YOLO = mock.MagicMock(return_value=model)
            fake_torch = types.ModuleType("torch")
            fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
            cfg = {"weights": str(root / "best.pt"), "dataset_yaml": str(root / "data.yaml"),
                   "imgsz": 1280, "batch": 4, "workers": 8, "device": "cpu",
                   "project": str(root), "name": "test_run",
                   "result_json": str(root / "result.json")}
            with mock.patch.dict(sys.modules, {"ultralytics": fake_ultra, "torch": fake_torch}):
                recorded = evaluate(cfg)
            model.val.assert_called_once_with(data=cfg["dataset_yaml"], split="test",
                imgsz=1280, batch=4, device="cpu", workers=8, project=cfg["project"],
                name="test_run", plots=True)
            self.assertEqual(recorded["map50_95"], .73)
            self.assertEqual(json.loads(Path(cfg["result_json"]).read_text())["precision"], .91)
