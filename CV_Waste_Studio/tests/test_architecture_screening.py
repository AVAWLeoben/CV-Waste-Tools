"""Architecture screening: validation-based shortlist without touching held-out test."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

from model_comparison_worker import validate_settings, run_comparison
from train_and_split_tool_GUI import _build_model_filename


class ScreeningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "data.yaml"
        self.data.write_text("path: .\ntrain: train/images\nval: val/images\nnames: [a,b]\n")
        self.settings = {"existing_data_yaml": str(self.data),
                         "models": [{"version": "10", "size": "m"},
                                    {"version": "12", "size": "l"}],
                         "screening_mode": True, "epochs": 3, "cache": "disk",
                         "output_dir": str(self.root / "cmp")}

    def test_generation_filenames(self):
        self.assertEqual(_build_model_filename("10", "m"), "yolov10m.pt")
        self.assertEqual(_build_model_filename("12", "l"), "yolo12l.pt")
        self.assertEqual(_build_model_filename("26", "x"), "yolo26x.pt")

    def test_screening_accepts_yaml_without_test_but_full_comparison_does_not(self):
        cfg = validate_settings(self.settings)
        self.assertTrue(cfg["screening_mode"])
        self.assertEqual([m["label"] for m in cfg["models"]], ["yolov10m", "yolo12l"])
        with self.assertRaisesRegex(ValueError, "test"):
            validate_settings({**self.settings, "screening_mode": False})

    def test_screening_requires_bool_flag(self):
        with self.assertRaisesRegex(ValueError, "true or false"):
            validate_settings({**self.settings, "screening_mode": "true"})

    def test_sequential_screening_calls_validation_not_test(self):
        splits = []
        def child(command, logfile, update=None):
            cfg = json.loads(Path(command[3]).read_text())
            if command[2].endswith("run_training_job.py"):
                run = Path(cfg["project"]) / cfg["run_name"]
                (run / "weights").mkdir(parents=True)
                (run / "weights" / "best.pt").write_bytes(b"checkpoint")
                (run / "results.csv").write_text(
                    "epoch,metrics/mAP50-95(B)\n1,0.6\n2,0.7\n", encoding="utf-8")
                return 0, {"run_dir": str(run), "oom": False}
            splits.append(cfg.get("split"))
            Path(cfg["result_json"]).write_text(json.dumps({"precision": .85, "recall": .72,
                 "map50": .77, "map50_95": .65, "save_dir": str(self.root / "validation")}))
            return 0, {"oom": False}
        with mock.patch("model_comparison_worker._run_child", side_effect=child), \
             mock.patch("model_comparison_worker.emit"):
            self.assertEqual(run_comparison(self.settings), "complete")
        self.assertEqual(splits, ["val", "val"])
        report = next((self.root / "cmp").glob("Comparison_*/comparison.json"))
        result = json.loads(report.read_text())
        self.assertTrue(result["screening_mode"])
        self.assertEqual(result["evaluated_split"], "val")
        self.assertTrue(all(r["evaluated_split"] == "val" for r in result["models"]))
        self.assertTrue(all(r["weights_mb"] is not None for r in result["models"]))
        self.assertTrue(all(r["train_seconds"] is not None for r in result["models"]))

    def test_validation_child_uses_selected_split(self):
        from run_model_comparison_eval import evaluate
        weights = self.root / "weights.pt"
        weights.write_bytes(b"mock")
        fake_result = types.SimpleNamespace(box=types.SimpleNamespace(
            mp=.8, mr=.7, map50=.9, map=.6), save_dir=self.root)
        fake_model = mock.Mock()
        fake_model.val.return_value = fake_result
        fake_ultra = types.ModuleType("ultralytics")
        fake_ultra.YOLO = mock.Mock(return_value=fake_model)
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        cfg = {"weights": str(weights), "dataset_yaml": str(self.data), "split": "val",
               "imgsz": 640, "batch": 2, "device": "cpu", "workers": 4,
               "project": str(self.root), "name": "val", "result_json": str(self.root / "result.json")}
        with mock.patch.dict(sys.modules, {"ultralytics": fake_ultra, "torch": fake_torch}):
            recorded = evaluate(cfg)
        self.assertEqual(recorded["map50_95"], .6)
        self.assertEqual(fake_model.val.call_args.kwargs["split"], "val")


if __name__ == "__main__":
    unittest.main()
