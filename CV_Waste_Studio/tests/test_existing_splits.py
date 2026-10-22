"""Run: python -m unittest discover -s tests -v (no GPU required)."""
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_and_split_tool_GUI as backend


class DatasetWorkflows(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "images").mkdir()
        (self.root / "labels").mkdir()
        self.yaml = self.root / "data.yaml"
        self.yaml.write_text("train: images/train\nval: images/val\nnames: [plastic, metal]\n",
                             encoding="utf-8")

    def test_sibling_folders_are_one_dataset(self):
        self.assertEqual(backend._autosplit_source_paths([
            self.root / "labels", self.root / "images", self.root]), [self.root])

    def test_existing_yaml_directory_discovery(self):
        result = backend.resolve_existing_dataset_yaml(self.root)
        self.assertEqual(result["yaml"], str(self.yaml))
        self.assertEqual(result["classes"], 2)

    def test_missing_validation_split_rejected(self):
        self.yaml.write_text("train: images/train\nnames: [plastic]\n")
        with self.assertRaisesRegex(ValueError, "val"):
            backend.resolve_existing_dataset_yaml(self.root)

    def test_existing_mode_never_calls_splitter(self):
        existing_bytes = self.yaml.read_bytes()
        invocations = []
        class FakeYOLO:
            def __init__(self, model):
                invocations.append(model)
            def train(self, **settings):
                invocations.append(settings)
                return "simulated-train"
        fake_ultra = types.ModuleType("ultralytics")
        fake_ultra.YOLO = FakeYOLO
        fake_torch = types.ModuleType("torch")
        with patch.object(backend, "autosplit_dataset", side_effect=AssertionError("Split was called")), \
             patch.object(backend, "_ensure_gpu_pytorch", return_value=None), \
             patch.object(backend, "_load_dependency", return_value=None), \
             patch.object(backend, "_select_training_device", return_value="cpu"), \
             patch.dict(sys.modules, {"ultralytics": fake_ultra, "torch": fake_torch}):
            results = backend.autosplit_and_train_model(
                existing_data_yaml=str(self.yaml), device="cpu", version="11",
                size="n", project=str(self.root / "results"), compact_progress=False,
            )
        self.assertIsNone(results["split"])
        self.assertEqual(results["training"], "simulated-train")
        self.assertEqual(invocations[1]["data"], str(self.yaml))
        self.assertEqual(self.yaml.read_bytes(), existing_bytes)


if __name__ == "__main__":
    unittest.main()
