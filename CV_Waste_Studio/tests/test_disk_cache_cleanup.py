"""Disk caching cleanup tests (simulate Ultralytics, no GPU required)."""
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_and_split_tool_GUI as backend


class DiskCacheCleanup(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.images = self.root / "image source"
        self.images.mkdir()
        self.train = self.images / "train image.jpg"
        self.val = self.images / "validation.png"
        for image in (self.train, self.val):
            image.write_bytes(b"original-image")
        self.label = self.images / "train image.txt"
        self.label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        self.train_manifest = self.root / "train.txt"
        self.val_manifest = self.root / "val.txt"
        self.train_manifest.write_text(self.train.as_posix() + "\n", encoding="utf-8")
        self.val_manifest.write_text(self.val.as_posix() + "\n", encoding="utf-8")
        self.yaml = self.root / "autosplit_data.yaml"
        self.yaml.write_text("path: .\ntrain: train.txt\nval: val.txt\nnames: [plastic]\n",
                             encoding="utf-8")

    def run_training(self, cache="disk", cleanup=True, fail=False, cache_manifest=None):
        images = [self.train, self.val]
        recorded = []

        class FakeYOLO:
            def __init__(self, model):
                pass

            def train(self, **kwargs):
                recorded.append(kwargs)
                if kwargs["cache"] == "disk":
                    for image in images:
                        image.with_suffix(".npy").write_bytes(b"generated-cache")
                if fail:
                    raise RuntimeError("simulated failure after disk cache creation")
                return "completed"

        fake_ultra = types.ModuleType("ultralytics")
        fake_ultra.YOLO = FakeYOLO
        with patch.object(backend, "autosplit_dataset", side_effect=AssertionError("split called")), \
             patch.object(backend, "_ensure_gpu_pytorch", return_value=None), \
             patch.object(backend, "_load_dependency", return_value=None), \
             patch.object(backend, "_select_training_device", return_value="cpu"), \
             patch.dict(sys.modules, {"ultralytics": fake_ultra, "torch": types.ModuleType("torch")}):
            return backend.autosplit_and_train_model(
                existing_data_yaml=str(self.yaml), cache=cache, workers=8,
                cleanup_disk_cache=cleanup, cache_cleanup_manifest=cache_manifest,
                project=str(self.root / "results"), compact_progress=False,
            ), recorded

    def test_disk_caches_deleted_after_success_sources_preserved(self):
        result, invocations = self.run_training()
        self.assertEqual(result["training"], "completed")
        self.assertEqual((invocations[0]["cache"], invocations[0]["workers"]), ("disk", 8))
        self.assertFalse(self.train.with_suffix(".npy").exists())
        self.assertFalse(self.val.with_suffix(".npy").exists())
        self.assertEqual(self.train.read_bytes(), b"original-image")
        self.assertEqual(self.label.read_text(), "0 0.5 0.5 0.2 0.2\n")

    def test_disk_caches_deleted_after_training_error(self):
        with self.assertRaisesRegex(RuntimeError, "simulated failure"):
            self.run_training(fail=True)
        self.assertFalse(self.train.with_suffix(".npy").exists())
        self.assertFalse(self.val.with_suffix(".npy").exists())

    def test_preexisting_disk_cache_not_deleted(self):
        previous = self.train.with_suffix(".npy")
        previous.write_bytes(b"existing-cache")
        self.run_training()
        self.assertTrue(previous.exists())
        self.assertFalse(self.val.with_suffix(".npy").exists())

    def test_disable_cleanup_preserves_generated_caches(self):
        self.run_training(cleanup=False)
        self.assertTrue(self.train.with_suffix(".npy").exists())
        self.assertTrue(self.val.with_suffix(".npy").exists())

    def test_manifest_cleanup_for_forced_stop(self):
        manifest = self.root / "job_manifest.json"
        candidates = backend._disk_cache_candidates(self.yaml, {
            "path": str(self.root), "train": "train.txt", "val": "val.txt"
        })
        self.assertEqual(candidates, {
            self.train.with_suffix(".npy"), self.val.with_suffix(".npy")
        })
        backend._save_disk_cache_manifest(manifest, candidates)
        self.assertTrue(manifest.is_file())
        for cache in candidates:
            cache.write_bytes(b"disk-cache")
        count, errors = backend.cleanup_disk_cache_manifest(manifest)
        self.assertEqual((count, errors), (2, []))
        self.assertFalse(manifest.exists())
        for cache in candidates:
            self.assertFalse(cache.exists())

    def test_no_disk_cleanup_for_ram(self):
        _, invocations = self.run_training(cache="ram")
        self.assertEqual(invocations[0]["cache"], "ram")
        self.assertFalse(self.train.with_suffix(".npy").exists())


if __name__ == "__main__":
    unittest.main()
