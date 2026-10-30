"""Checkpoint-aligned live best-epoch display without changing YOLO training."""
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import tkinter as tk
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_log import (BestEpochTracker, epoch_metrics_from_trainer,
                          format_epoch_metrics, parse_epoch_metrics)
from greenplast_launcher import Studio


def event(epoch, score, *, is_best=True, map95=None):
    return {"epoch": epoch, "epochs": 50, "fitness": score,
            "checkpoint_best": is_best,
            "metrics": {"metrics/precision(B)": .91,
                        "metrics/recall(B)": .78,
                        "metrics/mAP50(B)": .86,
                        "metrics/mAP50-95(B)": score if map95 is None else map95,
                        "val/cls_loss": 1.01,
                        "train/box_loss": .45}}


class BestEpochSelectionTests(unittest.TestCase):
    def test_native_ultralytics_checkpoint_selection_and_all_metrics(self):
        tracker = BestEpochTracker()
        self.assertIn("Waiting", tracker.display()[0])
        self.assertTrue(tracker.record(event(1, .54)))
        # Validation mAP can rise even though native fitness does not improve.
        self.assertFalse(tracker.record(event(2, .50, is_best=False, map95=.95)))
        self.assertEqual(tracker.best["epoch"], 1)
        self.assertTrue(tracker.record(event(3, .63)))
        title, metrics, losses = tracker.display()
        self.assertIn("3/50", title)
        self.assertIn("best.pt", title)
        self.assertIn("0.6300", title)
        self.assertIn("Precision 91.0%", metrics)
        self.assertIn("Recall 78.0%", metrics)
        self.assertIn("mAP50–95 63.0%", metrics)
        self.assertIn("Train box", losses)
        self.assertIn("Val cls", losses)
        self.assertFalse(tracker.record(event(3, .63)))
        # Ultralytics overwrites best.pt on a tied fitness, so report latest.
        self.assertTrue(tracker.record(event(4, .63)))
        self.assertEqual(tracker.best["epoch"], 4)
        tracker.reset()
        self.assertIsNone(tracker.best)

    def test_native_protocol_from_trainer_and_invalid_events(self):
        trainer = types.SimpleNamespace(
            epoch=1, epochs=10, fitness=.56, best_fitness=.56,
            loss_names=["box_loss", "cls_loss", "dfl_loss"],
            tloss=[.1, .2, .3],
            metrics={"metrics/mAP50-95(B)": .56})
        snapshot = epoch_metrics_from_trainer(trainer)
        self.assertTrue(snapshot["checkpoint_best"])
        self.assertEqual(snapshot["fitness"], .56)
        self.assertEqual(parse_epoch_metrics(format_epoch_metrics(snapshot)), snapshot)
        trainer.fitness = .51
        self.assertFalse(epoch_metrics_from_trainer(trainer)["checkpoint_best"])
        for invalid in ({**snapshot, "fitness": float('nan')},
                        {**snapshot, "checkpoint_best": "yes"},
                        {**snapshot, "fitness": None}):
            import json
            raw = "STUDIO_TRAIN_METRICS " + json.dumps(invalid)
            self.assertIsNone(parse_epoch_metrics(raw))

    def test_old_events_fall_back_to_peak_map95(self):
        tracker = BestEpochTracker()
        def old(epoch, value):
            return {"epoch": epoch, "epochs": 6,
                    "metrics": {"metrics/mAP50-95(B)": value}}
        self.assertTrue(tracker.record(old(1, .3)))
        self.assertFalse(tracker.record(old(2, .2)))
        self.assertTrue(tracker.record(old(3, .45)))
        self.assertEqual(tracker.best["epoch"], 3)
        self.assertIn("peak validation", tracker.display()[0])

    def test_gui_updates_best_epoch_without_latest_epoch_overwrite(self):
        class SimpleVar:
            def set(self, v): self.value = v
        class DummyGraph:
            def record(self, value): pass
        class Dummy:
            best_epoch_tracker = BestEpochTracker()
            training_graph = DummyGraph()
            best_heading = SimpleVar()
            best_scores = SimpleVar()
            best_losses = SimpleVar()
            def __init__(self): self.lines = []
            def log_line(self, key, text): self.lines.append(text)
            _refresh_best_epoch = Studio._refresh_best_epoch
        view = Dummy()
        Studio._training_output_line(view, format_epoch_metrics(event(1, .50)))
        Studio._training_output_line(view, format_epoch_metrics(event(2, .48, is_best=False)))
        self.assertIn("1/50", view.best_heading.value)
        Studio._training_output_line(view, format_epoch_metrics(event(3, .57)))
        self.assertIn("3/50", view.best_heading.value)
        self.assertEqual(sum(s.startswith('Best epoch updated') for s in view.lines), 2)

    def test_banner_visible_during_monitoring(self):
        try:
            with patch('greenplast_launcher._same_executable', return_value=True):
                app = Studio(prepared=('conda', Path(sys.prefix),
                                       {'torch': 'test', 'cuda': False}),
                             allow_multiple=True)
        except tk.TclError as exc:
            self.skipTest(str(exc))
        try:
            app._poll_training = lambda: None
            app.show('train')
            app.train_job = {'process': object(), 'kind': 'train'}
            app.train_state.set('Running training')
            app.set_graph_visibility(True)
            app._show('train')
            app._training_output_line(format_epoch_metrics(event(3, .664)))
            app.root.update()
            self.assertEqual(app.best_epoch_banner.winfo_manager(), 'pack')
            self.assertIn('3/50', app.best_heading.get())
            self.assertGreater(app.training_graph.winfo_height(), 200)
        finally:
            app.root.destroy()


if __name__ == '__main__':
    unittest.main()
