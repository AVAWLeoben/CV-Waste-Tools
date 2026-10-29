"""Live graph protocol and state tests without a GUI or GPU."""
import math
import sys
from pathlib import Path
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_log import (epoch_metrics_from_trainer, format_epoch_metrics,
                          parse_epoch_metrics)
from training_charts import EpochHistory
from greenplast_launcher import Studio
import train_and_split_tool_GUI as backend


class MetricsProtocolTests(unittest.TestCase):
    def test_snapshot_and_round_trip(self):
        trainer = types.SimpleNamespace(
            epoch=3, epochs=20, loss_names=("box_loss", "cls_loss", "dfl_loss"),
            tloss=[0.7, 1.1, 0.8], metrics={
                "val/box_loss": 0.6, "val/cls_loss": 0.95, "val/dfl_loss": 0.77,
                "metrics/precision(B)": 0.84, "metrics/recall(B)": 0.72,
                "metrics/mAP50(B)": 0.8, "metrics/mAP50-95(B)": 0.6,
                "fitness": 0.8,
            })
        event = epoch_metrics_from_trainer(trainer)
        self.assertEqual(event["epoch"], 4)
        self.assertEqual(event["metrics"]["train/cls_loss"], 1.1)
        self.assertEqual(event["metrics"]["val/dfl_loss"], 0.77)
        self.assertNotIn("fitness", event["metrics"])
        self.assertEqual(parse_epoch_metrics(format_epoch_metrics(event)), event)

    def test_invalid_metrics_are_rejected(self):
        for value in ('{"epoch":0,"epochs":10,"metrics":{}}',
                      '{"epoch":2,"epochs":1,"metrics":{}}',
                      '{"epoch":1,"epochs":2,"metrics":{"unknown":1}}',
                      '{"epoch":1,"epochs":2,"metrics":{"train/box_loss":NaN}}',
                      'not json'):
            self.assertIsNone(parse_epoch_metrics("STUDIO_TRAIN_METRICS " + value))
        self.assertIsNone(parse_epoch_metrics("Epoch 2 done"))

    def test_existing_and_latest_epoch_history(self):
        history = EpochHistory()
        history.add({"epoch": 1, "epochs": 600,
                     "metrics": {"train/box_loss": 0.8}})
        history.add({"epoch": 2, "epochs": 600,
                     "metrics": {"train/box_loss": 0.7,
                                 "metrics/mAP50(B)": 0.5}})
        history.add({"epoch": 2, "epochs": 600,
                     "metrics": {"metrics/mAP50(B)": 0.6}})
        self.assertEqual(history.series("train/box_loss"), [(1, 0.8), (2, 0.7)])
        self.assertEqual(history.series("metrics/mAP50(B)"), [(2, 0.6)])
        self.assertEqual(history.total, 600)
        history.reset()
        self.assertFalse(history.epochs)

    def test_graph_events_do_not_spam_activity_log(self):
        class DummyGraph:
            def __init__(self): self.events = []
            def record(self, value): self.events.append(value)

        class DummyStudio:
            training_graph = DummyGraph()
            def __init__(self): self.lines = []
            def log_line(self, key, line): self.lines.append((key, line))

        studio = DummyStudio()
        line = format_epoch_metrics({"epoch": 1, "epochs": 20,
                                     "metrics": {"train/box_loss": .7}})
        Studio._training_output_line(studio, line)
        self.assertEqual(len(studio.training_graph.events), 1)
        self.assertEqual(studio.lines, [])

    def test_compact_false_still_reports_metrics_without_changing_train_api(self):
        class Model:
            def __init__(self):
                self.callbacks = {k: [] for k in (
                    "on_train_epoch_start", "on_train_batch_end", "on_fit_epoch_end")}
            def add_callback(self, name, fn):
                self.callbacks[name].append(fn)
            def train(self, **kwargs):
                trainer = types.SimpleNamespace(epoch=0, epochs=5,
                    tloss=[.25, .5, .75], loss_names=["box_loss", "cls_loss", "dfl_loss"],
                    metrics={"metrics/mAP50(B)": .6})
                for callback in self.callbacks["on_fit_epoch_end"]:
                    callback(trainer)
                return kwargs
        model = Model()
        from io import StringIO
        from contextlib import redirect_stdout
        capture = StringIO()
        with redirect_stdout(capture):
            result = backend._train_with_compact_progress(model, {"workers": 8, "cache": "disk"}, False)
        self.assertEqual(result, {"workers": 8, "cache": "disk"})
        self.assertTrue(any(parse_epoch_metrics(line) for line in capture.getvalue().splitlines()))
        self.assertFalse(any(model.callbacks.values()))

if __name__ == "__main__":
    unittest.main()
