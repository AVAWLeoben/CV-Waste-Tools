"""Epoch ticks and live chart interactions (Tk tests run when a display exists)."""
import sys
import time
import tkinter as tk
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_charts import LiveMetricsGraph, epoch_ticks


class TickTests(unittest.TestCase):
    def test_ticks_are_integer_readable_and_include_endpoints(self):
        for start, end in ((1, 1), (1, 12), (1, 500), (75, 110), (2, 9999)):
            ticks = epoch_ticks(start, end)
            self.assertEqual(ticks[0], start)
            self.assertEqual(ticks[-1], end)
            self.assertEqual(ticks, sorted(set(ticks)))
            self.assertTrue(all(isinstance(tick, int) for tick in ticks))
            self.assertLessEqual(len(ticks), 14)


class CanvasInteractionTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError:
            self.skipTest('Tk display unavailable')
        self.root.geometry('1120x410')
        self.graph = LiveMetricsGraph(self.root)
        self.graph.pack(fill='both', expand=True)
        for epoch in range(1, 13):
            self.graph.record({'epoch': epoch, 'epochs': 500, 'metrics': {
                'train/box_loss': 0.9 - epoch * .02,
                'train/cls_loss': 2.1 - epoch * .05,
                'train/dfl_loss': 1.2 - epoch * .015,
                'val/box_loss': 1.0 - epoch * .025,
                'val/cls_loss': 2.0 - epoch * .055,
                'val/dfl_loss': 1.3 - epoch * .019,
                'metrics/precision(B)': .3 + epoch * .04,
                'metrics/recall(B)': .2 + epoch * .05,
                'metrics/mAP50(B)': .22 + epoch * .03,
                'metrics/mAP50-95(B)': .14 + epoch * .025,
            }})
        self.root.update()

    def tearDown(self):
        if hasattr(self, 'root'):
            self.root.destroy()

    def finish_animation(self):
        self.graph._transition['start'] = time.monotonic() - 1.0
        self.graph._animation_frame()
        self.root.update()

    def test_hover_halo_and_click_to_enlarge_and_close(self):
        self.graph.event_generate('<Motion>', x=120, y=80)
        self.root.update()
        self.assertEqual(self.graph.hovered, 0)
        self.assertEqual(self.graph.cget('cursor'), 'hand2')
        self.assertGreater(len(self.graph.find_all()), 200)
        self.assertGreaterEqual(sum(self.graph.type(item) == "oval"
                                    for item in self.graph.find_all()), 120)
        # All 12 actual epochs are marked across the ten plotted metric series.
        self.graph.event_generate('<Button-1>', x=120, y=80)
        self.root.update()
        self.assertIsNotNone(self.graph._transition)
        self.finish_animation()
        self.assertEqual(self.graph.expanded, 0)
        # The top-right × is exposed only in the expanded state.
        self.graph.event_generate('<Button-1>', x=self.graph.winfo_width() - 23, y=22)
        self.root.update()
        self.finish_animation()
        self.assertIsNone(self.graph.expanded)

    def test_reset_also_resets_enlargement_and_history(self):
        self.graph._animate(2, opening=True)
        self.finish_animation()
        self.assertEqual(self.graph.expanded, 2)
        self.graph.reset()
        self.assertIsNone(self.graph.expanded)
        self.assertEqual(self.graph.history.epochs, {})


if __name__ == '__main__':
    unittest.main()
