"""Fourth-tab smoke test, safe scan and explicit in-place confirmation."""
from pathlib import Path
import sys
import tempfile
import time
import tkinter as tk
from tkinter import ttk
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_and_split_tool_GUI import _all_in_one_gui


class SanitiseUITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source = self.base/'dataset'
        (self.source/'images').mkdir(parents=True)
        (self.source/'labels').mkdir(parents=True)
        (self.source/'images/a.jpg').write_bytes(b'fake')
        (self.source/'labels/a.txt').write_text('')
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest('No graphical display: ' + str(exc))
        self.addCleanup(self.root.destroy)
        self.root.geometry('1250x900')
        self.calls = []
        self.form, self.train = _all_in_one_gui(self.root, on_submit=self.calls.append,
            sanitise_options={'logs_dir': self.base/'logs', 'can_start': lambda: True})
        self.panel = self.form.sanitise_panel
        self.panel.source.set(str(self.source))
        self.panel.output.set(str(self.base/'cleaned'))
        self.root.update()

    def test_fourth_tab_and_preview_without_deleting(self):
        notebook = next(w for w in self.form.winfo_children() if isinstance(w, ttk.Notebook))
        self.assertEqual(notebook.tab(3, 'text'), '4. Sanitise Datasets')
        notebook.select(3)
        self.root.update()
        self.assertIn('Scan', self.train.cget('text'))
        self.train.invoke()
        deadline = time.monotonic() + 5
        while self.panel.busy and time.monotonic() < deadline:
            self.root.update()
            self.panel.poll()
            time.sleep(0.02)
        self.panel.poll()
        self.assertFalse(self.panel.busy)
        self.assertEqual(len(self.panel.summary['remove']), 1)
        self.assertTrue((self.source/'images/a.jpg').exists())
        self.assertFalse(self.calls)

    def test_changed_settings_invalidate_preview(self):
        self.panel.preview_settings = self.panel._settings()
        self.panel.summary = {'remove': [], 'total': 1}
        self.panel.plan_path = self.base/'plan.json'
        self.panel.apply_btn.configure(state='normal')
        self.panel.conf.set('0.8')
        self.assertIsNone(self.panel.plan_path)
        self.assertEqual(str(self.panel.apply_btn['state']), 'disabled')

    def test_saves_all_preview_candidates_to_csv(self):
        import csv
        self.panel.plan_path = self.base/'scan_plan.json'
        self.panel.summary = {'total': 1, 'remove': [
            {'image': 'images/a.jpg', 'label': 'labels/a.txt'}]}
        path = self.base/'review.csv'
        with patch('dataset_sanitise_ui.filedialog.asksaveasfilename', return_value=str(path)):
            self.panel.save_candidates()
        with path.open(encoding='utf-8-sig', newline='') as inp:
            rows = list(csv.reader(inp))
        self.assertEqual(rows[1], ['images/a.jpg', 'labels/a.txt'])

    def test_settings_change_during_scan_invalidates_preview(self):
        self.panel.start_scan()
        self.assertTrue(self.panel.busy)
        self.panel.action.set('inplace')
        deadline = time.monotonic() + 5
        while self.panel.busy and time.monotonic() < deadline:
            self.root.update()
            self.panel.poll()
            time.sleep(0.02)
        self.panel.poll()
        self.assertFalse(self.panel.busy)
        self.assertIsNone(self.panel.plan_path)
        self.assertEqual(str(self.panel.apply_btn['state']), 'disabled')
        self.assertTrue((self.source/'images/a.jpg').exists())

    def test_inplace_confirm_runs_worker_and_removes_pair(self):
        self.panel.action.set('inplace')
        self.panel.start_scan()
        deadline = time.monotonic() + 5
        while self.panel.busy and time.monotonic() < deadline:
            self.root.update()
            self.panel.poll()
            time.sleep(0.02)
        self.panel.poll()
        with patch('dataset_sanitise_ui.simpledialog.askstring', return_value='DELETE'):
            self.panel.apply()
        self.assertTrue(self.panel.busy)
        deadline = time.monotonic() + 5
        while self.panel.busy and time.monotonic() < deadline:
            self.root.update()
            self.panel.poll()
            time.sleep(0.02)
        self.panel.poll()
        self.assertFalse(self.panel.busy)
        self.assertFalse((self.source/'images/a.jpg').exists())
        self.assertFalse((self.source/'labels/a.txt').exists())

    def test_inplace_decline_leaves_files_untouched(self):
        self.panel.action.set('inplace')
        self.panel.start_scan()
        deadline = time.monotonic() + 5
        while self.panel.busy and time.monotonic() < deadline:
            self.root.update()
            self.panel.poll()
            time.sleep(0.02)
        self.panel.poll()
        with patch('dataset_sanitise_ui.simpledialog.askstring', return_value=None):
            self.panel.apply()
        self.assertTrue((self.source/'images/a.jpg').exists())
        self.assertTrue((self.source/'labels/a.txt').exists())
        self.assertFalse(self.panel.busy)


if __name__ == '__main__':
    unittest.main()
