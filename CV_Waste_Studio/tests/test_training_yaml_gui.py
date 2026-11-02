"""Tk smoke test for training YAML preview and application."""
import pathlib
import sys
import tempfile
import tkinter as tk
from tkinter import ttk
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from train_and_split_tool_GUI import _all_in_one_gui


def descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from descendants(child)


class YAMLImportGUI(unittest.TestCase):
    def test_file_selection_previews_and_applies_advanced_settings(self):
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / 'args.yaml'
            path.write_text('epochs: 19\nimgsz: 960\ncls: 2.2\nworkers: 6\n'
                            'lr0: 0.0003\nrect: true\n', encoding='utf-8')
            try:
                root = tk.Tk()
            except tk.TclError as exc:
                self.skipTest('No Tk display: ' + str(exc))
            try:
                root.geometry('1190x780')
                _, _button = _all_in_one_gui(root, on_submit=lambda _settings: None)
                root.update()
                import_buttons = [b for b in descendants(root)
                                  if isinstance(b, ttk.Button) and b.cget('text') == 'Browse / import…']
                self.assertEqual(len(import_buttons), 1)
                with patch('tkinter.filedialog.askopenfilename', return_value=str(path)):
                    import_buttons[0].invoke()
                root.update()
                previews = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)]
                self.assertEqual(len(previews), 1)
                apply_buttons = [b for b in descendants(previews[0])
                                 if isinstance(b, ttk.Button) and b.cget('text') == 'Apply settings']
                self.assertEqual(len(apply_buttons), 1)
                apply_buttons[0].invoke()
                root.update()
                status = [w.cget('text') for w in descendants(root) if isinstance(w, ttk.Label)
                          and 'Import Ultralytics' in str(w.cget('text'))]
                self.assertTrue(status)
                entries = [w.get() for w in descendants(root) if isinstance(w, ttk.Entry)]
                self.assertIn('960', entries)
                self.assertIn('2.2', entries)
                self.assertIn('6', entries)
                self.assertIn(str(path), entries)
            finally:
                root.destroy()


if __name__ == '__main__':
    unittest.main()
