"""GUI-space tests. These run with a real Tk display, including under Xvfb."""
from pathlib import Path
import sys
import tkinter as tk
import unittest
from unittest.mock import patch, Mock
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from greenplast_launcher import Studio


class AdaptiveWorkspaceTests(unittest.TestCase):
    def setUp(self):
        try:
            with patch('greenplast_launcher._same_executable', return_value=True):
                self.app = Studio(prepared=('conda', Path(sys.prefix),
                                            {'torch': 'test', 'cuda': False}),
                                  allow_multiple=True)
            self.app.root.update()
        except tk.TclError as exc:
            self.skipTest(f'Tk display not available: {exc}')
        self.assertTrue(self.app.ready)
        self.app._poll_training = lambda: None
        self.app.show('train')
        self.app.root.update()

    def tearDown(self):
        if hasattr(self, 'app'):
            self.app.root.destroy()

    def test_setup_gives_most_space_to_parameters(self):
        app = self.app
        self.assertEqual(app.page_card.winfo_manager(), 'pack')
        self.assertEqual(app.metric_frame.winfo_manager(), '')
        self.assertLess(app.log_frame.winfo_height(), 180)
        self.assertGreater(app.page_card.winfo_height(), 450)

    def test_monitor_settings_switch_and_other_pages(self):
        app = self.app
        app.train_job = {'process': object(), 'kind': 'train'}
        app.train_state.set('Running existing dataset training')
        app.set_graph_visibility(True)
        app.log_visible = True
        app._show('train')
        app.root.update()
        self.assertEqual(app.page_card.winfo_manager(), '')
        self.assertGreater(app.training_graph.winfo_height(), 300)
        self.assertEqual(app.settings_toggle.cget('text'), 'Show settings')
        self.assertEqual(app.title.cget('text'), 'Training monitor')
        app.toggle_training_settings()
        app.root.update()
        self.assertEqual(app.page_card.winfo_manager(), 'pack')
        self.assertGreater(app.page_card.winfo_height(), 300)
        self.assertEqual(app.settings_toggle.cget('text'), 'Hide settings')
        self.assertEqual(app.iteration_bar.winfo_manager(), 'pack')
        app.toggle_training_settings()
        app.root.update()
        self.assertEqual(app.page_card.winfo_manager(), '')
        app.train_job = None
        app.set_graph_visibility(False)
        app._show('train')
        app.root.update()
        self.assertLess(app.log_frame.winfo_height(), 180)
        app.show('results')
        app.root.update()
        self.assertEqual(app.log_frame.winfo_manager(), '')

    def test_real_job_start_switches_to_monitor(self):
        app = self.app
        app.env = Path(sys.prefix)
        fake_process = Mock()
        fake_process.pid = 1224
        fake_process.poll.return_value = None
        with tempfile.TemporaryDirectory() as tmp:
            with patch('greenplast_launcher.APP_DIR', Path(tmp)), \
                 patch('greenplast_launcher.env_python', return_value=Path(sys.executable)), \
                 patch('greenplast_launcher.process_environment', return_value={}), \
                 patch('greenplast_launcher.subprocess.Popen', return_value=fake_process):
                app.start_training({'train_after_split': True, 'cache': 'disk',
                                    'cleanup_disk_cache': False})
            app.root.update()
            self.assertEqual(app.page_card.winfo_manager(), '')
            self.assertEqual(app.settings_toggle.cget('text'), 'Show settings')
            self.assertEqual(app.log_toggle.cget('text'), 'Hide log')
            self.assertEqual(app.training_graph.winfo_manager(), 'pack')
            self.assertGreater(app.training_graph.winfo_height(), 300)
            self.assertIn('Running', app.monitor_label.cget('text'))
        app.train_job = None


if __name__ == '__main__':
    unittest.main()
