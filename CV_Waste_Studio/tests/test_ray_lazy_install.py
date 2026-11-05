"""Optional Ray dependencies must be checked lazily and installed only by consent."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import tempfile
import tkinter as tk
import unittest
from unittest.mock import MagicMock, patch

from hyperparameter_tuning_ui import HyperparameterTuningPanel, ray_install_command


class RayLazySetupTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.panel = HyperparameterTuningPanel(self.root,
            env_python=lambda: sys.executable,
            process_env=lambda: os.environ.copy(), logs_dir=self.base)
        self.addCleanup(self.root.destroy)

    def test_initialization_does_not_schedule_or_run_probe(self):
        self.assertIsNone(self.panel._ray_checked_python)
        self.assertIsNone(self.panel.ray_probe)
        self.assertIn('checked when you open', self.panel.compat_label.get())
        self.assertIn('disabled', self.panel.ray_radio.state())
        with patch.object(self.panel, 'check_ray_compatibility') as check:
            self.root.update()
            check.assert_not_called()

    def test_check_on_first_tab_visit_only(self):
        # Avoid a real subprocess; verify the tab lifecycle sends one probe.
        def record_probe():
            self.panel._ray_checked_python = str(sys.executable)
        with patch.object(self.panel, 'check_ray_compatibility', side_effect=record_probe) as check:
            self.panel.on_show()
            self.panel.on_show()
            self.assertEqual(check.call_count, 1)

    def test_failed_probe_can_be_declined_without_install(self):
        with patch('hyperparameter_tuning_ui.messagebox.askyesno', return_value=False) as ask, \
             patch.object(self.panel, 'install_ray_dependencies') as install:
            self.panel._ray_unavailable("ModuleNotFoundError: No module named 'ray'")
        self.assertEqual(ask.call_count, 1)
        install.assert_not_called()
        self.assertEqual(self.panel.engine.get(), 'native')
        self.assertIn('disabled', self.panel.ray_radio.state())
        self.assertNotIn('disabled', self.panel.install_ray_btn.state())

    def test_approval_installs_in_selected_python(self):
        fake_proc = MagicMock()
        fake_proc.poll.return_value = None
        with patch('hyperparameter_tuning_ui.messagebox.askyesno', return_value=True), \
             patch('hyperparameter_tuning_ui.subprocess.Popen', return_value=fake_proc) as popen, \
             patch.object(self.panel, 'after') as after:
            self.panel._ray_unavailable("ModuleNotFoundError: No module named 'ray'")
            self.assertEqual(popen.call_count, 1)
            command=popen.call_args.args[0]
            self.assertEqual(command, ray_install_command(sys.executable))
            self.assertIn('ray[tune]>=2.41.0', command)
            self.assertTrue(self.panel.installing)
            self.assertTrue(self.panel.ray_install_job['log'].exists())
            self.assertIn('disabled', self.panel.install_ray_btn.state())
            self.assertIn('disabled', self.panel.start_btn.state())
            after.assert_called()

    def test_success_rechecks_without_reprompting(self):
        fake_proc = MagicMock()
        fake_proc.poll.return_value = 0
        logfile = self.base / 'pip.log'
        logfile.write_text('Successfully installed ray optuna\n')
        self.panel.ray_install_job={'process':fake_proc,'log':logfile,'offset':0}
        with patch.object(self.panel, 'check_ray_compatibility') as check:
            self.panel._poll_ray_install()
        self.assertIsNone(self.panel.ray_install_job)
        check.assert_called_once_with(offer_install=False)
        self.assertNotIn('disabled', self.panel.start_btn.state())

    def test_failed_install_keeps_native_and_shows_log_path(self):
        fake_proc=MagicMock()
        fake_proc.poll.return_value=1
        logfile=self.base/'failed_pip.log'
        logfile.write_text('ERROR: No matching distribution found')
        self.panel.ray_install_job={'process':fake_proc,'log':logfile,'offset':0}
        self.panel._poll_ray_install()
        self.assertIn('failed', self.panel.compat_label.get())
        self.assertIn(str(logfile), self.panel.compat_label.get())
        self.assertIn('disabled', self.panel.ray_radio.state())
        self.assertNotIn('disabled', self.panel.install_ray_btn.state())

    def test_probe_success_enables_ray_without_install(self):
        report={'available':True, 'versions':{'ultralytics':'8.4.174','ray':'2.50','optuna':'4.0'}}
        fake_proc=MagicMock()
        fake_proc.poll.return_value=0
        fake_proc.communicate.return_value=(json.dumps(report).encode(),b'')
        self.panel.ray_probe=fake_proc
        self.panel._finish_ray_probe()
        self.assertTrue(self.panel.ray_available)
        self.assertNotIn('disabled', self.panel.ray_radio.state())
        self.assertIn('disabled', self.panel.install_ray_btn.state())


if __name__ == '__main__':
    unittest.main()
