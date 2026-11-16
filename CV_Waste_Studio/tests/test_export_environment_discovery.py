"""Regression tests for Conda environments missing from the Model Export dropdown."""
from pathlib import Path
import json
import os
import sys
import tempfile
import time
import tkinter as tk
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_environment import discover_export_environments, python_in_environment
from model_export_ui import ModelExportPanel


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)

    def put_python(self, prefix, style='win'):
        path = prefix / ('python.exe' if style == 'win' else 'bin/python')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'')
        return str(path.resolve())

    def test_windows_conda_interpreters_live_at_root_not_scripts(self):
        env = self.home / 'AppData/Local/anaconda3/envs/Some Project'
        python = self.put_python(env)
        self.assertEqual(str(python_in_environment(env)), python)

    def test_enumerates_other_windows_conda_envs_without_conda_on_path(self):
        prefix = self.home / 'AppData/Local/anaconda3'
        own = self.put_python(prefix / 'envs/CV-Waste')
        other = self.put_python(prefix / 'envs/Few_Bag_Learning_Paper')
        third = self.put_python(prefix / 'envs/YOLO-TensorRT')
        self.put_python(prefix)
        with patch.dict(os.environ, {'LOCALAPPDATA':str(self.home / 'AppData/Local')}), \
             patch('export_environment.shutil.which', return_value=None):
            environments = discover_export_environments(own, home=self.home, conda_envs=[])
        paths = [e['python'] for e in environments]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertTrue({own, other, third}.issubset(set(paths)))
        self.assertGreaterEqual(len(paths), 4)

    def test_custom_prefix_from_conda_registry_and_extra_env_directories(self):
        own = self.put_python(self.home / 'miniconda3/envs/CV-Waste', 'linux')
        registered = self.put_python(self.home / 'external/Another Custom Conda', 'linux')
        venv = self.put_python(self.home / '.virtualenvs/tf-inference', 'linux')
        registry = self.home / '.conda/environments.txt'
        registry.parent.mkdir(parents=True)
        registry.write_text(str(Path(registered).parent.parent) + '\n# old environment\n/nonexistent/conda\n')
        envs = discover_export_environments(own, self.home, [])
        self.assertTrue({registered, venv}.issubset({x['python'] for x in envs}))

    def test_conda_discovered_outside_standard_roots(self):
        own = self.put_python(self.home / 'miniconda3/envs/CV-Waste', 'linux')
        other = self.put_python(self.home / 'special/trt-python', 'linux')
        conda = self.home / 'miniconda3/bin/conda'
        conda.parent.mkdir(parents=True, exist_ok=True)
        conda.touch()
        class DummyResult:
            returncode = 0
            stdout = json.dumps({'envs':[str(Path(other).parent.parent)]})
        with patch('export_environment.subprocess.run', return_value=DummyResult()) as run:
            envs = discover_export_environments(own, self.home)
        self.assertIn(other, {e['python'] for e in envs})
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0][:2], [str(conda), 'env'])


class GUIRefreshTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk(); self.root.withdraw()
        except tk.TclError as exc:
            self.skipTest(str(exc))
        self.addCleanup(self.root.destroy)
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.rootdir = Path(self.temp.name)
        self.managed = self.rootdir / 'envs/managed/python.exe'
        self.other = self.rootdir / 'envs/tensorrt/python.exe'
        for path in (self.managed, self.other):
            path.parent.mkdir(parents=True, exist_ok=True); path.touch()
        self.discover_count = 0
        def discover(*args, **kwargs):
            self.discover_count += 1
            result = [{'label':'Studio managed','python':str(self.managed)}]
            if self.discover_count > 1:
                result.append({'label':'Conda · TensorRT','python':str(self.other)})
            return result
        with patch('model_export_ui.discover_export_environments', side_effect=discover), \
             patch('model_export_ui.load_selection',return_value=''), \
             patch.object(ModelExportPanel,'check_compatibility',lambda self:None):
            self.panel = ModelExportPanel(self.root, env_python=lambda: str(self.managed),
                                          process_env=lambda: {'PATH':'/usr/bin'},
                                          logs_dir=self.temp.name)
        self.panel.pack(fill='both',expand=True)
        self.discover = discover

    def test_background_refresh_adds_environments_preserves_selection(self):
        self.assertEqual(len(self.panel._env_options), 1)
        with patch('model_export_ui.discover_export_environments', side_effect=self.discover):
            self.panel.refresh_environments()
            end = time.monotonic() + 4
            while self.panel._env_discovery_running and time.monotonic() < end:
                self.root.update()
                time.sleep(.015)
        self.assertFalse(self.panel._env_discovery_running)
        self.assertEqual(len(self.panel._env_options), 2)
        self.assertEqual(self.panel._export_python(),str(self.managed))
        self.assertIn('2 Python environments',self.panel.env_discovery_status.get())
        self.assertIn(str(self.other),' '.join(self.panel.env_combo['values']))

    def test_stale_probe_from_old_python_is_ignored(self):
        self.panel._env_options['Custom · ' + str(self.other)] = str(self.other)
        self.panel._probe_python = str(self.managed)
        self.panel._probe_pending = True
        self.panel._probe_queue.put(('probe', {'ultralytics':'old environment'}))
        with patch('model_export_ui.save_selection'), \
             patch.object(self.panel, 'check_compatibility') as recheck, \
             patch.object(self.panel, '_apply_probe') as apply:
            self.panel.export_env_choice.set('Custom · '+str(self.other))
            self.panel._environment_changed()
            self.panel.poll()
        apply.assert_not_called()
        self.assertEqual(recheck.call_count, 2)

    def test_refresh_preserves_saved_external_interpreter(self):
        self.panel._env_options['Custom · ' + str(self.other)] = str(self.other)
        self.panel.export_env_choice.set('Custom · '+str(self.other))
        self.panel._populate_environments([{'label':'Studio managed','python':str(self.managed)}],
                                          self.panel._export_python())
        self.assertEqual(self.panel._export_python(),str(self.other))


if __name__ == '__main__': unittest.main()
