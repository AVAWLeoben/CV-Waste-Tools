"""WSL LiteRT bridge: Windows checks, safe subprocess args and output preservation."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import wsl_litert
from model_export import validate_export_settings


def cp(stdout=b"", code=0, stderr=b""):
    return subprocess.CompletedProcess([], code, stdout, stderr)


class WslProbeTests(unittest.TestCase):
    def test_utf16_distro_names(self):
        self.assertEqual(wsl_litert.decode_wsl_output("Ubuntu\r\nDebian\r\n".encode("utf-16-le")),
                         "Ubuntu\r\nDebian")

    def test_missing_wsl_is_non_destructive(self):
        with patch.object(wsl_litert.sys, "platform", "win32"):
            result = wsl_litert.probe_wsl_litert(run=lambda args, timeout: cp(code=1, stderr=b"WSL is not installed"))
        self.assertEqual(result["phase"], "missing_wsl")
        self.assertFalse(result["available"])

    def test_wsl_environment_ready(self):
        calls = []
        def run(args, timeout):
            calls.append(args)
            if "--list" in args:
                return cp("Ubuntu\nDebian".encode("utf-16-le"))
            if "/bin/uname" in args:
                return cp(b"x86_64\n")
            response = {"python": "3.11.11", "missing": [], "format": "litert"}
            return cp(("STUDIO_WSL_PROBE=" + json.dumps(response) + "\n").encode())
        with patch.object(wsl_litert.sys, "platform", "win32"):
            result = wsl_litert.probe_wsl_litert(run=run)
        self.assertTrue(result["available"])
        self.assertEqual(result["distribution"], "Ubuntu")
        self.assertTrue(all(isinstance(x, list) for x in calls))
        self.assertEqual(calls[-1][1:3], ["--distribution", "Ubuntu"])

    def test_wsl1_reports_upgrade_command(self):
        def run(args, timeout):
            if "--quiet" in args:
                return cp("Ubuntu\n".encode("utf-16-le"))
            if "--verbose" in args:
                return cp("NAME STATE VERSION\n* Ubuntu Stopped 1\n".encode("utf-16-le"))
            raise AssertionError(args)
        with patch.object(wsl_litert.sys, "platform", "win32"):
            result = wsl_litert.probe_wsl_litert(run=run)
        self.assertEqual(result["phase"], "wsl1")
        self.assertIn("wsl --set-version Ubuntu 2", result["reason"])

    def test_missing_venv_asks_for_setup(self):
        def run(args, timeout):
            if "--list" in args:
                return cp("Ubuntu\n".encode("utf-16-le"))
            if "/bin/uname" in args:
                return cp(b"x86_64")
            return cp(code=42)
        with patch.object(wsl_litert.sys, "platform", "win32"):
            result = wsl_litert.probe_wsl_litert(run=run)
        self.assertEqual(result["phase"], "missing_env")
        self.assertEqual(result["distribution"], "Ubuntu")

    def test_finds_working_exporter_in_second_distribution(self):
        calls = []
        def run(args, timeout):
            calls.append(args)
            if "--quiet" in args:
                return cp("Debian\nUbuntu\n".encode("utf-16-le"))
            if "--verbose" in args:
                return cp("NAME STATE VERSION\n Debian Stopped 2\n* Ubuntu Running 2\n".encode("utf-16-le"))
            if "/bin/uname" in args:
                return cp(b"x86_64\n")
            if "Debian" in args:
                return cp(code=42)
            response = {"python": "3.11.10", "missing": [], "format": "litert"}
            return cp(("STUDIO_WSL_PROBE=" + json.dumps(response) + "\n").encode())
        with patch.object(wsl_litert.sys, "platform", "win32"):
            result = wsl_litert.probe_wsl_litert(run=run)
        self.assertTrue(result["available"])
        self.assertEqual(result["distribution"], "Ubuntu")
        self.assertEqual([a["distribution"] for a in result["attempts"]], ["Debian", "Ubuntu"])

    def test_optional_package_metadata_absence_does_not_trigger_reinstall(self):
        def run(args, timeout):
            if "--quiet" in args:
                return cp("Ubuntu\n".encode("utf-16-le"))
            if "--verbose" in args:
                return cp(code=1)
            if "/bin/uname" in args:
                return cp(b"x86_64\n")
            response = {"python": "3.12.7", "missing": [], "optional_missing":
                        ["litert-torch", "ai-edge-litert"], "format": "litert"}
            return cp(("STUDIO_WSL_PROBE=" + json.dumps(response) + "\n").encode())
        with patch.object(wsl_litert.sys, "platform", "win32"):
            result = wsl_litert.probe_wsl_litert(run=run)
        self.assertTrue(result["available"])
        self.assertEqual(result["phase"], "ready")
        self.assertIn("optional package checks", result["reason"])

    def test_working_python312_litert_torch_environment_without_tensorflow(self):
        # Real export log: Ultralytics 8.4.174, Python 3.12.15 and
        # litert_torch 0.9.4 converted successfully without TensorFlow.
        def run(args, timeout):
            if "--quiet" in args:
                return cp("Ubuntu\n".encode("utf-16-le"))
            if "--verbose" in args:
                return cp("NAME STATE VERSION\n* Ubuntu Running 2\n".encode("utf-16-le"))
            if "/bin/uname" in args:
                return cp(b"x86_64\n")
            python_probe = args[-1]
            self.assertIn('("torch", "ultralytics", "litert-torch")', python_probe)
            self.assertNotIn('("torch", "ultralytics", "tensorflow")', python_probe)
            response = {"python": "3.12.15", "missing": [],
                        "torch": "2.13.0", "ultralytics": "8.4.174",
                        "litert-torch": "0.9.4", "format": "litert",
                        "optional_missing": ["tensorflow"]}
            return cp(("STUDIO_WSL_PROBE=" + json.dumps(response) + "\n").encode())
        with patch.object(wsl_litert.sys, "platform", "win32"):
            result = wsl_litert.probe_wsl_litert(run=run)
        self.assertTrue(result["available"])
        self.assertEqual(result["phase"], "ready")
        self.assertIn("Python 3.12.15", result["reason"])
        self.assertEqual(result["argument"], "litert")

    @unittest.skipIf(sys.platform == "win32", "Needs a POSIX shell")
    def test_setup_helper_does_not_reinstall_an_existing_usable_environment(self):
        # Use a fake successful interpreter: this exercises the non-destructive
        # shortcut before the installer tries uv, pip, or removing the env.
        import os
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            python = home / "yolo_export" / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            python.chmod(0o755)
            script = Path(__file__).resolve().parents[1] / "setup_yolo_export.sh"
            result = subprocess.run(["bash", str(script)], env={**os.environ, "HOME": str(home)},
                                    text=True, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("environment is ready", result.stdout)
            self.assertTrue(python.is_file())

    def test_missing_actual_litert_torch_dependency_is_reported(self):
        def run(args, timeout):
            if "--quiet" in args:
                return cp("Ubuntu\n".encode("utf-16-le"))
            if "--verbose" in args:
                return cp(code=1)
            if "/bin/uname" in args:
                return cp(b"x86_64\n")
            response = {"python": "3.12.15", "missing": ["litert-torch"], "format": "litert"}
            return cp(("STUDIO_WSL_PROBE=" + json.dumps(response) + "\n").encode())
        with patch.object(wsl_litert.sys, "platform", "win32"):
            result = wsl_litert.probe_wsl_litert(run=run)
        self.assertFalse(result["available"])
        self.assertEqual(result["phase"], "missing_packages")
        self.assertIn("litert-torch", result["reason"])

    def test_diagnostic_preserves_wsl_import_error(self):
        def run(args, timeout):
            if "--quiet" in args:
                return cp("Ubuntu\n".encode("utf-16-le"))
            if "--verbose" in args:
                return cp(code=1)
            if "/bin/uname" in args:
                return cp(b"x86_64\n")
            response = {"python": "3.11.10", "missing": [], "format": None,
                        "import_error": "ImportError: tensorflow mismatch"}
            return cp(("STUDIO_WSL_PROBE=" + json.dumps(response) + "\n").encode())
        with patch.object(wsl_litert.sys, "platform", "win32"):
            result = wsl_litert.probe_wsl_litert(run=run)
        self.assertFalse(result["available"])
        self.assertEqual(result["phase"], "env_error")
        self.assertIn("tensorflow mismatch", result["reason"])

    def test_invalid_path_translation_refused(self):
        with self.assertRaisesRegex(RuntimeError, "invalid path"):
            wsl_litert.windows_to_wsl(r"C:\\data\\model.pt", "Ubuntu", run=lambda *a, **k: cp(b"not-absolute"))


class WslExportTests(unittest.TestCase):
    def test_windows_litert_validation_with_wsl_capability(self):
        with tempfile.TemporaryDirectory() as folder:
            pt = Path(folder) / "best.pt"
            pt.write_bytes(b"MODEL")
            cap = {"formats": {"litert": {"available": True, "argument": "litert", "backend": "wsl"}}}
            vals = validate_export_settings({"model": str(pt), "format": "litert", "imgsz": "640x1088"}, cap)
            self.assertEqual(vals["backend"], "wsl")
            self.assertEqual(vals["imgsz"], (640, 1088))

    def test_run_wsl_export_keeps_original_and_sends_wsl_positional_args(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            pt = root / "original weights.pt"
            pt.write_bytes(b"real source file")
            formats = {"available": True, "argument": "litert", "distribution": "Ubuntu",
                       "distributions": ["Ubuntu"]}
            args_seen = []
            def run(args, timeout):
                if "wslpath" in args:
                    # Stub only for test; actual WSL maps C:\ to /mnt/c/...
                    return cp(b"/mnt/c/studio/data\n")
                raise AssertionError(args)
            class FakePopen:
                def __init__(self, command, stdin=None):
                    args_seen.append(command)
                def wait(self):
                    folder = next((root / "out").glob("export_*_litert"))
                    (folder / "exported.tflite").write_bytes(b"lite-rt")
                    (folder / "wsl_export_result.json").write_text(json.dumps(
                        {"filename": "exported.tflite", "ultralytics_arguments": {"format": "litert"}}))
                    return 0
            with patch.object(wsl_litert, "windows_to_wsl", side_effect=lambda path, distro, run=None: "/mnt/c/spaces here/x"):
                result = wsl_litert.run_wsl_litert_export(
                    {"format": "litert", "model": str(pt), "imgsz": "640x1088", "output": str(root / "out")},
                    probe=lambda: formats, run=run, popen=FakePopen)
            self.assertEqual(pt.read_bytes(), b"real source file")
            self.assertTrue(Path(result["exported_model"]).is_file())
            self.assertEqual(list((root / "out").rglob("*.pt")), [])
            self.assertIn("$@", args_seen[0][6])
            self.assertIn("/mnt/c/spaces here/x", args_seen[0])

    def test_export_exception_cleans_only_staged_model(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            pt = root / "best.pt"
            pt.write_bytes(b"do not delete")
            formats = {"available": True, "argument": "litert", "distribution": "Ubuntu", "distributions": ["Ubuntu"]}
            class FakePopen:
                def __init__(self, *args, **kwargs): pass
                def wait(self): return 13
            with patch.object(wsl_litert, "windows_to_wsl", return_value="/mnt/c/path"):
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    wsl_litert.run_wsl_litert_export({"format": "litert", "model": str(pt),
                        "output": str(root / "out")}, probe=lambda: formats, popen=FakePopen)
            self.assertEqual(pt.read_bytes(), b"do not delete")
            self.assertEqual(list((root / "out").rglob("*.pt")), [])

    def test_worker_writes_litert_summary_without_modifying_pt(self):
        import wsl_litert_export_worker as worker
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            pt = root / "weights.pt"
            pt.write_bytes(b"source")
            job = root / "job.json"
            job.write_text(json.dumps({"model": str(pt), "output": str(root), "format": "litert",
                                       "imgsz": [640, 1088], "batch": 1, "precision": "fp32", "nms": False}))
            class FakeModel:
                def __init__(self, path): self.path = Path(path)
                def export(self, **kwargs):
                    output = self.path.with_suffix(".tflite")
                    output.write_bytes(b"litert")
                    return str(output)
            with patch.dict(sys.modules, {"ultralytics": types.ModuleType("ultralytics"),
                                          "ultralytics.cfg": types.SimpleNamespace(DEFAULT_CFG_DICT={})}):
                worker.main([str(job)], FakeModel)
            self.assertEqual(pt.read_bytes(), b"source")
            self.assertTrue((root / "wsl_export_result.json").exists())
            self.assertFalse((root / "wsl_export.pid").exists())


if __name__ == "__main__":
    unittest.main()
