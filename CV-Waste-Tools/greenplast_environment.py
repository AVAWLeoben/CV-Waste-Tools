# -*- coding: utf-8 -*-
"""Conda environment detection, reuse and CUDA-aware setup for GreenPlast.

All functions here use Python's standard library. The Linux bootstrap calls
these functions *before* a GUI Python environment is available.
"""
from __future__ import annotations

import json
import hashlib
import os
import platform
from datetime import datetime, timezone
import queue
import threading
import time
from pathlib import Path
import re
import shutil
import subprocess
import sys

APP_DIR = Path(__file__).resolve().parent
MANAGED_NAME = "cv-waste-tools"
MANAGED_DISPLAY_NAME = "CV Waste Tools"

# Persistent, editable JSON configuration. A successful full validation is
# cached; most launches need only filesystem checks + one fast GPU identity
# query, not imports of torch, torchvision, Ultralytics, or Conda env discovery.
CACHE_SCHEMA = 2
if os.name == "nt":
    CONFIG_DIR = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "CVWasteTools"
else:
    CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "cv-waste-tools"
CONFIG_FILE = CONFIG_DIR / "launcher_config.json"
MIN_PYTHON = (3, 10)
MAX_PYTHON = (3, 14)

# Lightweight import-discovery check: fast rejection of environments missing
# dependencies, without importing heavyweight CUDA/PyTorch modules.
PREFLIGHT = r'''
import importlib.util, json, sys
modules = ('tkinter', 'numpy', 'yaml', 'cv2', 'torch', 'torchvision', 'ultralytics')
missing = [name for name in modules if importlib.util.find_spec(name) is None]
print('CVW_PREFLIGHT=' + json.dumps({'python': list(sys.version_info[:3]),
                                    'missing': missing}), flush=True)
'''

# Print each stage *before* importing a heavy dependency. Windows antivirus,
# first-import caches, and CUDA initialization can easily exceed 100 seconds.
PROBE = r'''
import json, sys
out = {'python': list(sys.version_info[:3]), 'ok': False, 'cuda': False}
def stage(name):
    print('CVW_STAGE=' + name, flush=True)
try:
    stage('Importing GUI, NumPy and OpenCV')
    import tkinter
    import numpy
    import yaml
    import cv2
    stage('Importing PyTorch')
    import torch
    stage('Importing torchvision')
    import torchvision
    stage('Importing Ultralytics')
    import ultralytics
    from ultralytics.data.split import autosplit
    out.update(ok=True, torch=str(torch.__version__),
               ultralytics=str(ultralytics.__version__))
    stage('Checking CUDA availability')
    out['cuda'] = bool(torch.cuda.is_available())
    if out['cuda']:
        out['gpu'] = torch.cuda.get_device_name(0)
        out['compute_capability'] = list(torch.cuda.get_device_capability(0))
        out['compiled_arches'] = list(torch.cuda.get_arch_list())
        # CUDA can report an available GPU even when the installed wheel has
        # no compiled kernels for its architecture (e.g. GTX 1060 + CUDA 13).
        architecture = 'sm_' + ''.join(map(str, out['compute_capability']))
        if out['compiled_arches'] and architecture not in out['compiled_arches']:
            raise RuntimeError(
                f'PyTorch {torch.__version__} has no {architecture} kernels for '
                f'{out["gpu"]}; compiled architectures: '
                + ', '.join(out['compiled_arches']))
        stage('Testing CUDA computation')
        x = torch.ones(2, device='cuda')
        torch.cuda.synchronize()
        assert float(x.sum().item()) == 2.0, 'GPU compute probe failed'
    else:
        out['gpu'] = ''
except Exception as exc:
    out['ok'] = False
    out['cuda'] = False
    out['error'] = type(exc).__name__ + ': ' + str(exc)
print('CVW_PROBE=' + json.dumps(out), flush=True)
'''


def env_python(prefix: Path) -> Path:
    """Return the interpreter belonging to *this* Conda prefix directly."""
    prefix = Path(prefix)
    return prefix / ('python.exe' if os.name == 'nt' else 'bin/python')


def process_environment(prefix: Path) -> dict:
    """Provide Conda DLL/shared-library search paths without `conda run`."""
    prefix = Path(prefix)
    env = os.environ.copy()
    if os.name == 'nt':
        bin_dirs = [prefix, prefix / 'Library' / 'mingw-w64' / 'bin',
                    prefix / 'Library' / 'usr' / 'bin', prefix / 'Library' / 'bin',
                    prefix / 'Scripts']
    else:
        bin_dirs = [prefix / 'bin']
    env['PATH'] = os.pathsep.join(map(str, bin_dirs)) + os.pathsep + env.get('PATH', '')
    env['CONDA_PREFIX'] = str(prefix)
    env['CONDA_DEFAULT_ENV'] = prefix.name
    return env


def find_conda() -> str:
    """Find a real Conda executable, including on Linux without shell init."""
    is_win = os.name == "nt"
    exe = "conda.exe" if is_win else "conda"
    candidates = [os.environ.get("CONDA_EXE"), shutil.which(exe)]
    if not is_win:
        # On Linux `conda` is often a shell function, so look for its binary.
        candidates.append(shutil.which("conda"))
    roots = [
        Path(sys.prefix), Path.home() / "miniconda3",
        Path.home() / "anaconda3", Path.home() / "miniforge3",
        Path.home() / "mambaforge", Path("/opt/conda"),
        Path("/opt/miniconda3"), Path("/opt/miniforge3"),
    ]
    if is_win:
        local = Path(os.environ.get("LOCALAPPDATA", str(Path.home())))
        roots.extend([local / "miniconda3", local / "anaconda3"])
    for root in roots:
        candidates.append(str(root / ("Scripts" if is_win else "bin") / exe))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    raise RuntimeError(
        "Conda was not found. Install Miniconda, Miniforge, or Anaconda "
        "and rerun START_GREENPLAST.bat (Windows) or START_GREENPLAST.sh (Linux)."
    )


def list_envs(conda: str) -> list[Path]:
    result = subprocess.run([conda, "env", "list", "--json"], capture_output=True,
                            text=True, errors="replace", timeout=90, check=False)
    if result.returncode:
        raise RuntimeError("Could not list Conda environments:\n" +
                           (result.stderr or result.stdout)[-1800:])
    data = json.loads(result.stdout)
    return [Path(p) for p in data.get("envs", [])]


def _nvidia_compute_capability(executable: str, gpu_name: str) -> tuple[int, int] | None:
    """Ask the driver for GPU 0's actual SM version (not its CUDA version).

    nvidia-smi --query-gpu=compute_cap is supported on modern drivers. Some
    older drivers omit that field; known Pascal GTX 10-series GPUs have a
    documented 6.x architecture, which is sufficient to select legacy wheels.
    Unknown GPUs are never guessed to be Pascal from their driver version.
    """
    try:
        query = subprocess.run(
            [executable, '--query-gpu=compute_cap', '--format=csv,noheader'],
            capture_output=True, text=True, errors='replace', timeout=12,
            check=False)
        if query.returncode == 0:
            match = re.search(r'(?<!\d)(\d+)\.(\d+)',
                              query.stdout.splitlines()[0]) if query.stdout.strip() else None
            if match:
                return int(match[1]), int(match[2])
    except (OSError, subprocess.TimeoutExpired):
        pass
    # A conservative fallback for Pascal consumer cards on old nvidia-smi.
    if re.search(r'\bGTX\s*10\d{2}(?:\s*Ti)?\b', gpu_name, re.IGNORECASE):
        return (6, 1)
    return None


def detect_nvidia() -> tuple[str, tuple[int, int] | None, tuple[int, int] | None] | None:
    """Return (GPU 0 name, driver maximum CUDA, hardware compute capability)."""
    candidates = [shutil.which('nvidia-smi')]
    if os.name == 'nt':
        candidates += [r'C:\Windows\System32\nvidia-smi.exe',
                       r'C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe']
    for candidate in candidates:
        if not candidate or not Path(candidate).is_file():
            continue
        try:
            result = subprocess.run([candidate], capture_output=True,
                                    text=True, errors='replace', timeout=12,
                                    check=False)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode != 0 or 'NVIDIA-SMI' not in result.stdout:
            continue
        driver_match = re.search(r'CUDA Version:\s*(\d+)\.(\d+)', result.stdout)
        cuda = ((int(driver_match[1]), int(driver_match[2]))
                if driver_match else None)
        gpu_name = 'NVIDIA GPU'
        try:
            names = subprocess.run([candidate, '--query-gpu=name',
                                    '--format=csv,noheader'],
                                   capture_output=True, text=True,
                                   errors='replace', timeout=12, check=False)
            if names.returncode == 0 and names.stdout.strip():
                gpu_name = names.stdout.strip().splitlines()[0].strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
        if gpu_name == 'NVIDIA GPU':
            # The summary includes the name even on older driver builds.
            gpu_match = re.search(r'(GeForce GTX\s*10\d{2}\s*(?:Ti)?)',
                                  result.stdout, re.IGNORECASE)
            if gpu_match:
                gpu_name = gpu_match[1]
        cc = _nvidia_compute_capability(candidate, gpu_name)
        return gpu_name, cuda, cc
    return None


# PyTorch 2.5.1 CUDA 12.1/11.8 wheels retain sm_61 support for Pascal.
# This version is intentionally pinned: installing a newer CUDA 13 wheel
# produces 'no kernel image available' on GTX 1060 (compute capability 6.1).
PASCAL_TORCH = '2.5.1'
PASCAL_TORCHVISION = '0.20.1'


def torch_install_plan(driver_cuda: tuple[int, int] | None,
                       gpu_cc: tuple[int, int] | None) -> dict:
    """Choose wheel series using BOTH the driver and the hardware architecture.

    The driver's 'CUDA Version' is a *maximum supported runtime*, not the
    architecture of the card. For sm_60/sm_61/sm_70 use the last-known
    compatible, pinned PyTorch family rather than tracking latest.
    """
    if gpu_cc is not None and gpu_cc < (7, 5):
        if driver_cuda is not None and driver_cuda >= (12, 1):
            wheel = 'cu121'
        elif driver_cuda is not None and driver_cuda >= (11, 8):
            wheel = 'cu118'
        else:
            raise RuntimeError(
                f'NVIDIA compute capability {gpu_cc[0]}.{gpu_cc[1]} needs '
                'a legacy PyTorch CUDA wheel, but the driver does not '
                'report CUDA >= 11.8. Update the driver first.')
        return {'wheel': wheel, 'torch': PASCAL_TORCH,
                'torchvision': PASCAL_TORCHVISION, 'legacy': True}
    wheel = torch_index(driver_cuda)
    return {'wheel': wheel, 'torch': None, 'torchvision': None,
            'legacy': False}


def torch_index(cuda: tuple[int, int] | None) -> str | None:
    """Pick a prebuilt CUDA PyTorch wheel compatible with the NVIDIA driver.

    Driver CUDA version (nvidia-smi) is a *maximum*, not a toolkit requirement.
    Pip wheels bring their own CUDA user-space runtime libraries.
    """
    if cuda is None:
        return None
    for required, wheel in (
        ((13, 2), "cu132"), ((13, 0), "cu130"),
        ((12, 8), "cu128"), ((12, 6), "cu126"),
        ((11, 8), "cu118"),
    ):
        if cuda >= required:
            return wheel
    return None


def _run_staged_probe(command: list[str], prefix: Path, timeout: int, report=None) -> dict:
    """Stream probe stages; report precisely which import hangs or fails.

    Uses the environment's Python directly rather than `conda run`, avoiding
    Conda process startup, plugin scanning and its Windows output wrapper.
    """
    messages = queue.Queue()
    lines = []
    stage = 'Starting Python'
    started = time.monotonic()
    try:
        proc = subprocess.Popen(command, cwd=str(APP_DIR),
                                env=process_environment(prefix),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, errors='replace', bufsize=1)
    except OSError as exc:
        return {'ok': False, 'error': f'Could not start environment Python: {exc}'}

    def reader():
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                messages.put(line.rstrip('\r\n'))
        finally:
            messages.put(None)

    threading.Thread(target=reader, daemon=True).start()
    finished = False
    while not finished:
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            if proc.stdout is not None:
                proc.stdout.close()
            return {'ok': False, 'timeout': True,
                    'error': (f'Environment check exceeded {timeout}s during: {stage}. '
                              'This can be a slow Windows first import, driver issue, '
                              'or a blocked Python process; it does not prove the packages '
                              'are missing. See the launcher log for the last stage.')}
        try:
            line = messages.get(timeout=min(0.5, remaining))
        except queue.Empty:
            continue
        if line is None:
            finished = True
        else:
            lines.append(line)
            if line.startswith('CVW_STAGE='):
                stage = line.partition('=')[2]
                if report:
                    report('status', f'Validating {prefix.name}: {stage} ...')
            elif report and line and not line.startswith('CVW_PROBE='):
                report('detail', line)
    code = proc.wait()
    if proc.stdout is not None:
        proc.stdout.close()
    results = [line.partition('=')[2] for line in lines if line.startswith('CVW_PROBE=')]
    if not results:
        return {'ok': False, 'error': f'Probe exited {code} at {stage}: '
                + '\n'.join(lines[-5:])[-1500:]}
    try:
        return json.loads(results[-1])
    except json.JSONDecodeError:
        return {'ok': False, 'error': 'Could not decode Python environment check'}


def probe_env(conda: str, prefix: Path, needs_cuda: bool,
              timeout: int = 360, report=None) -> dict:
    """Validate dependencies and actual GPU compute, with a generous timeout.

    `conda` is retained in the signature for callers; direct prefix Python is
    used to eliminate `conda run` overhead and the 100-second timeout failure.
    """
    prefix = Path(prefix)
    python = env_python(prefix)
    if not python.is_file():
        return {'ok': False, 'error': f'Conda environment Python missing: {python}'}
    try:
        quick = subprocess.run([str(python), '-u', '-c', PREFLIGHT],
                               cwd=str(APP_DIR), env=process_environment(prefix),
                               capture_output=True, text=True, errors='replace',
                               timeout=25, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'ok': False, 'error': f'Python preflight failed: {exc}'}
    quick_lines = [line.partition('=')[2] for line in quick.stdout.splitlines()
                   if line.startswith('CVW_PREFLIGHT=')]
    if not quick_lines:
        return {'ok': False, 'error': 'Preflight did not respond: '
                + (quick.stderr or quick.stdout)[-700:]}
    try:
        metadata = json.loads(quick_lines[-1])
    except json.JSONDecodeError:
        return {'ok': False, 'error': 'Invalid preflight response'}
    py = tuple(metadata.get('python', [])[:2])
    if not (MIN_PYTHON <= py < MAX_PYTHON):
        return {'ok': False, 'error': f'Unsupported Python version: {py}'}
    missing = metadata.get('missing', [])
    if missing:
        return {'ok': False, 'missing': missing,
                'error': 'Missing dependencies: ' + ', '.join(missing)}
    data = _run_staged_probe([str(python), '-u', '-c', PROBE], prefix, timeout, report)
    data['ok'] = bool(data.get('ok') and (data.get('cuda') or not needs_cuda))
    if needs_cuda and not data['ok'] and not data.get('cuda') and not data.get('error'):
        data['error'] = 'PyTorch cannot access the NVIDIA GPU'
    return data



def _stat_signature(path: Path) -> dict | None:
    """Cheap change detector for Python, Conda installs and pip installs."""
    try:
        st = path.stat()
        return {'mtime_ns': st.st_mtime_ns, 'size': st.st_size}
    except OSError:
        return None


def _environment_signature(prefix: Path) -> dict:
    prefix = Path(prefix)
    python = env_python(prefix)
    if os.name == 'nt':
        sites = [prefix / 'Lib' / 'site-packages']
    else:
        sites = list((prefix / 'lib').glob('python*/site-packages'))
    return {
        'python': _stat_signature(python),
        'conda_history': _stat_signature(prefix / 'conda-meta' / 'history'),
        'site_packages': {str(p): _stat_signature(p) for p in sites},
    }


def _requirements_digest() -> str:
    return hashlib.sha256((APP_DIR / 'requirements.txt').read_bytes()).hexdigest()


def _gpu_fingerprint() -> str:
    """One lightweight nvidia-smi query; no torch/CUDA imports.

    Includes the driver so a driver replacement forces a fresh CUDA test.
    A changed GPU (for example GTX 1060 -> RTX 5090) also invalidates cache.
    """
    candidates = [shutil.which('nvidia-smi')]
    if os.name == 'nt':
        candidates += [r'C:\Windows\System32\nvidia-smi.exe',
                       r'C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe']
    executable = next((str(p) for p in candidates if p and Path(p).is_file()), None)
    if not executable:
        return 'no-nvidia-smi'
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    # Some older drivers do not provide compute_cap; the fallback still
    # detects a GPU/driver swap without starting torch.
    for fields in ('name,compute_cap,driver_version', 'name,driver_version'):
        try:
            result = subprocess.run(
                [executable, f'--query-gpu={fields}', '--format=csv,noheader'],
                capture_output=True, text=True, errors='replace', check=False,
                timeout=10, creationflags=flags)
            if result.returncode == 0 and result.stdout.strip():
                return f'{fields}: ' + result.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            break
    return 'gpu-query-failed'


def _read_config() -> dict:
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}


def cached_environment(expected_conda: str | None = None,
                       expected_prefix: Path | None = None) -> tuple[str, Path, dict] | None:
    """Return verified saved setup when cheap identity checks still pass.

    No heavy Python imports, no Conda subprocess, and no CUDA computation.
    Never trust cached evidence on another machine or after a package/driver
    update. An explicit GUI recheck bypasses this entire function.
    """
    data = _read_config()
    if data.get('schema_version') != CACHE_SCHEMA or data.get('always_recheck') is True:
        return None
    saved = data.get('environment')
    if not isinstance(saved, dict) or saved.get('verified') is not True:
        return None
    conda = saved.get('conda_exe')
    prefix_str = saved.get('prefix')
    details = saved.get('probe')
    if not isinstance(conda, str) or not isinstance(prefix_str, str) or not isinstance(details, dict):
        return None
    prefix = Path(prefix_str)
    if not Path(conda).is_file() or not env_python(prefix).is_file():
        return None
    if expected_conda and os.path.normcase(os.path.abspath(conda)) != os.path.normcase(os.path.abspath(expected_conda)):
        return None
    if expected_prefix and os.path.normcase(os.path.abspath(str(prefix))) != os.path.normcase(os.path.abspath(str(expected_prefix))):
        return None
    if saved.get('machine') != platform.node() or saved.get('platform') != sys.platform:
        return None
    if saved.get('requirements_sha256') != _requirements_digest():
        return None
    if saved.get('environment_signature') != _environment_signature(prefix):
        return None
    if saved.get('gpu_fingerprint') != _gpu_fingerprint():
        return None
    if not details.get('ok'):
        return None
    return conda, prefix, details.copy()


def save_cached_environment(conda: str, prefix: Path, details: dict) -> None:
    """Atomically persist an environment only after a successful full probe."""
    if not details.get('ok'):
        raise ValueError('Cannot cache an unverified environment')
    config = _read_config()
    config.update({
        'schema_version': CACHE_SCHEMA,
        'always_recheck': bool(config.get('always_recheck', False)),
        'environment': {
            'verified': True,
            'name': Path(prefix).name,
            'prefix': str(Path(prefix).resolve()),
            'conda_exe': str(Path(conda).resolve()),
            'machine': platform.node(),
            'platform': sys.platform,
            'verified_at_utc': datetime.now(timezone.utc).isoformat(),
            'requirements_sha256': _requirements_digest(),
            'environment_signature': _environment_signature(prefix),
            'gpu_fingerprint': _gpu_fingerprint(),
            'probe': details,
        },
    })
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    target = CONFIG_FILE.with_name(CONFIG_FILE.name + '.tmp')
    try:
        target.write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n',
                          encoding='utf-8')
        os.replace(target, CONFIG_FILE)
    finally:
        target.unlink(missing_ok=True)

class EnvironmentManager:
    def __init__(self, report):
        self.report = report

    def _save_cache_safe(self, conda: str, prefix: Path, check: dict):
        try:
            save_cached_environment(conda, prefix, check)
            self.report('log', 'Saved environment configuration: ' + str(CONFIG_FILE))
        except (OSError, ValueError, TypeError) as exc:
            self.report('log', f'Warning: could not save environment cache: {exc}')

    def run_checked(self, command: list[str], title: str):
        self.report("log", title)
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, errors="replace", bufsize=1)
        tail = []
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                tail.append(line)
                if len(tail) > 30:
                    tail.pop(0)
                self.report("detail", line)
        result = proc.wait()
        if result:
            raise RuntimeError(f"{title} failed (exit code {result}).\n" +
                               "\n".join(tail[-12:]))

    def setup(self, force_recheck: bool = False) -> tuple[str, Path, dict]:
        if not force_recheck:
            cached = cached_environment()
            if cached:
                self.report('log', f'Fast start: using verified configuration: {CONFIG_FILE}')
                self.report('log', f'Reusing Conda environment: {cached[1]}')
                return cached
        else:
            self.report('log', 'Explicit environment recheck requested (ignoring saved verification).')
        conda = find_conda()
        self.report("log", "Conda detected: " + conda)
        nvidia = detect_nvidia()
        gpu_cc = nvidia[2] if nvidia and len(nvidia) > 2 else None
        plan = torch_install_plan(nvidia[1], gpu_cc) if nvidia else {
            'wheel': None, 'torch': None, 'torchvision': None, 'legacy': False}
        gpu_wheel = plan['wheel']
        needs_cuda = bool(nvidia and gpu_wheel)
        if nvidia:
            self.report('log', f'NVIDIA GPU 0: {nvidia[0]}')
            self.report('log', 'Driver maximum CUDA version: ' +
                        ('.'.join(map(str, nvidia[1])) if nvidia[1] else 'unknown'))
            self.report('log', 'Hardware compute capability: ' +
                        ('.'.join(map(str, gpu_cc)) if gpu_cc else 'unknown'))
            if plan['legacy']:
                self.report('log', f'Pascal/legacy GPU: pinning torch '
                            f'{plan["torch"]} + torchvision {plan["torchvision"]} '
                            f'({gpu_wheel}) to retain sm_61/sm_70 kernels.')
            if not needs_cuda:
                self.report("log", "Driver CUDA version unsupported/unknown: GPU setup is unavailable; using CPU-compatible setup.")
        else:
            self.report("log", "No NVIDIA driver detected: CPU-compatible setup.")
        envs = list_envs(conda)
        preferred = sorted(envs, key=lambda p: (p.name.lower() != MANAGED_NAME,
                                               p.name.lower(), str(p)))
        self.report("log", f"Checking {len(preferred)} existing Conda environments ...")
        for prefix in preferred:
            self.report("status", "Checking " + prefix.name + " ...")
            check = probe_env(conda, prefix, needs_cuda,
                              timeout=480 if prefix.name.lower() == MANAGED_NAME else 150,
                              report=self.report if prefix.name.lower() == MANAGED_NAME else None)
            if check.get("ok"):
                self.report("log", f"Reusing environment: {prefix}")
                self._save_cache_safe(conda, prefix, check)
                return conda, prefix, check
            # A timeout is NOT evidence of missing packages. Never reinstall
            # packages into an environment simply because initialization is slow.
            if prefix.name.lower() == MANAGED_NAME and check.get('timeout'):
                raise RuntimeError(str(check['error']))
            # Avoid excessive logs when a machine has many other projects.
            if prefix.name.lower() == MANAGED_NAME:
                self.report("log", "Dedicated environment requires setup: " +
                            str(check.get("error", "missing dependencies")))
        # Never install into an unrelated existing environment.
        managed = next((env for env in envs if env.name.lower() == MANAGED_NAME), None)
        if managed is None:
            self.report("status", "Creating Conda environment (first run) ...")
            self.run_checked([conda, "create", "-y", "-n", MANAGED_NAME,
                              "python=3.11", "pip", "tk"],
                             f"Creating persistent Conda environment: {MANAGED_DISPLAY_NAME} ({MANAGED_NAME})")
            envs = list_envs(conda)
            managed = next((env for env in envs if env.name.lower() == MANAGED_NAME), None)
            if managed is None:
                raise RuntimeError("Conda reported successful creation, but the new environment "
                                   "cannot be located.")
        else:
            self.report("log", f"Repairing existing {MANAGED_DISPLAY_NAME} environment only.")
        base_cmd = [conda, "run", "--no-capture-output", "-p", str(managed),
                    "python", "-m", "pip"]
        self.report("status", "Installing PyTorch (first setup can be large) ...")
        index = (f"https://download.pytorch.org/whl/{gpu_wheel}" if needs_cuda else
                 "https://download.pytorch.org/whl/cpu")
        if plan['legacy']:
            torch_packages = [f'torch=={plan["torch"]}+{gpu_wheel}',
                              f'torchvision=={plan["torchvision"]}+{gpu_wheel}']
            # Deliberately replace the incompatible CUDA 13 wheels.
            torch_command = base_cmd + ['install', '--force-reinstall',
                                        *torch_packages, '--index-url', index]
        else:
            torch_command = base_cmd + ['install', '--upgrade', 'torch',
                                        'torchvision', '--index-url', index]
        self.run_checked(torch_command,
                         f"Installing PyTorch ({gpu_wheel or 'CPU'})")
        self.report("status", "Installing Ultralytics and GUI dependencies ...")
        self.run_checked(base_cmd + ["install", "-r", str(APP_DIR / "requirements.txt")],
                         "Installing application dependencies")
        # Third-party dependencies must not silently replace the intentionally
        # pinned Pascal-compatible torch build. Verification checks installed
        # architecture support as well as actual GPU computation.
        self.report("status", "Validating environment ...")
        check = probe_env(conda, managed, needs_cuda, timeout=480,
                          report=self.report)
        if not check.get("ok"):
            raise RuntimeError("New environment failed its dependency/CUDA check: " +
                               str(check.get("error", check)))
        self.report("log", "Environment ready. Future launches will reuse it without reinstalling.")
        self._save_cache_safe(conda, managed, check)
        return conda, managed, check


