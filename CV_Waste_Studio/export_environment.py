"""Discover/select export interpreters without modifying their dependencies.

The Studio GUI continues to use its own managed Python. An explicitly chosen
export interpreter is launched as an isolated child process only for exports.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def python_in_environment(folder: str | Path) -> Path | None:
    base = Path(folder).expanduser()
    paths = (base / 'bin' / 'python', base / 'Scripts' / 'python.exe')
    return next((p.resolve() for p in paths if p.is_file()), None)


def discover_export_environments(managed_python: str, home: str | Path | None = None,
                                 conda_envs: list[str] | None = None) -> list[dict]:
    """Find Conda and SenSoRTC installer venvs, including non-Conda paths."""
    base = Path(home).expanduser() if home else Path.home()
    candidates: list[tuple[str, Path]] = [('Studio managed environment', Path(managed_python))]
    if conda_envs is None:
        conda_envs = []
        for exe in ('conda', str(base / 'miniconda3' / 'bin' / 'conda')):
            try:
                result = subprocess.run([exe, 'env', 'list', '--json'], capture_output=True,
                                        text=True, timeout=6, check=False)
                if result.returncode == 0:
                    conda_envs = json.loads(result.stdout).get('envs', [])
                    break
            except (OSError, subprocess.TimeoutExpired, ValueError):
                continue
    for folder in conda_envs:
        p = python_in_environment(folder)
        if p is not None:
            candidates.append((f'Conda · {Path(folder).name}', p))
    senso_root = base / '.local' / 'share' / 'SenSoRTC' / 'envs'
    if senso_root.is_dir():
        for folder in sorted(senso_root.glob('install-*'), reverse=True):
            p = python_in_environment(folder)
            if p is not None:
                candidates.append((f'SenSoRTC · {folder.name}', p))
    seen = set()
    items = []
    for name, path in candidates:
        try:
            resolved = str(path.expanduser().resolve())
        except OSError:
            continue
        if resolved in seen or not Path(resolved).is_file():
            continue
        items.append({'label': name, 'python': resolved})
        seen.add(resolved)
    return items


def make_export_process_env(base_env: dict, python: str, managed_python: str) -> dict:
    """Prevent Studio's Conda/Python paths contaminating external interpreters."""
    env = dict(base_env)
    env['YOLO_AUTOINSTALL'] = 'False'
    # Export dependencies are preflight checked; never auto-repair a runtime.
    if Path(python).resolve() != Path(managed_python).resolve():
        for key in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'CONDA_PREFIX',
                    'CONDA_DEFAULT_ENV', 'CONDA_SHLVL', 'LD_LIBRARY_PATH',
                    'LD_PRELOAD', 'QT_QPA_PLATFORM_PLUGIN_PATH'):
            env.pop(key, None)
        env['PATH'] = str(Path(python).parent) + os.pathsep + env.get('PATH', '')
        env['PYTHONNOUSERSITE'] = '1'
    return env


def settings_path() -> Path:
    root = (Path(os.environ['APPDATA']) if os.name == 'nt' and os.environ.get('APPDATA')
            else Path.home() / '.config')
    return root / 'CV-Waste-Studio' / 'model_export_environment.json'


def load_selection() -> str:
    try:
        return str(json.loads(settings_path().read_text(encoding='utf-8')).get('python', ''))
    except (OSError, ValueError, TypeError):
        return ''


def save_selection(python: str) -> None:
    target = settings_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({'python': str(Path(python).resolve())}, indent=2), encoding='utf-8')
