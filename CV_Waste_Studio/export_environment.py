"""Discover and select external Python environments without changing their packages.

The Studio GUI keeps its own interpreter. Model-export probes/workers run with the
explicitly selected interpreter. Environment *discovery* never imports packages
from, executes, or modifies any candidate environment.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def python_in_environment(folder: str | Path) -> Path | None:
    """Return a venv/Conda interpreter inside a prefix, on either platform.

    Crucially, Windows Conda uses ``PREFIX/python.exe``, NOT
    ``PREFIX/Scripts/python.exe`` (the latter is an uncommon layout).
    """
    base = Path(folder).expanduser()
    paths = (base / 'python.exe', base / 'bin' / 'python',
             base / 'Scripts' / 'python.exe')
    return next((p.resolve() for p in paths if p.is_file()), None)


def _environment_prefix(interpreter: str | Path) -> Path:
    python = Path(interpreter).expanduser()
    return python.parent.parent if python.parent.name.lower() in ('bin', 'scripts') else python.parent


def _candidate_roots(managed_python: str, home: Path) -> list[Path]:
    """Known installations plus the Conda installation containing Studio.

    No expensive recursive home-directory search: enumerate only immediate envs.
    """
    prefix = _environment_prefix(managed_python)
    roots = [prefix, Path(sys.prefix), home / 'miniconda3', home / 'anaconda3',
             home / 'miniforge3', home / 'mambaforge', home / 'micromamba',
             Path('/opt/conda'), Path('/opt/miniconda3'), Path('/opt/miniforge3')]
    if prefix.parent.name.lower() == 'envs':
        roots.insert(0, prefix.parent.parent)
    for variable in ('CONDA_PREFIX', 'CONDA_ROOT', 'MAMBA_ROOT_PREFIX'):
        value = os.environ.get(variable)
        if value:
            root = Path(value).expanduser()
            roots.append(root.parent.parent if root.parent.name.lower() == 'envs' else root)
    local = Path(os.environ.get('LOCALAPPDATA', str(home / 'AppData' / 'Local')))
    programdata = Path(os.environ.get('ProgramData', 'C:/ProgramData'))
    for parent in (local, home, programdata):
        roots.extend(parent / name for name in
                     ('anaconda3', 'miniconda3', 'miniforge3', 'mambaforge', 'Miniconda3', 'Anaconda3'))
    return list(dict.fromkeys(roots))


def _conda_executables(roots: list[Path]) -> list[str]:
    candidates = [os.environ.get('CONDA_EXE'), shutil.which('conda'),
                  shutil.which('conda.exe')]
    for root in roots:
        candidates.extend((str(root / 'Scripts' / 'conda.exe'),
                           str(root / 'condabin' / 'conda.bat'),
                           str(root / 'bin' / 'conda')))
    seen, usable = set(), []
    for candidate in candidates:
        if not candidate:
            continue
        key = os.path.normcase(os.path.abspath(candidate))
        if key not in seen and Path(candidate).is_file():
            seen.add(key)
            usable.append(candidate)
    return usable


def _registered_conda_prefixes(home: Path) -> list[str]:
    """Read Conda's environment registry (important for custom -p prefixes)."""
    registered = home / '.conda' / 'environments.txt'
    try:
        return [line.strip() for line in registered.read_text(encoding='utf-8-sig').splitlines()
                if line.strip() and not line.lstrip().startswith('#')]
    except OSError:
        return []


def discover_export_environments(managed_python: str, home: str | Path | None = None,
                                 conda_envs: list[str] | None = None) -> list[dict]:
    """List all discoverable Conda/venv/SenSoRTC interpreters, without probing packages.

    A supplied ``conda_envs`` list skips external Conda CLI calls; it does not
    disable filesystem discovery. The GUI uses this for its immediate first list
    and performs full Conda discovery on a background thread on page open.
    """
    base = Path(home).expanduser() if home is not None else Path.home()
    roots = _candidate_roots(managed_python, base)
    candidates: list[tuple[str, Path]] = [('Studio managed environment', Path(managed_python))]

    # Fast filesystem paths are reliable even when `conda` is not on PATH.
    for root in roots:
        p = python_in_environment(root)
        if p:
            candidates.append((f'Conda base · {root.name}', p))
        env_dir = root / 'envs'
        if env_dir.is_dir():
            try:
                subfolders = sorted((p for p in env_dir.iterdir() if p.is_dir()),
                                    key=lambda p: p.name.lower())
                for folder in subfolders:
                    p = python_in_environment(folder)
                    if p:
                        candidates.append((f'Conda · {folder.name}', p))
            except OSError:
                pass

    for folder in _registered_conda_prefixes(base):
        p = python_in_environment(folder)
        if p:
            candidates.append((f'Conda registered · {Path(folder).name}', p))

    for directory in (base / '.conda' / 'envs', base / '.virtualenvs',
                      base / '.venvs', base / 'venvs',
                      base / '.local' / 'share' / 'virtualenvs'):
        if directory.is_dir():
            try:
                for folder in sorted(directory.iterdir(), key=lambda p: p.name.lower()):
                    if folder.is_dir():
                        p = python_in_environment(folder)
                        if p:
                            candidates.append((f'Python environment · {folder.name}', p))
            except OSError:
                pass

    senso_root = base / '.local' / 'share' / 'SenSoRTC' / 'envs'
    if senso_root.is_dir():
        for folder in sorted(senso_root.glob('install-*'), reverse=True):
            p = python_in_environment(folder)
            if p:
                candidates.append((f'SenSoRTC · {folder.name}', p))

    if conda_envs is None:
        conda_envs = []
        # Locate the executable from the managed prefix and standard locations.
        # `conda` often is not exposed on PATH for Windows desktop shortcuts.
        for exe in _conda_executables(roots):
            try:
                result = subprocess.run([exe, 'env', 'list', '--json'],
                                        capture_output=True, text=True, errors='replace',
                                        timeout=18, check=False)
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    if isinstance(data.get('envs'), list):
                        conda_envs = data['envs']
                        break
            except (OSError, subprocess.TimeoutExpired, ValueError, TypeError):
                continue
    for folder in conda_envs:
        if not isinstance(folder, str):
            continue
        p = python_in_environment(folder)
        if p:
            candidates.append((f'Conda · {Path(folder).name}', p))

    seen: set[str] = set()
    items = []
    for label, path in candidates:
        try:
            resolved = str(path.expanduser().resolve())
        except (OSError, RuntimeError):
            continue
        key = os.path.normcase(resolved)
        if key in seen or not Path(resolved).is_file():
            continue
        items.append({'label': label, 'python': resolved})
        seen.add(key)
    return items


def make_export_process_env(base_env: dict, python: str, managed_python: str) -> dict:
    """Prevent Studio's Conda/Python paths contaminating external interpreters."""
    env = dict(base_env)
    env['YOLO_AUTOINSTALL'] = 'False'
    if Path(python).resolve() != Path(managed_python).resolve():
        for key in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'CONDA_PREFIX',
                    'CONDA_DEFAULT_ENV', 'CONDA_SHLVL', 'LD_LIBRARY_PATH',
                    'LD_PRELOAD', 'QT_QPA_PLATFORM_PLUGIN_PATH'):
            env.pop(key, None)
        prefix = _environment_prefix(python)
        bins = [prefix, prefix / 'Library' / 'mingw-w64' / 'bin',
                prefix / 'Library' / 'usr' / 'bin', prefix / 'Library' / 'bin',
                prefix / 'Scripts'] if os.name == 'nt' else [Path(python).parent]
        env['PATH'] = os.pathsep.join(map(str, bins)) + os.pathsep + env.get('PATH', '')
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
