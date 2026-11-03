"""Dependency probing and Ray result adaptation for the optional Bayesian tuner.

No Ray, PyTorch, CUDA or GUI imports at module import time.  The probe is
executed by a separate interpreter from the selected Conda environment.
"""
from __future__ import annotations

import csv
import inspect
import json
import math
from pathlib import Path

INSTALL_PACKAGES = ('ray[tune]>=2.41.0', 'optuna>=3')


def dependency_report():
    """Return a JSON-friendly compatibility report from the *running* Python."""
    import sys
    from importlib import metadata
    result = {'available': False, 'python': sys.executable,
              'packages': list(INSTALL_PACKAGES), 'reason': '', 'versions': {}}
    try:
        import ultralytics
        from ultralytics import YOLO
        from ultralytics.utils.tuner import run_ray_tune
        from ultralytics.utils.callbacks import raytune
        from ray import tune
        from ray.tune.search.optuna import OptunaSearch
        from ray.tune.schedulers import ASHAScheduler
        import optuna
        for name in ('ultralytics','ray','optuna'):
            result['versions'][name] = metadata.version(name)
        from packaging.version import Version
        if Version(result['versions']['ray']) < Version('2.41.0'):
            raise RuntimeError('Ray Tune 2.41 or newer is required by this Ultralytics integration.')
        if Version(result['versions']['optuna']) < Version('3.0.0'):
            raise RuntimeError('Optuna 3 or newer is required for TPE.')
        result['versions']['pydantic'] = metadata.version('pydantic')
        if not Version('2.0.0') <= Version(result['versions']['pydantic']) < Version('2.12.0'):
            raise RuntimeError('Installed Ultralytics Ray Tune requires pydantic>=2,<2.12.')
        sig = inspect.signature(run_ray_tune)
        required = ('space', 'grace_period', 'gpu_per_trial', 'iterations', 'search_alg')
        if not all(x in sig.parameters for x in required):
            raise RuntimeError('This Ultralytics version lacks Ray TPE / ASHA tuning arguments.')
        if 'use_ray' not in inspect.signature(YOLO.tune).parameters:
            raise RuntimeError('Ultralytics YOLO.tune() does not expose use_ray.')
        if not callable(getattr(tune, 'get_context', None)):
            raise RuntimeError('Installed Ray Tune lacks get_context() for epoch reporting.')
        if not raytune.callbacks.get('on_fit_epoch_end'):
            raise RuntimeError('Ultralytics Ray epoch reporting callback is unavailable.')
        result['available'] = True
    except Exception as exc:
        result['reason'] = f'{type(exc).__name__}: {exc}'
    return result


def ray_space(bounds):
    """Transform native (min, max) bounds into Ray Tune distributions."""
    if bounds is None:
        return None  # Ultralytics supplies its Ray-native default space
    from ray import tune
    return {key: (tune.randint(int(lo), int(hi)+1) if key == 'close_mosaic'
                  else tune.uniform(float(lo), float(hi)))
            for key, (lo, hi) in bounds.items()}


def _finite(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def _last_record(path):
    """Read JSON-lines trial metrics; incomplete last lines are ignored."""
    last = None
    try:
        with path.open(encoding='utf-8', errors='replace') as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                    if isinstance(row, dict):
                        last = row
                except ValueError:
                    continue
    except OSError:
        pass
    return last


def read_ray_trials(folder, max_trials=1000):
    """Adapt Ray result.json / progress.csv to Studio's existing chart shape."""
    folder = Path(folder)
    records = []
    # Files are written by Ray, not by a custom training loop.
    json_logs = sorted(folder.glob('**/result.json'))[:max_trials]
    csv_logs = sorted(folder.glob('**/progress.csv'))[:max_trials]
    sources = {p.parent: p for p in csv_logs}
    sources.update({p.parent: p for p in json_logs})
    for directory, path in sorted(sources.items(), key=lambda pair: str(pair[0])):
        if path.suffix == '.json':
            row = _last_record(path)
        else:
            row = None
            try:
                with path.open(newline='', encoding='utf-8', errors='replace') as f:
                    for record in csv.DictReader(f):
                        row = record
            except OSError:
                pass
        if not row:
            continue
        # Ray records parameters in config/<name> as well as config dictionaries.
        hp = dict(row.get('config') or {}) if isinstance(row.get('config'), dict) else {}
        for k, v in row.items():
            if k.startswith('config/'):
                hp[k[7:]] = v
        metrics = {k: v for k, v in row.items() if k.startswith(('metrics/','val/','train/'))}
        score = _finite(row.get('metrics/mAP50-95(B)'))
        if score is None:
            score = _finite(row.get('fitness'))
        if score is None:
            continue
        trial_epochs = int(_finite(row.get('epoch')) or 0)
        records.append({'iteration': len(records)+1, 'fitness': score,
                        'hyperparameters': {k: float(v) if _finite(v) is not None else v
                                            for k, v in hp.items()},
                        'metrics': metrics, 'failed': False, 'epochs': trial_epochs,
                        'trial_id': directory.name, 'status': 'Recorded',
                        'trial_dir': str(directory)})
    status_file = folder / 'studio_ray_trials.json'
    if status_file.is_file():
        try:
            statuses = json.loads(status_file.read_text(encoding='utf-8'))
            for record in records:
                record['status'] = statuses.get(record['trial_id'], record['status'])
        except (OSError, ValueError):
            pass
    return records


def export_ray_best(folder, result_grid, search_space_keys, total_epochs=None):
    """Persist an interoperable native-style best_hyperparameters.yaml.

    Only actual tuned hyperparameters are written, never old dataset/model
    paths or Ray internal metadata.  No checkpoint is fabricated.
    """
    import yaml
    folder = Path(folder)
    candidates = []
    statuses = {}
    for result in result_grid:
        config = getattr(result, 'config', {}) or {}
        metrics = getattr(result, 'metrics', {}) or {}
        score = _finite(metrics.get('metrics/mAP50-95(B)'))
        if score is None:
            score = _finite(metrics.get('fitness'))
        trial_path = str(getattr(result, 'path', ''))
        if trial_path:
            epoch = _finite(metrics.get('epoch')) or _finite(metrics.get('training_iteration'))
            if getattr(result, 'error', None):
                status = 'Failed'
            elif epoch is not None and total_epochs and epoch < total_epochs:
                status = 'Early stop'
            else:
                status = 'Finished'
            statuses[Path(trial_path).name] = status
        if score is not None and not getattr(result, 'error', None):
            candidates.append((score, config, result))
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'studio_ray_trials.json').write_text(
        json.dumps(statuses, indent=2), encoding='utf-8')
    if not candidates:
        return None
    score, config, selected = max(candidates, key=lambda item: item[0])
    values = {k: float(config[k]) for k in search_space_keys
              if k in config and _finite(config[k]) is not None}
    if not values:
        return None
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'best_hyperparameters.yaml').write_text(
        '# Best Ray Tune / Optuna TPE + ASHA search values\n'
        + yaml.safe_dump(values, sort_keys=False), encoding='utf-8')
    (folder / 'studio_ray_best.json').write_text(json.dumps({
        'fitness': score, 'parameters': values,
        'trial_path': str(getattr(selected, 'path', ''))}, indent=2), encoding='utf-8')
    return values
