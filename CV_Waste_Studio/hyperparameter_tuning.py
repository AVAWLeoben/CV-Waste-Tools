"""Native Ultralytics tuner integration and safe, dependency-light results reader.

No search algorithm or custom training loop lives here: only validation, launch
arguments, and parsing Ultralytics' own tune_results.ndjson output.
"""
from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path

# Bounded ranges for the supported detection hyperparameters. Full native mode
# omits the space argument and lets the installed Ultralytics choose its space.
RANGES = {
    'lr0': (1e-5, 1e-2), 'lrf': (0.01, 1.0),
    'momentum': (0.7, .98), 'weight_decay': (0., .001),
    'warmup_epochs': (0., 5.), 'warmup_momentum': (0., .95),
    'box': (1., 20.), 'cls': (.1, 4.), 'dfl': (.4, 12.),
    'hsv_h': (0., .1), 'hsv_s': (0., .9), 'hsv_v': (0., .9),
    'degrees': (0., 45.), 'translate': (0., .9),
    'scale': (0., .95), 'shear': (0., 10.),
    'perspective': (0., .001), 'flipud': (0., 1.),
    'fliplr': (0., 1.), 'mosaic': (0., 1.),
    'mixup': (0., 1.), 'close_mosaic': (0., 10.),
}
PRESETS = {
    'Core optimisation': ('lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_epochs', 'box', 'cls', 'dfl'),
    'Augmentation': ('hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 'scale', 'fliplr', 'mosaic', 'mixup'),
    'Balanced': ('lr0', 'lrf', 'momentum', 'weight_decay', 'box', 'cls',
                 'hsv_s', 'hsv_v', 'translate', 'scale', 'mosaic'),
    'Full native': (),
}


def _int(value, label, low=0, high=100000):
    if isinstance(value, bool):
        raise ValueError(f'{label} must be an integer from {low} to {high}.')
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{label} must be an integer from {low} to {high}.') from exc
    if str(number) != str(value).strip() or not low <= number <= high:
        raise ValueError(f'{label} must be an integer from {low} to {high}.')
    return number


def validate_settings(config):
    """Validate before launching a subprocess; do not touch source splits."""
    if not isinstance(config, dict):
        raise ValueError('Tuning configuration must be an object.')
    from train_and_split_tool_GUI import resolve_existing_dataset_yaml, _normalize_device_choice
    dataset = resolve_existing_dataset_yaml(config.get('data', ''))
    model = str(config.get('model', 'yolo11l.pt')).strip().strip('"')
    if not model or not model.lower().endswith('.pt'):
        raise ValueError('Select pretrained YOLO detection weights or a local .pt file.')
    if '/' in model or '\\' in model or Path(model).is_absolute():
        if not Path(model).is_file():
            raise ValueError(f'Weights file does not exist: {model}')
        model = str(Path(model).resolve())
    output = Path(config.get('output_dir') or Path(dataset['yaml']).parent / 'hyperparameter_tuning').expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise ValueError('Tuning results path must be a directory.')
    name = str(config.get('name', '')).strip()
    if name and (name in ('.', '..') or Path(name).name != name or any(x in name for x in '/\\:')):
        raise ValueError('Run name must be a simple folder name.')
    epochs = _int(config.get('epochs', 25), 'Epochs per trial', 1, 1000)
    trials = _int(config.get('iterations', 20), 'Tuning trials', 2, 1000)
    imgsz = _int(config.get('imgsz', 640), 'Image size', 32, 8192)
    batch = _int(config.get('batch', 8), 'Batch size', 1, 1024)
    workers = _int(config.get('workers', 8), 'Dataloader workers', 0, 128)
    cache = str(config.get('cache', 'off')).lower()
    if cache not in ('off', 'disk', 'ram'):
        raise ValueError('Cache must be off, disk or ram.')
    optimizer = str(config.get('optimizer', 'AdamW')).strip()
    if optimizer not in ('auto', 'SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp'):
        raise ValueError('Unsupported optimiser.')
    space = config.get('space')
    if space is not None:
        if not isinstance(space, dict) or not space:
            raise ValueError('Select at least one parameter, or use Full native space.')
        normalized = {}
        for key, bounds in space.items():
            if key not in RANGES:
                raise ValueError(f'Unsupported tuning parameter: {key}')
            if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
                raise ValueError(f'{key}: specify minimum and maximum.')
            try:
                lo, hi = float(bounds[0]), float(bounds[1])
            except (ValueError, TypeError) as exc:
                raise ValueError(f'{key}: bounds must be numbers.') from exc
            limit_lo, limit_hi = RANGES[key]
            if not (math.isfinite(lo) and math.isfinite(hi) and
                    limit_lo <= lo < hi <= limit_hi):
                raise ValueError(f'{key}: bounds must satisfy {limit_lo} <= min < max <= {limit_hi}.')
            normalized[key] = (lo, hi)
        space = normalized
    resume = bool(config.get('resume', False))
    if resume and not name:
        raise ValueError('To resume, select a named run folder.')
    if resume and not (output / name / 'tune_results.ndjson').is_file():
        raise ValueError('Selected run has no tune_results.ndjson to resume.')
    return {
        'data': str(dataset['yaml']), 'model': model, 'output_dir': str(output),
        'name': name, 'epochs': epochs, 'iterations': trials, 'batch': batch,
        'imgsz': imgsz, 'workers': workers, 'cache': cache,
        'optimizer': optimizer, 'device': _normalize_device_choice(config.get('device', 'auto')),
        'space': space, 'resume': resume,
    }


def tune_arguments(settings, run_name):
    """Arguments passed verbatim to the native model.tune(**kwargs)."""
    d = {'data': settings['data'], 'epochs': settings['epochs'],
         'iterations': settings['iterations'], 'batch': settings['batch'],
         'imgsz': settings['imgsz'], 'workers': settings['workers'],
         'cache': False if settings['cache'] == 'off' else settings['cache'],
         'optimizer': settings['optimizer'], 'project': settings['output_dir'],
         'name': run_name, 'exist_ok': settings['resume'],
         'resume': settings['resume'], 'plots': False,
         'save': True, 'val': True, 'device': settings['device']}
    if settings['space'] is not None:
        d['space'] = settings['space']
    return d


def read_trials(folder):
    """Read complete NDJSON records, tolerating an in-progress final line.

    Each record comes from Ultralytics, not a reconstructed training result.
    """
    file = Path(folder) / 'tune_results.ndjson'
    if not file.is_file():
        return []
    trials = []
    with file.open('r', encoding='utf-8', errors='replace') as stream:
        for line in stream:
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    index = _int(item.get('iteration', len(trials) + 1), 'Trial', 1)
                    fitness = float(item.get('fitness', 0.))
                    if not math.isfinite(fitness):
                        continue
                    measurements = item.get('metrics') or {}
                    if not isinstance(measurements, dict):
                        measurements = {}
                    datasets = item.get('datasets') or {}
                    if isinstance(datasets, dict) and datasets:
                        measurements = next((v for v in datasets.values() if isinstance(v, dict)), measurements)
                    actual_metrics = any(str(k).startswith(('metrics/', 'val/', 'train/'))
                                         for k in measurements)
                    failed = not actual_metrics or bool(item.get('failed_datasets'))
                    trials.append({'iteration': index, 'fitness': fitness,
                                   'hyperparameters': item.get('hyperparameters') or {},
                                   'metrics': measurements, 'failed': failed})
            except (ValueError, TypeError, json.JSONDecodeError):
                continue
    return sorted(trials, key=lambda record: record['iteration'])


def best_trial(trials):
    valid = [t for t in trials if not t['failed']]
    return max(valid, key=lambda t: t['fitness']) if valid else None


def create_run_name():
    return 'Tune_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')


def run_native_tuning(settings):
    """Heavy imports only inside worker, after Windows __main__ guard."""
    import os
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')  # existing app workaround, worker only
    from ultralytics import YOLO
    model = YOLO(settings['model'])
    name = settings['name'] or create_run_name()
    root = Path(settings['output_dir'])
    folder = root / name
    if folder.exists() and not settings['resume']:
        raise FileExistsError(f'Tuning folder already exists: {folder}')
    root.mkdir(parents=True, exist_ok=True)
    # Sidecar sits outside tune_dir, so an interrupted/forced-stopped run can
    # be resumed with exactly the original training/search configuration.
    (root / (name + '.studio_tuning_settings.json')).write_text(
        json.dumps(settings, indent=2), encoding='utf-8')
    # Only select existing folder with explicit resume; no overwrite of old runs.
    # Leave the destination absent: Ultralytics get_save_dir() otherwise
    # silently increments the name and the GUI would monitor the wrong folder.
    args = tune_arguments(settings, name)
    print('STUDIO_TUNE_FOLDER ' + str(folder), flush=True)
    print(f"Native Ultralytics model.tune(): {settings['iterations']} trials, "
          f"{settings['epochs']} epochs/trial, workers={settings['workers']}, "
          f"cache={settings['cache']}", flush=True)
    # Native tuner spawns individual train processes; no parallel custom trainers.
    # No invalid device='auto'/'gpu' strings are forwarded to Ultralytics.
    import torch
    if args['device'] == 'auto':
        args['device'] = 0 if torch.cuda.is_available() else 'cpu'
    elif args['device'] == 'gpu':
        if not torch.cuda.is_available():
            raise RuntimeError('GPU selected but CUDA is unavailable.')
        args['device'] = 0
    # Protect pre-existing image caches. Native tuner runs many child trainers;
    # clean only images absent at the start, never delete other users' .npy files.
    pending_caches = set()
    manifest = None
    if settings['cache'] == 'disk':
        import yaml
        from train_and_split_tool_GUI import _disk_cache_candidates, _save_disk_cache_manifest
        dataset_cfg = yaml.safe_load(Path(settings['data']).read_text(encoding='utf-8-sig'))
        pending_caches = {p for p in _disk_cache_candidates(settings['data'], dataset_cfg)
                          if not p.exists()}
        manifest = root / (name + '_disk_cache_manifest.json')
        _save_disk_cache_manifest(manifest, pending_caches)
    try:
        model.tune(use_ray=False, **args)
    finally:
        if pending_caches:
            from train_and_split_tool_GUI import _remove_new_disk_cache
            _remove_new_disk_cache(pending_caches, manifest)
        elif manifest is not None:
            manifest.unlink(missing_ok=True)
    (folder / 'studio_tuning_settings.json').write_text(
        json.dumps(settings, indent=2), encoding='utf-8')
    trials = read_trials(folder)
    winner = best_trial(trials)
    if winner is None:
        raise RuntimeError('No successful tuning trial was recorded. Review the worker log.')
    print('STUDIO_TUNE_COMPLETE ' + str(folder), flush=True)
    print(f"Best trial {winner['iteration']} · fitness {winner['fitness']:.5f}", flush=True)
    return folder
