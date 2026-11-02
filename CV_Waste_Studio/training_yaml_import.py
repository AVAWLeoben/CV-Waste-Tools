"""Read Ultralytics train args / tuned hyperparameters without importing run paths.

The GUI owns model, dataset, output location and run name.  Only documented
training knobs are forwarded as advanced model.train(**kwargs) overrides.
"""
from __future__ import annotations

import math
from pathlib import Path

from augmentation_options import FIELDS, validate_augmentation

# Keep this allowlist explicit: args.yaml also contains inference, export, run
# routing and internal keys that should not be replayed in a new training run.
UI_KEYS = frozenset({'epochs', 'time', 'batch', 'imgsz', 'workers', 'cache', 'cls', 'device'})
ADVANCED_FLOATS = frozenset({
    'lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_epochs',
    'warmup_momentum', 'warmup_bias_lr', 'box', 'dfl', 'pose', 'kobj',
    'dropout', 'fraction', 'bgr', 'shear', 'perspective', 'cutmix',
    'copy_paste', 'cls_pw', 'angle',
    'erasing',
})
ADVANCED_INTS = frozenset({'patience', 'save_period', 'seed', 'nbs', 'mask_ratio'})
ADVANCED_BOOLS = frozenset({
    'save', 'deterministic', 'single_cls', 'rect', 'cos_lr', 'amp',
    'overlap_mask', 'val', 'plots', 'verbose',
})
ADVANCED_STRINGS = frozenset({'optimizer', 'copy_paste_mode', 'auto_augment'})
ADVANCED_SPECIAL = frozenset({'freeze', 'multi_scale', 'pretrained'})
ADVANCED_KEYS = (ADVANCED_FLOATS | ADVANCED_INTS | ADVANCED_BOOLS |
                 ADVANCED_STRINGS | ADVANCED_SPECIAL)


def validate_training_overrides(overrides):
    """Validate a small, safe subset of train kwargs, including native types."""
    if overrides is None:
        return {}
    if not isinstance(overrides, dict):
        raise ValueError('Advanced training overrides must be a dictionary.')
    normalized = {}
    for key, value in overrides.items():
        if key not in ADVANCED_KEYS:
            raise ValueError(f'Unsupported advanced training parameter: {key!r}')
        if value is None:
            continue
        if key in ADVANCED_BOOLS:
            if not isinstance(value, bool):
                raise ValueError(f'{key} must be true or false.')
        elif key in ADVANCED_INTS:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f'{key} must be an integer.')
            if key in {'patience', 'nbs', 'mask_ratio'} and value < 0:
                raise ValueError(f'{key} must not be negative.')
        elif key in ADVANCED_FLOATS:
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f'{key} must be a finite number.')
            if key in {'lr0', 'lrf', 'weight_decay', 'warmup_epochs', 'box', 'dfl', 'pose', 'kobj'} and value < 0:
                raise ValueError(f'{key} must not be negative.')
        elif key in ADVANCED_STRINGS:
            if not isinstance(value, str) or not value.strip() or len(value) > 80:
                raise ValueError(f'{key} must be a nonempty option name.')
        elif key == 'freeze':
            if isinstance(value, bool) or not (
                isinstance(value, int) and value >= 0 or
                isinstance(value, list) and all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in value)
            ):
                raise ValueError('freeze must be a nonnegative integer or list of layer indices.')
        elif key == 'pretrained':
            if not isinstance(value, bool):
                raise ValueError('pretrained must be true or false; choose weights in the GUI.')
        elif key == 'multi_scale':
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError('multi_scale must be a nonnegative number.')
        normalized[key] = value
    return normalized


def _normalise_gui_value(key, value):
    if key == 'cache':
        if value in (None, False):
            return 'off'
        if value is True:
            return 'ram'
        if value in ('off', 'ram', 'disk'):
            return value
        raise ValueError('cache must be off, ram or disk.')
    if key == 'device':
        if value is None or str(value).strip().lower() in {'auto', 'none'}:
            return 'auto'
        if str(value).strip().lower() in {'0', 'cuda:0', 'gpu'}:
            return 'gpu'
        if str(value).strip().lower() == 'cpu':
            return 'cpu'
        raise ValueError('device must be auto, GPU 0 or CPU in this GUI.')
    if key in {'epochs', 'batch', 'imgsz', 'workers'}:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f'{key} must be an integer.')
        if value < (0 if key == 'workers' else 1):
            raise ValueError(f'{key} must be positive (workers may be zero).')
        return value
    if key in {'time', 'cls'}:
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f'{key} must be a finite number.')
        if (key == 'time' and value <= 0) or (key == 'cls' and value < 0):
            raise ValueError(f'{key} must be positive or nonnegative.')
        return value
    return value


def parse_training_yaml(path):
    """Return a preview and train-only arguments from args.yaml or tune YAML.

    No IO paths from the source YAML are trusted or reused. Unsupported fields
    are reported rather than silently interpreted as Ultralytics training args.
    """
    import yaml
    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.suffix.lower() not in {'.yaml', '.yml'}:
        raise ValueError('Choose a readable .yaml or .yml training settings file.')
    if source.stat().st_size > 2_000_000:
        raise ValueError('Training settings YAML is unexpectedly large (over 2 MB).')
    try:
        raw = yaml.safe_load(source.read_text(encoding='utf-8-sig'))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f'Cannot read training YAML: {exc}') from exc
    if not isinstance(raw, dict) or not raw:
        raise ValueError('Training YAML must contain a nonempty mapping of parameter names to values.')
    if any(not isinstance(key, str) for key in raw):
        raise ValueError('All training YAML parameter names must be strings.')

    visible, extras, augmentation, skipped = {}, {}, {}, {}
    errors = []
    for key, value in raw.items():
        if value is None:
            skipped[key] = 'No value'
            continue
        try:
            if key in UI_KEYS:
                visible[key] = _normalise_gui_value(key, value)
            elif key in FIELDS:
                augmentation[key] = value
            elif key in ADVANCED_KEYS:
                extras[key] = value
            else:
                skipped[key] = ('Managed by Studio' if key in {
                    'data', 'model', 'project', 'name', 'save_dir', 'resume',
                    'exist_ok', 'task', 'mode', 'cfg', 'source', 'split'}
                    else 'Not a supported training override')
        except ValueError as exc:
            errors.append(f'{key}: {exc}')
    try:
        augmentation = validate_augmentation(augmentation)
        extras = validate_training_overrides(extras)
    except ValueError as exc:
        errors.append(str(exc))
    if errors:
        raise ValueError('Invalid YAML training settings:\n' + '\n'.join(errors))
    if 'time' in visible and 'epochs' in visible:
        # Ultralytics records both even when a time limit takes precedence.
        visible.pop('epochs')
    if not visible and not augmentation and not extras:
        raise ValueError('No usable training settings found in this YAML.')
    return {
        'source': str(source), 'visible': visible,
        'augmentation': augmentation, 'advanced': extras, 'skipped': skipped,
    }
