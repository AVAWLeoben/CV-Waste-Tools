"""Read a previous HPT result and propose an editable, bounded follow-up search.

A best_hyperparameters.yaml is a seed for a NEW experiment, not a tuner
checkpoint.  Genuine continuation is handled separately using the full run
folder and the original settings/optimizer state.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from hyperparameter_tuning import RANGES, best_trial, read_trials

LOG_PARAMETERS = frozenset(('lr0', 'weight_decay'))


def saved_settings(folder):
    """Return Studio's original run configuration, when available."""
    folder = Path(folder)
    for path in (folder.parent / (folder.name + '.studio_tuning_settings.json'),
                 folder / 'studio_tuning_settings.json'):
        if path.is_file():
            try:
                content = json.loads(path.read_text(encoding='utf-8'))
                if isinstance(content, dict):
                    return content
            except (OSError, ValueError):
                pass
    return None


def ray_has_state(folder):
    """Fast disk-state check; actual Ray Tuner.can_restore is checked in worker."""
    folder = Path(folder)
    return (folder / 'tuner.pkl').is_file() and bool(list(folder.glob('experiment_state*.json')))


def load_tuning_source(path):
    """Read a saved tuning folder or a YAML, returning supported numeric values.

    No untrusted model/data/output locations are imported from arbitrary YAMLs.
    """
    import yaml

    path = Path(path).expanduser().resolve()
    if path.is_dir():
        folder = path
        yaml_path = folder / 'best_hyperparameters.yaml'
    elif path.is_file() and path.suffix.lower() in ('.yaml', '.yml'):
        yaml_path = path
        folder = path.parent if path.name == 'best_hyperparameters.yaml' else None
    else:
        raise ValueError('Choose a saved HPT folder or a best_hyperparameters.yaml file.')
    config = saved_settings(folder) if folder else None
    source_space = (config or {}).get('space')
    if not isinstance(source_space, dict):
        source_space = {}
    winner = None
    values = {}
    if yaml_path.is_file():
        try:
            raw = yaml.safe_load(yaml_path.read_text(encoding='utf-8-sig'))
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError(f'Unable to read best YAML: {exc}') from exc
        if not isinstance(raw, dict):
            raise ValueError('Best hyperparameters YAML must contain a parameter mapping.')
        if isinstance(raw.get('hyperparameters'), dict):
            raw = raw['hyperparameters']
        values = raw
    elif folder:
        winner = best_trial(read_trials(folder))
        if winner is None:
            try:
                from ray_tune_support import read_ray_trials
                winner = best_trial(read_ray_trials(folder))
            except (OSError, ValueError):
                pass
        if winner:
            values = winner['hyperparameters']
        else:
            raise ValueError('No best YAML or successful native/Ray tuning records in this folder.')
    else:
        raise ValueError('Best hyperparameters YAML is missing.')
    numeric = {}
    for key, value in values.items():
        if key not in RANGES or isinstance(value, bool):
            continue
        try:
            center = float(value)
        except (TypeError, ValueError):
            continue
        lo, hi = RANGES[key]
        if math.isfinite(center) and lo <= center <= hi:
            numeric[key] = center
    if not numeric:
        raise ValueError('No supported numeric hyperparameters within the allowed tuning ranges.')
    return {'hyperparameters': numeric, 'source': str(path), 'folder': folder,
            'settings': config, 'previous_space': source_space,
            'winner': winner, 'engine': (config or {}).get('engine')}


def propose_refinement(values, previous_space=None, width=0.20, threshold=0.07):
    """Propose editable bounds and annotate search-boundary hits.

    Search values remain inside the GUI's allowed ranges. Parameters found at
    a previous search boundary are flagged, never silently declared converged.
    """
    if not 0 < width <= 2:
        raise ValueError('Search width must be greater than 0 and at most 200%.')
    if not 0 <= threshold <= 0.5:
        raise ValueError('Boundary threshold must lie between 0 and 0.5.')
    previous_space = previous_space or {}
    proposals = {}
    for key, center in values.items():
        if key not in RANGES:
            continue
        global_lo, global_hi = RANGES[key]
        center = float(center)
        if not math.isfinite(center) or not global_lo <= center <= global_hi:
            continue
        old_bounds = previous_space.get(key)
        try:
            old_lo, old_hi = (float(old_bounds[0]), float(old_bounds[1]))
            if not (global_lo <= old_lo < old_hi <= global_hi):
                raise ValueError('Out-of-range old search bounds')
        except (TypeError, IndexError, KeyError, ValueError):
            old_lo, old_hi = global_lo, global_hi
        fraction = (center - old_lo) / (old_hi - old_lo)
        at_min = fraction <= threshold
        at_max = fraction >= 1-threshold
        # Multiplicative search width is preferable for rates that span orders
        # of magnitude. Zero-valued weight decay gets a small additive width.
        if key in LOG_PARAMETERS and center > 0:
            lo, hi = center / (1 + width), center * (1 + width)
        else:
            radius = max(abs(center) * width, (global_hi - global_lo) * .012)
            lo, hi = center - radius, center + radius
        # A boundary hit calls for continued exploration *past* the previous
        # boundary, where supported, rather than blind narrowing inside it.
        if at_min and old_lo > global_lo:
            lo = min(lo, old_lo - (old_hi - old_lo) * .15)
        if at_max and old_hi < global_hi:
            hi = max(hi, old_hi + (old_hi - old_lo) * .15)
        lo = max(global_lo, lo)
        hi = min(global_hi, hi)
        if lo >= hi:
            radius = (global_hi - global_lo) * 0.01
            lo = max(global_lo, center - radius)
            hi = min(global_hi, center + radius)
        if lo >= hi:
            continue
        flag = ('Near lower bound' if at_min else 'Near upper bound' if at_max else '')
        if flag and (center <= global_lo or center >= global_hi):
            flag += ' (global limit)'
        proposals[key] = {'center': center, 'min': lo, 'max': hi,
                          'warning': flag, 'previous': (old_lo, old_hi)}
    return proposals


def source_resume_status(folder):
    """Return a safe, user-facing resume capability assessment."""
    folder = Path(folder).expanduser().resolve()
    original = saved_settings(folder)
    if not original:
        return False, 'The original Studio settings JSON is missing. Import the best YAML to start a new search.'
    if original.get('engine') == 'ray':
        if not ray_has_state(folder):
            return False, 'Ray Tuner state is missing. A best YAML cannot restore Ray history.'
        return True, 'Ray can restore unfinished trials from its saved state, subject to a runtime compatibility check.'
    if (folder / 'tune_results.ndjson').is_file():
        return True, 'Native tuning history is available for continuation.'
    return False, 'Native tune_results.ndjson is missing; use a new refinement search.'
