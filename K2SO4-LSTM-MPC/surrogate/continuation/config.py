"""Strict continuation plans; architecture, losses and labels are inherited from the parent checkpoint."""
import copy
import math
from pathlib import Path

import yaml

BASE = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = BASE / 'mpc_continuation.yaml'
ROOT_KEYS = {'schema_version', 'parent_run', 'checkpoint', 'output_root', 'seed', 'device', 'profiles'}
PLAN_KEYS = {'epochs', 'updates_per_epoch', 'horizon', 'batch_size', 'train_per_family',
             'validation_per_family', 'development_per_family', 'lr', 'every_epochs', 'full_every_epochs'}


def _integer(value, label, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(label + f"must be an integer >= {minimum}.")


def load_settings(path=DEFAULT_CONFIG, mode='formal'):
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(raw, dict) or set(raw) != ROOT_KEYS:
        raise ValueError('Continuation settings have missing or unknown fields.')
    if type(raw['schema_version']) is not int or raw['schema_version'] != 1:
        raise ValueError('Continuation schema_version must be 1.')
    if mode not in ('smoke', 'formal'):
        raise ValueError('Continuation supports smoke or formal mode.')
    if raw['checkpoint'] != 'best':
        raise ValueError('Continuation starts from best weights.')
    if raw['device'] not in ('auto', 'cpu', 'cuda'):
        raise ValueError('device must be auto, cpu or cuda.')
    _integer(raw['seed'], 'seed', 0)
    if not isinstance(raw['profiles'], dict) or set(raw['profiles']) != {'smoke', 'formal'}:
        raise ValueError('profiles must contain exactly smoke and formal.')
    for name, profile in raw['profiles'].items():
        if not isinstance(profile, dict) or set(profile) != PLAN_KEYS:
            raise ValueError(name + 'Plan fields are missing or unknown.')
        for key in PLAN_KEYS - {'lr'}:
            _integer(profile[key], name + '.' + key, 0 if key.endswith('_per_family') else 1)
        if profile['horizon'] > 180:
            raise ValueError('horizon cannot exceed 180.')
        if isinstance(profile['lr'], bool) or not isinstance(profile['lr'], (int, float)):
            raise ValueError('lr must be finite and positive.')
        if not math.isfinite(profile['lr']) or profile['lr'] <= 0:
            raise ValueError('lr must be finite and positive.')
    formal = raw['profiles']['formal']
    if any(formal[k] != 0 for k in ('train_per_family', 'validation_per_family', 'development_per_family')):
        raise ValueError('Formal continuation must inherit the parent train/validation/development sets without reselection.')
    result = copy.deepcopy(raw)
    for key in ('parent_run', 'output_root'):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ValueError(key + 'must be a nonempty directory path.')
        value = Path(raw[key])
        result[key] = str((path.parent / value).resolve() if not value.is_absolute() else value.resolve())
    result.update(mode=mode, plan=copy.deepcopy(raw['profiles'][mode]))
    return result
