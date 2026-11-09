"""Shared configuration with paths relative to the YAML file and strict unknown-field checks."""
import copy
from pathlib import Path

import numpy as np
import yaml

BASE = Path(__file__).resolve().parents[1]


def exact(block, keys, name):
    if not isinstance(block, dict) or set(block) != set(keys):
        raise ValueError(f"{name} must contain exactly {sorted(keys)}; extra fields are rejected.")


def positive(value, name, integer=False):
    if isinstance(value, bool) or (integer and type(value) is not int) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive " + ('integer.' if integer else 'number.'))


def load_config(path, mode, dataset=None):
    path = Path(path).resolve()
    c = yaml.safe_load(path.read_text(encoding='utf-8'))
    exact(c, ('schema_version', 'seed', 'dataset', 'output_root', 'device', 'model', 'loss',
              'response', 'selection', 'sampling', 'profiles', 'evaluation'), 'configuration')
    if c['schema_version'] != 1 or type(c['seed']) is not int or c['seed'] < 0:
        raise ValueError('schema_version must be 1 and seed a nonnegative integer.')
    if c['device'] not in ('auto', 'cpu', 'cuda'):
        raise ValueError('device must be auto, cpu or cuda.')
    exact(c['model'], ('history', 'hidden', 'embedding', 'direct_width', 'head_width', 'solubility', 'init_logit'), 'model')
    for k in ('history', 'hidden', 'embedding', 'direct_width', 'head_width'):
        positive(c['model'][k], 'model.'+k, True)
    positive(c['model']['init_logit'], 'model.init_logit')
    if len(c['model']['solubility']) != 3 or not np.isfinite(c['model']['solubility']).all():
        raise ValueError('solubility requires three finite polynomial coefficients.')
    if c['model']['history'] > 181:
        raise ValueError('history cannot exceed 181.')
    cs = c['model']['solubility']
    ts = np.linspace(19.5, 60.5, 100)
    if np.any(cs[0]+cs[1]*ts+cs[2]*ts**2 <= 0):
        raise ValueError('The solubility polynomial must be positive throughout the nominal coverage interval.')
    exact(c['loss'], ('number', 'mass', 'log', 'change_scaled', 'concentration', 'inventory'), 'loss')
    for k,v in c['loss'].items():
        positive(v, 'loss.'+k)
    exact(c['response'], ('weight', 'groups_per_update', 'number_floor', 'mass_floor'), 'response')
    positive(c['response']['groups_per_update'], 'response.groups_per_update', True)
    for k in ('weight', 'number_floor', 'mass_floor'):
        positive(c['response'][k], 'response.'+k)
    exact(c['sampling'], ('batch_size', 'time_edges', 'grad_clip', 'weight_decay'), 'sampling')
    positive(c['sampling']['batch_size'], 'sampling.batch_size', True)
    positive(c['sampling']['grad_clip'], 'sampling.grad_clip')
    if not np.isfinite(c['sampling']['weight_decay']) or c['sampling']['weight_decay'] < 0:
        raise ValueError('weight_decay cannot be negative.')
    edges = c['sampling']['time_edges']
    if edges[0] != 0 or edges[-1] != 180 or any(type(x) is not int for x in edges) or any(a >= b for a,b in zip(edges[:-1], edges[1:])):
        raise ValueError('time_edges must increase strictly from 0 to 180.')
    exact(c['selection'], ('horizons', 'origins', 'response_weight'), 'selection')
    positive(c['selection']['response_weight'], 'selection.response_weight')
    exact(c['profiles'], ('smoke', 'learn', 'formal'), 'profiles')
    for name, profile in c['profiles'].items():
        exact(profile, ('min_train_parents', 'min_validation_parents', 'stages'), 'profiles.'+name)
        for k in ('min_train_parents', 'min_validation_parents'):
            positive(profile[k], name+'.'+k, True)
        if not profile['stages']:
            raise ValueError('At least one training stage is required.')
        for stage in profile['stages']:
            exact(stage, ('horizon', 'epochs', 'updates_per_epoch', 'lr'), 'stage')
            for k in ('horizon', 'epochs', 'updates_per_epoch'):
                positive(stage[k], 'stage.'+k, True)
            positive(stage['lr'], 'stage.lr')
            if stage['horizon'] > 180:
                raise ValueError('The training horizon cannot exceed 180.')
        if any(a['horizon'] > b['horizon'] for a,b in zip(profile['stages'][:-1], profile['stages'][1:])):
            raise ValueError('The from-scratch horizon curriculum must be nondecreasing.')
    exact(c['evaluation'], ('origins', 'response_horizons', 'direction_d50_tolerance', 'direction_span_tolerance'), 'evaluation')
    for block, key, lo, hi in [('selection', 'horizons', 1, 180), ('selection', 'origins', 0, 179),
                             ('evaluation', 'origins', 0, 179), ('evaluation', 'response_horizons', 1, 180)]:
        a = c[block][key]
        if not a or len(set(a)) != len(a) or any(type(x) is not int or not lo <= x <= hi for x in a):
            raise ValueError(f"{block}.{key} requires distinct valid integers.")
    for k in ('direction_d50_tolerance', 'direction_span_tolerance'):
        positive(c['evaluation'][k], 'evaluation.'+k)
    c = copy.deepcopy(c)
    c['dataset'] = str(Path(dataset).resolve() if dataset else (path.parent/c['dataset']).resolve())
    c['output_root'] = str((path.parent/c['output_root']).resolve())
    c['mode'] = mode
    return c
