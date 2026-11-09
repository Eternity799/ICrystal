"""Control-oriented schema-2 configuration with strict field validation."""
import copy
import math
from pathlib import Path

import yaml

from ..config import load_config as load_base

BASE = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = BASE/'mpc_training.yaml'
ARMS = ('replay_csd', 'control_priority')
QUALITY_KEYS = ('d50_um', 'span', 'concentration', 'yield_fraction')


def exact(block, keys, label):
    if not isinstance(block, dict) or set(block) != set(keys):
        raise ValueError(f"{label} must contain exactly {sorted(keys)}.")


def number(value, label, minimum=0, integer=False):
    if isinstance(value, bool) or (integer and type(value) is not int) or not isinstance(value, (int, float)):
        raise ValueError(label+' requires a numeric value.')
    if not math.isfinite(value) or value < minimum:
        raise ValueError(label+f" must be finite and at least {minimum}.")


def weights(block, keys, label):
    exact(block, keys, label)
    for k, value in block.items():
        number(value, label+'.'+k)
    if not any(block.values()):
        raise ValueError(label+' cannot contain only zeros.')


def load_config(path=DEFAULT_CONFIG, mode='smoke', dataset=None, arm='all'):
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding='utf-8'))
    exact(raw, ('schema_version', 'base_config', 'seed', 'dataset', 'output_root', 'device', 'arms',
                'objective_config', 'quality_weights', 'auxiliary_loss', 'absolute_scale_mode','scale_floors', 'response',
                'selection', 'profiles', 'evaluation'), 'control-training configuration')
    if raw['schema_version'] != 2 or mode not in ('smoke', 'pilot', 'formal'):
        raise ValueError('schema_version must be 2 and mode smoke, pilot or formal.')
    number(raw['seed'], 'seed', integer=True)
    if raw['device'] not in ('auto', 'cpu', 'cuda'):
        raise ValueError('device must be auto, cpu or cuda.')
    if raw['absolute_scale_mode'] not in ('process','train'):
        raise ValueError('absolute_scale_mode must be process or train.')
    if not raw['arms'] or len(set(raw['arms'])) != len(raw['arms']) or not set(raw['arms']) <= set(ARMS):
        raise ValueError('arms must contain distinct replay_csd/control_priority entries.')
    if arm != 'all' and arm not in ARMS:
        raise ValueError('Unknown comparison arm.')
    if mode == 'formal' and arm == 'all' and len(raw['arms']) > 1:
        raise ValueError('Formal mode trains one selected arm; specify --arm replay_csd or --arm control_priority.')
    weights(raw['quality_weights'], QUALITY_KEYS, 'quality_weights')
    weights(raw['auxiliary_loss'], ('number','mass','log','change_scaled','concentration','inventory'), 'auxiliary_loss')
    exact(raw['scale_floors'], ('absolute','response'), 'scale_floors')
    for kind, block in raw['scale_floors'].items():
        exact(block, QUALITY_KEYS, 'scale_floors.'+kind)
        for key, value in block.items():
            number(value, 'scale_floors.'+kind+'.'+key, 1e-30)
    exact(raw['response'], ('weight','groups_per_update','mixture'), 'response')
    number(raw['response']['weight'], 'response.weight', 1e-30)
    number(raw['response']['groups_per_update'], 'response.groups_per_update', 1, True)
    weights(raw['response']['mixture'], ('short','long','terminal'), 'response.mixture')
    exact(raw['selection'], ('horizons','origins','every_epochs','full_every_epochs',
          'response_groups_per_parent','response_weight','terminal_weight','ranking_weight','tie_tolerance'), 'selection')
    for key in ('every_epochs','full_every_epochs','response_groups_per_parent'):
        number(raw['selection'][key], 'selection.'+key, 1, True)
    for key in ('response_weight','terminal_weight','ranking_weight','tie_tolerance'):
        number(raw['selection'][key], 'selection.'+key)
    exact(raw['profiles'], ('smoke','pilot','formal'), 'profiles')
    for name, profile in raw['profiles'].items():
        exact(profile, ('train_per_family','validation_per_family','development_per_family','stages','patience'), name)
        for key in ('train_per_family','validation_per_family','development_per_family','patience'):
            number(profile[key], name+'.'+key, integer=True)
        if not profile['stages']:
            raise ValueError('At least one curriculum stage is required.')
        for stage in profile['stages']:
            exact(stage, ('horizon','epochs','updates_per_epoch','lr','full_terminal'), 'stage')
            for key in ('horizon','epochs','updates_per_epoch'):
                number(stage[key], 'stage.'+key, 1, True)
            if stage['horizon'] > 180 or type(stage['full_terminal']) is not bool:
                raise ValueError('Training H cannot exceed 180; full_terminal must be Boolean.')
            number(stage['lr'], 'stage.lr', 1e-30)
        if any(a['horizon'] > b['horizon'] for a,b in zip(profile['stages'], profile['stages'][1:])):
            raise ValueError('The horizon curriculum must be nondecreasing.')
    exact(raw['evaluation'], ('origins','response_horizons','direction_tolerance'), 'evaluation')
    weights(raw['evaluation']['direction_tolerance'], QUALITY_KEYS, 'evaluation.direction_tolerance')
    for block,key,lo,hi in ((raw['selection'],'horizons',1,180),(raw['selection'],'origins',0,179),
                          (raw['evaluation'],'origins',0,179),(raw['evaluation'],'response_horizons',1,180)):
        values = block[key]
        if not values or len(set(values)) != len(values) or any(type(x) is not int or not lo <= x <= hi for x in values):
            raise ValueError(key+' requires distinct valid integers.')
    base_path = (path.parent/raw['base_config']).resolve()
    base = load_base(base_path, 'formal' if mode == 'formal' else 'learn')
    result = copy.deepcopy(base)
    result.update(copy.deepcopy(raw))
    for key in ('base_config','dataset','output_root','objective_config'):
        result[key] = str((path.parent/raw[key]).resolve())
    if dataset is not None:
        result['dataset'] = str(Path(dataset).resolve())
    result['mode'] = mode
    result['arms'] = raw['arms'] if arm == 'all' else [arm]
    result['sampling'] = base['sampling']
    result['model'] = base['model']
    result['loss'] = base['loss']
    result['legacy_response'] = base['response']
    return result
