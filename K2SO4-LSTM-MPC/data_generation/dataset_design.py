"""Temperature profiles and state-action branch plans."""
from __future__ import annotations
import numpy as np
PATTERNS = ('offset', 'rate_block', 'change_then_hold')
INITIAL_KEYS = ('temperature_c', 'saturation', 'seed_d50_um', 'seed_sigma_g', 'seed_mass_fraction')

def _keys(block, allowed, name):
    if not isinstance(block, dict) or set(block) - set(allowed):
        raise ValueError(f'{name} contains unsupported fields or is not a configuration mapping.')

def _positive_int(value, name, zero=False):
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError(f"{name} must be a {('nonnegative' if zero else 'positive')} integer.")

def _edges(values, name):
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or len(a) < 2 or (not np.isfinite(a).all()) or np.any(np.diff(a) <= 0):
        raise ValueError(f'{name} must contain finite, strictly increasing edges.')

def validate_design(c):
    revision = c.get('design_revision', 1)
    if type(revision) is not int or revision not in (1, 2):
        raise ValueError('design_revision must be 1 or 2; the data schema remains 1.')
    _keys(c['counts'], ('smoke', 'pilot', 'formal'), 'counts')
    for mode, counts in c['counts'].items():
        if set(counts) != {'train', 'validation', 'test'}:
            raise ValueError(f'counts.{mode} must specify all three roles.')
        for value in counts.values():
            _positive_int(value, f'counts.{mode}')
    _keys(c['coverage'], ('temperature_c', 'rate_c_min', 'endpoint_c', 'arrival_min', 'knots', 'short_segment_probability', 'short_durations_min'), 'coverage')
    _keys(c['initial'], (*INITIAL_KEYS, 'nominal_every'), 'initial')
    _keys(c['branches'], ('origins_min', 'horizon_steps', 'offsets_c', 'strata_min', 'origins_per_stratum', 'patterns', 'durations_steps', 'long_horizon_steps', 'long_groups_per_parent', 'terminal_every', 'terminal_origins_min'), 'branches')
    _keys(c['quality'], ('mass_relative_tolerance', 'reproduction_relative_l1', 'schedule_attempts', 'coverage', 'numerical', 'require_positive_crystals', 'trajectory_attempts'), 'quality')
    if revision == 1:
        if any((k in c['branches'] for k in ('patterns', 'strata_min', 'terminal_every'))):
            raise ValueError('The new action design requires design_revision: 2.')
        return
    _positive_int(c['initial']['nominal_every'], 'initial.nominal_every', zero=True)
    probability = c['coverage']['short_segment_probability']
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError('The short-segment probability must be between 0 and 1.')
    durations = c['coverage']['short_durations_min']
    if not durations or len(set(durations)) != len(durations):
        raise ValueError('Short-segment durations must be nonempty and distinct.')
    for duration in durations:
        _positive_int(duration, 'coverage.short_durations_min')
    if np.ceil(c['coverage']['arrival_min'][0]) > np.floor(c['coverage']['arrival_min'][1]):
        raise ValueError('The arrival-time range must contain an integer minute.')
    b = c['branches']
    _edges(b['strata_min'], 'branches.strata_min')
    if b['strata_min'][0] != 0 or b['strata_min'][-1] != 180:
        raise ValueError('Branch-stratum edges must span 0 to 180 minutes.')
    if any((type(x) is not int for x in b['strata_min'])):
        raise ValueError('Branch-stratum edges must be integer minutes.')
    _positive_int(b['origins_per_stratum'], 'branches.origins_per_stratum')
    for lo, hi in zip(b['strata_min'][:-1], b['strata_min'][1:]):
        if sum((lo <= x < hi for x in b['origins_min'])) < b['origins_per_stratum']:
            raise ValueError(f'The origin pool in [{lo},{hi}) is too small for stratified sampling.')
    if not b['patterns'] or len(set(b['patterns'])) != len(b['patterns']) or (not set(b['patterns']) <= set(PATTERNS)):
        raise ValueError('Action patterns are empty, duplicated or unsupported.')
    if not b['durations_steps'] or len(set(b['durations_steps'])) != len(b['durations_steps']):
        raise ValueError('Action durations are empty or duplicated.')
    for duration in b['durations_steps']:
        _positive_int(duration, 'branches.durations_steps')
    positives = {float(x) for x in b['offsets_c'] if x > 0}
    negatives = {-float(x) for x in b['offsets_c'] if x < 0}
    if not positives or positives != negatives:
        raise ValueError('The revision-2 offset pool must contain symmetric positive/negative scales and exactly one zero baseline.')
    _positive_int(b['long_horizon_steps'], 'branches.long_horizon_steps')
    if not b['horizon_steps'] <= b['long_horizon_steps'] <= 180:
        raise ValueError('Long-branch horizons must be at least the ordinary horizon and at most 180 steps.')
    _positive_int(b['long_groups_per_parent'], 'branches.long_groups_per_parent', zero=True)
    local_count = (len(b['strata_min']) - 1) * b['origins_per_stratum']
    if b['long_groups_per_parent'] > local_count:
        raise ValueError('The long-branch group count exceeds the local-branch budget.')
    minimum_eligible = sum((max(0, b['origins_per_stratum'] - sum((x + b['long_horizon_steps'] > 180 for x in b['origins_min'] if lo <= x < hi))) for lo, hi in zip(b['strata_min'][:-1], b['strata_min'][1:])))
    if b['long_groups_per_parent'] > minimum_eligible:
        raise ValueError('Stratified selection cannot guarantee enough origins for complete long branches; adjust the origin pool or long horizon.')
    _positive_int(b['terminal_every'], 'branches.terminal_every', zero=True)
    terminal = b['terminal_origins_min']
    if not terminal or len(set(terminal)) != len(terminal) or any((type(x) is not int or not 0 <= x < 180 for x in terminal)):
        raise ValueError('Terminal-branch origins must be distinct integer minutes from 0 to 179.')
    q = c['quality']['coverage']
    _keys(q, ('temperature_edges', 'saturation_edges', 'rate_edges', 'small_size_um', 'large_size_um'), 'quality.coverage')
    for key in ('temperature_edges', 'saturation_edges', 'rate_edges'):
        _edges(q[key], f'quality.coverage.{key}')
    if not np.isfinite([q['small_size_um'], q['large_size_um']]).all() or not 0 < q['small_size_um'] < q['large_size_um']:
        raise ValueError('Small/large size thresholds must be finite, positive and ascending.')
    n = c['quality']['numerical']
    _keys(n, ('enabled', 'modes', 'parents_per_family', 'time_step_factor', 'grid_factor'), 'quality.numerical')
    if type(n['enabled']) is not bool or not n['modes'] or (not set(n['modes']) <= set(c['counts'])):
        raise ValueError('Invalid numerical-check enabled flag or modes.')
    _positive_int(n['parents_per_family'], 'quality.numerical.parents_per_family')
    if not np.isfinite(n['time_step_factor']) or not 0 < n['time_step_factor'] < 1:
        raise ValueError('The time-step refinement factor must lie strictly between 0 and 1.')
    _positive_int(n['grid_factor'], 'quality.numerical.grid_factor')
    if n['grid_factor'] < 2:
        raise ValueError('The grid-refinement factor must be at least 2.')
    if any((n['parents_per_family'] > c['counts'][mode]['train'] for mode in n['modes'])):
        raise ValueError('The numerical-check parent count cannot exceed the per-family training count in any selected mode.')
    if type(c['quality']['require_positive_crystals']) is not bool:
        raise ValueError('require_positive_crystals must be a boolean.')
    _positive_int(c['quality']['trajectory_attempts'], 'quality.trajectory_attempts')

def temperature_nodes(c, family, start, rng):
    """Use integer-minute nodes and balance mixed short/long segments to the total temperature drop."""
    endpoint = rng.uniform(*c['endpoint_c'])
    arrival = int(rng.integers(int(np.ceil(c['arrival_min'][0])), int(np.floor(c['arrival_min'][1])) + 1))
    times = [0]
    long_mean = arrival / c['knots']
    while times[-1] < arrival:
        if rng.random() < c['short_segment_probability']:
            duration = int(rng.choice(c['short_durations_min']))
        else:
            duration = max(1, int(round(rng.uniform(0.7, 1.3) * long_mean)))
        times.append(min(arrival, times[-1] + duration))
    times = np.asarray(times, dtype=float)
    weights = rng.uniform(0.3, 1.7, len(times) - 1)
    if family == 'hold':
        weights[rng.choice(len(weights), max(1, len(weights) // 4), replace=False)] = 0.0
    drops = weights * np.diff(times)
    nodes = start + (endpoint - start) * np.r_[0.0, np.cumsum(drops) / drops.sum()]
    if family == 'local_reheat':
        j = int(rng.integers(0, len(weights) - 1))
        maximum = min(1.5, c['rate_c_min'][1] * (times[j + 1] - times[j]))
        if maximum <= 0:
            return None
        nodes[j + 1] = nodes[j] + rng.uniform(min(0.05, maximum), maximum)
    return (times, nodes)

def select_groups(cfg, entry):
    b = cfg['branches']
    if cfg.get('design_revision', 1) == 1:
        return [dict(origin=x, horizon=min(b['horizon_steps'], 180 - x), pattern='offset', duration=1, offsets=list(b['offsets_c']), terminal=False) for x in b['origins_min']]
    rng = np.random.default_rng([*entry['seed'], 917])
    origins = []
    for lo, hi in zip(b['strata_min'][:-1], b['strata_min'][1:]):
        pool = [x for x in b['origins_min'] if lo <= x < hi]
        origins.extend((int(x) for x in rng.choice(pool, b['origins_per_stratum'], replace=False)))
    origins.sort()
    eligible = [j for j, origin in enumerate(origins) if origin + b['long_horizon_steps'] <= 180]
    long_indices = set((int(x) for x in rng.choice(eligible, b['long_groups_per_parent'], replace=False)))
    scales = sorted((float(x) for x in b['offsets_c'] if x > 0))
    index = int(entry['id'].rsplit('_', 1)[1])
    sequence = index + int(entry['seed'][2]) + int(entry['seed'][3])
    groups = []
    for j, origin in enumerate(origins):
        amplitude = scales[(sequence + j) % len(scales)]
        pattern = b['patterns'][(sequence + j) % len(b['patterns'])]
        duration = int(rng.choice(b['durations_steps']))
        horizon = b['long_horizon_steps'] if j in long_indices else b['horizon_steps']
        groups.append(dict(origin=origin, horizon=min(horizon, 180 - origin), pattern=pattern, duration=min(duration, horizon, 180 - origin), offsets=[0.0, -amplitude, amplitude], terminal=False))
    if b['terminal_every'] and index % b['terminal_every'] == 0:
        origin = int(rng.choice(b['terminal_origins_min']))
        amplitude = scales[sequence % len(scales)]
        groups.append(dict(origin=origin, horizon=180 - origin, pattern=b['patterns'][sequence % len(b['patterns'])], duration=min(int(rng.choice(b['durations_steps'])), 180 - origin), offsets=[0.0, -amplitude, amplitude], terminal=True))
    return groups

def branch_plan(parent_t, spec, offset):
    origin, horizon = (spec['origin'], spec['horizon'])
    original = np.asarray(parent_t[origin:origin + horizon + 1], dtype=float)
    if len(original) != horizon + 1:
        raise ValueError('The parent temperature path is shorter than the branch horizon.')
    plan = original.copy()
    if offset == 0:
        return plan
    duration = min(int(spec['duration']), horizon)
    if spec['pattern'] == 'offset':
        plan[1:] += offset
    elif spec['pattern'] == 'rate_block':
        plan[1:] += offset * np.minimum(np.arange(1, horizon + 1), duration)
    elif spec['pattern'] == 'change_then_hold':
        plan[1:duration + 1] = original[1] + offset
        plan[duration + 1:] = plan[duration] + original[duration + 1:] - original[duration]
    else:
        raise ValueError('Unsupported temperature action.')
    return plan

def planned_steps(cfg, entry):
    """Upper bound on simulated minute-steps before infeasible branches are skipped; not wall-clock time."""
    return 180 + sum((g['horizon'] * len(g['offsets']) for g in select_groups(cfg, entry)))
