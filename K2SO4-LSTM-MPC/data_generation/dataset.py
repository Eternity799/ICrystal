"""Nominal PBM trajectories and paired temperature-action branches."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path
import numpy as np
if __package__:
    from .dataset_design import INITIAL_KEYS, branch_plan, select_groups, temperature_nodes, validate_design
else:
    from dataset_design import INITIAL_KEYS, branch_plan, select_groups, temperature_nodes, validate_design
BASE = Path(__file__).resolve().parent
ROLES = ('train', 'validation', 'test')
FAMILIES = ('piecewise', 'hold', 'local_reheat')

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write_json(path, data):
    """Use a unique adjacent temporary file and retry Windows sharing violations; retain the previous file on failure."""
    path = Path(path)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.2 * (attempt + 1))

def save_npz(path, **arrays):
    path = Path(path)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.2 * (attempt + 1))

def resolve(cfg, key):
    return (Path(cfg.get('_base', BASE)) / cfg[key]).resolve()

def validate_config(c):
    allowed = {'schema_version', 'design_revision', 'seed', 'pbm_source', 'pbm_config', 'output_root', 'batch_minutes', 'sample_seconds', 'families', 'counts', 'coverage', 'initial', 'branches', 'quality', '_base'}
    if set(c) - allowed:
        raise ValueError(f'Unsupported configuration fields: {sorted(set(c) - allowed)}; unsupported settings, including noise, are not silently ignored.')
    if type(c['seed']) is not int or c['seed'] < 0:
        raise ValueError('seed must be a nonnegative integer.')
    if c['schema_version'] != 1 or c['sample_seconds'] != 60 or c['batch_minutes'] != 180:
        raise ValueError('This interface requires schema 1, a 180-minute batch and 60-second sampling.')
    if not c['families'] or len(set(c['families'])) != len(c['families']) or (not set(c['families']) <= set(FAMILIES)):
        raise ValueError('Trajectory families are empty, duplicated or unsupported.')
    for mode in ('smoke', 'formal'):
        if set(c['counts'][mode]) != set(ROLES):
            raise ValueError('Specify counts for train, validation and test explicitly.')
        if any((type(n) is not int or n < 1 for n in c['counts'][mode].values())):
            raise ValueError('Each family/role count must be a positive integer.')
    for block, keys in [('coverage', ('temperature_c', 'rate_c_min', 'endpoint_c', 'arrival_min')), ('initial', ('temperature_c', 'saturation', 'seed_d50_um', 'seed_sigma_g', 'seed_mass_fraction'))]:
        for key in keys:
            a = np.asarray(c[block][key], dtype=float)
            if a.shape != (2,) or not np.isfinite(a).all() or a[0] > a[1]:
                raise ValueError(f'{block}.{key} must contain two finite numbers in ascending order.')
    lo, hi = c['coverage']['temperature_c']
    for key, block in [('endpoint_c', 'coverage'), ('temperature_c', 'initial')]:
        a, b = c[block][key]
        if not lo <= a <= b <= hi:
            raise ValueError(f'{block}.{key} exceeds the configured temperature coverage.')
    rlo, rhi = c['coverage']['rate_c_min']
    amin, amax = c['coverage']['arrival_min']
    if not rlo < 0 <= rhi or not 0 < amin <= amax <= 180:
        raise ValueError('Invalid cooling-rate or arrival-time range.')
    if type(c['coverage']['knots']) is not int or c['coverage']['knots'] < 2:
        raise ValueError('knots must be an integer of at least 2.')
    for k in ('saturation', 'seed_d50_um', 'seed_mass_fraction'):
        if c['initial'][k][0] <= 0:
            raise ValueError(f'initial.{k} must be positive.')
    if c['initial']['seed_sigma_g'][0] <= 1:
        raise ValueError('seed_sigma_g must be greater than 1.')
    b = c['branches']
    if type(b['horizon_steps']) is not int or b['horizon_steps'] < 1:
        raise ValueError('The branch horizon must be a positive integer.')
    if len(set(b['origins_min'])) != len(b['origins_min']) or any((type(x) is not int or not 0 <= x < 180 for x in b['origins_min'])):
        raise ValueError('Branch origins must be distinct integer minutes from 0 to 179.')
    offsets = b['offsets_c']
    if not np.isfinite(offsets).all() or len(set(offsets)) != len(offsets) or 0.0 not in offsets:
        raise ValueError('Offsets must be finite and distinct and include a zero-offset replay.')
    for key in ('mass_relative_tolerance', 'reproduction_relative_l1'):
        if not np.isfinite(c['quality'][key]) or c['quality'][key] <= 0:
            raise ValueError('Mass-balance and replay tolerances must be finite and positive.')
    if type(c['quality']['schedule_attempts']) is not int or c['quality']['schedule_attempts'] < 1:
        raise ValueError('schedule_attempts must be a positive integer.')
    validate_design(c)

def load_pbm(cfg):
    source = resolve(cfg, 'pbm_source')
    name = '_full_csd_dataset_pbm'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, source / '__init__.py', submodule_search_locations=[str(source)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    elif Path(sys.modules[name].__file__).resolve() != source / '__init__.py':
        raise ValueError('A process cannot switch between PBM source directories.')
    from importlib import import_module
    return dict(BatchPlant=import_module(name + '.simulation').BatchPlant, load_config=import_module(name + '.config').load_config, solubility=import_module(name + '.kinetics').solubility, config_path=resolve(cfg, 'pbm_config'), source=source)

def make_plan(cfg, mode):
    if mode not in cfg['counts']:
        raise ValueError(f'The configuration does not define mode {mode}.')
    rows = []
    for family in cfg['families']:
        for r, role in enumerate(ROLES):
            for i in range(cfg['counts'][mode][role]):
                rows.append(dict(id=f'{family}_{role}_{i:05d}', family=family, role=role, seed=[int(cfg['seed']), {'smoke': 0, 'formal': 1, 'pilot': 2}[mode], FAMILIES.index(family), r, i]))
    if cfg.get('design_revision', 1) == 2:
        rows.sort(key=lambda x: (x['seed'][-1], x['seed'][2], x['seed'][3]))
        every = cfg['initial']['nominal_every']
        for row in rows:
            row['nominal_initial'] = bool(every and row['seed'][-1] % every == 0)
    return rows

def legal_schedule(temperatures, cfg):
    t = np.asarray(temperatures)
    lo, hi = cfg['coverage']['temperature_c']
    rlo, rhi = cfg['coverage']['rate_c_min']
    rate = np.diff(t) / (cfg['sample_seconds'] / 60.0)
    return bool(np.isfinite(t).all() and np.all(t >= lo - 1e-09) and np.all(t <= hi + 1e-09) and np.all(rate >= rlo - 1e-09) and np.all(rate <= rhi + 1e-09))

def temperature_schedule(cfg, family, start, rng):
    """Rejection-sample piecewise nodes without clipping; vary the endpoint and arrival time."""
    c = cfg['coverage']
    k = c['knots']
    minutes = np.arange(181, dtype=float)
    for _ in range(cfg['quality']['schedule_attempts']):
        if cfg.get('design_revision', 1) == 2:
            candidate = temperature_nodes(c, family, start, rng)
            if candidate is None:
                continue
            nodes_t, nodes = candidate
            t = np.interp(minutes, nodes_t, nodes)
            if legal_schedule(t, cfg):
                return t
            continue
        endpoint = rng.uniform(*c['endpoint_c'])
        arrival = rng.uniform(*c['arrival_min'])
        nodes_t = np.linspace(0.0, arrival, k + 1)
        weights = rng.uniform(0.3, 1.7, k)
        if family == 'hold':
            weights[rng.choice(k, max(1, k // 4), replace=False)] = 0.0
        nodes = start + (endpoint - start) * np.r_[0.0, np.cumsum(weights) / weights.sum()]
        if family == 'local_reheat':
            j = int(rng.integers(1, k - 1))
            nodes[j + 1] = nodes[j] + rng.uniform(0.2, 1.5)
        t = np.interp(minutes, nodes_t, nodes)
        rates = np.diff(nodes) / np.diff(nodes_t)
        if legal_schedule(t, cfg) and np.all(rates >= c['rate_c_min'][0]) and np.all(rates <= c['rate_c_min'][1]):
            return t
    raise ValueError(f'{family} trajectory rejection sampling exhausted; check endpoint, arrival-time and rate ranges.')

def branch_schedule(parent_t, origin, horizon, offset, cfg):
    branch = np.asarray(parent_t[origin:origin + horizon + 1], dtype=float).copy()
    branch[1:] += offset
    return branch if legal_schedule(branch, cfg) else None

def restore_plant(plant, sample):
    """Restore only dynamic states, preserving the parent batch inventory, total solute and initial seed mass."""
    plant._counts = sample.counts_per_kg_water.copy()
    plant._current_time_s = sample.time_s
    plant._temperature_c = sample.temperature_c
    plant._samples = [plant._make_sample()]

def sample_arrays(samples):
    return dict(time_s=np.array([s.time_s for s in samples]), counts=np.stack([s.counts_per_kg_water for s in samples]), concentration=np.array([s.concentration_kg_kg_water for s in samples]), temperature=np.array([s.temperature_c for s in samples]), saturation=np.array([s.supersaturation_ratio for s in samples]))

def audit_arrays(a, particle_mass, total, tolerance):
    finite = all((np.isfinite(a[k]).all() for k in ('counts', 'concentration', 'temperature', 'saturation')))
    if not finite or not np.isfinite(total) or total <= 0 or (not np.isfinite(particle_mass).all()):
        return dict(valid=False, reason='Nonfinite labels are present.')
    err = float(np.max(np.abs(a['concentration'] + a['counts'] @ particle_mass - total)) / total)
    return dict(valid=bool(a['counts'].min() >= 0 and a['concentration'].min() >= 0 and (err <= tolerance)), max_relative_mass_error=err, minimum_saturation=float(a['saturation'].min()), maximum_saturation=float(a['saturation'].max()), minimum_concentration=float(a['concentration'].min()), maximum_concentration=float(a['concentration'].max()), minimum_temperature_c=float(a['temperature'].min()), maximum_temperature_c=float(a['temperature'].max()), minimum_total_number=float(a['counts'].sum(axis=-1).min()), maximum_total_number=float(a['counts'].sum(axis=-1).max()))

def initial_plant(api, base, cfg, rng, nominal=False):
    """Construct the seed shape, then rebalance the 0.8-L inventory at the sampled initial saturation ratio."""
    c = cfg['initial']
    sampled = {key: float(rng.uniform(*c[key])) for key in INITIAL_KEYS}
    if nominal:
        sampled.update(temperature_c=base.batch.start_temperature_c, saturation=1.0, seed_d50_um=base.seed.d50_um, seed_sigma_g=base.seed.sigma_g, seed_mass_fraction=base.seed.mass_fraction_of_dissolved)
    model_cfg = replace(base, seed=replace(base.seed, d50_um=sampled['seed_d50_um'], sigma_g=sampled['seed_sigma_g'], mass_fraction_of_dissolved=sampled['seed_mass_fraction']), batch=replace(base.batch, start_temperature_c=sampled['temperature_c'], total_minutes=cfg['batch_minutes'], sample_seconds=cfg['sample_seconds']))
    model_cfg.validate()
    plant = api['BatchPlant'].initialize(model_cfg)
    concentration = float(api['solubility'](sampled['temperature_c'])) * sampled['saturation']
    fraction = concentration / (1.0 + concentration)
    v = model_cfg.vessel
    seed_ratio = sampled['seed_mass_fraction']
    solution_volume = v.slurry_volume_l * 0.001 / (1.0 + seed_ratio * fraction * v.solution_density_kg_m3 / v.crystal_density_kg_m3)
    solution_mass = solution_volume * v.solution_density_kg_m3
    water = solution_mass / (1.0 + concentration)
    dissolved = water * concentration
    seed_mass = seed_ratio * dissolved
    old = plant.inventory
    counts = old.seed_counts_per_kg_water * (seed_mass / water) / (old.seed_mass_kg / old.water_mass_kg)
    inv = replace(old, solution_volume_l=solution_volume * 1000.0, solution_mass_kg=solution_mass, water_mass_kg=water, dissolved_solute_mass_kg=dissolved, seed_mass_kg=seed_mass, initial_concentration_kg_kg_water=concentration, seed_counts_per_kg_water=counts)
    return (api['BatchPlant'](model_cfg, plant.grid, inv, plant.operators), sampled)

def generate_parent(api, base, cfg, entry, out):
    rng = np.random.default_rng(entry['seed'])
    plant, initial = initial_plant(api, base, cfg, rng, entry.get('nominal_initial', False))
    initial_sample = plant.current
    require_crystals = cfg['quality'].get('require_positive_crystals', False)
    pbm_steps = 0
    for trajectory_attempt in range(cfg['quality'].get('trajectory_attempts', 1)):
        restore_plant(plant, initial_sample)
        temperatures = temperature_schedule(cfg, entry['family'], plant.current.temperature_c, rng)
        states = [plant.current]
        extinct = False
        for t in temperatures[1:]:
            state = plant.advance(t)
            pbm_steps += 1
            states.append(state)
            if require_crystals and (state.counts_per_kg_water.sum() <= 0 or state.crystal_mass_kg_kg_water <= 0):
                extinct = True
                break
        if not extinct:
            break
        print(f"{entry['id']} temperature plan caused complete dissolution outside the positive-crystal training domain; resampling the plan...", flush=True)
    else:
        raise ValueError('Repeated plans caused complete dissolution; the current training domain requires positive crystal number/mass. Adjust reheating or seed coverage.')
    a = sample_arrays(states)
    mass = plant.cfg.vessel.crystal_density_kg_m3 * plant.cfg.vessel.volume_shape_factor * plant.grid.volumes_m3
    total = plant._total_solute
    qa = audit_arrays(a, mass, total, cfg['quality']['mass_relative_tolerance'])
    if not qa['valid']:
        raise ValueError(f"Parent-trajectory label audit failed: {entry['id']}，{qa}")
    common = dict(schema_version=np.array(1), parent_id=np.array(entry['id']), role=np.array(entry['role']), initial_total_solute=np.array(total), size_pivots_m=plant.grid.pivots_m, size_edges_m=plant.grid.length_edges_m, particle_mass_kg=mass, interval_seconds=np.array(cfg['sample_seconds']))
    files = []

    def save(name, arrays, **extra):
        path = out / name
        save_npz(path, **arrays, **common, **extra)
        files.append(dict(file=name, sha256=sha256(path)))
    save(entry['id'] + '.npz', a, valid_mask=np.ones(len(states), dtype=bool))
    groups, skipped = ([], [])
    specs = select_groups(cfg, entry)
    for group_index, spec in enumerate(specs):
        origin, horizon = (spec['origin'], spec['horizon'])
        branches, offsets, audits = ([], [], [])
        reproduction = None
        for offset in spec['offsets']:
            schedule = branch_plan(temperatures, spec, offset)
            if not legal_schedule(schedule, cfg):
                skipped.append(dict(origin_min=origin, offset_c=offset, pattern=spec['pattern'], duration_steps=spec['duration'], horizon_steps=horizon, reason='Temperature or rate is infeasible; no clipping was applied.'))
                continue
            restore_plant(plant, states[origin])
            samples = [plant.current]
            extinct = False
            for t in schedule[1:]:
                state = plant.advance(t)
                pbm_steps += 1
                samples.append(state)
                if require_crystals and (state.counts_per_kg_water.sum() <= 0 or state.crystal_mass_kg_kg_water <= 0):
                    extinct = True
                    break
            if extinct:
                if offset == 0:
                    raise ValueError('The zero-action baseline reached a different complete dissolution state and did not reproduce the parent.')
                skipped.append(dict(origin_min=origin, offset_c=offset, pattern=spec['pattern'], duration_steps=spec['duration'], horizon_steps=horizon, reason='complete dissolution outside the positive-crystal training domain; not imputed or included as a label'))
                continue
            b = sample_arrays(samples)
            audit = audit_arrays(b, mass, total, cfg['quality']['mass_relative_tolerance'])
            if not audit['valid']:
                raise ValueError(f"Branch-label audit failed: {entry['id']}，{origin}，{offset}，{audit}")
            if offset == 0:
                truth = a['counts'][origin:origin + horizon + 1]
                reproduction = float(np.max(np.sum(np.abs(b['counts'] - truth), axis=-1) / np.maximum(np.sum(np.abs(truth), axis=-1), 1e-30)))
                if reproduction > cfg['quality']['reproduction_relative_l1']:
                    raise ValueError(f'Zero-offset branch did not reproduce the parent: {reproduction}')
            branches.append(b)
            offsets.append(offset)
            audits.append(audit)
        if reproduction is None:
            raise ValueError('The zero-offset branch is missing.')
        suffix = f'_plan_{group_index:02d}' if cfg.get('design_revision', 1) == 2 else ''
        name = f"{entry['id']}_origin_{origin:03d}{suffix}.npz"
        save(name, {key: np.stack([b[key] for b in branches]) for key in branches[0]}, offsets_c=np.array(offsets), origin_min=np.array(origin), design_revision=np.array(cfg.get('design_revision', 1)), plan_pattern=np.array(spec['pattern']), duration_steps=np.array(spec['duration']), terminal_plan=np.array(spec['terminal']), valid_mask=np.ones((len(branches), horizon + 1), dtype=bool))
        groups.append(dict(file=name, origin_min=origin, horizon_steps=horizon, offsets_c=offsets, pattern=spec['pattern'], duration_steps=spec['duration'], terminal=spec['terminal'], reproduction_relative_l1=reproduction, audits=audits))
        if cfg.get('design_revision', 1) == 2:
            if __package__:
                from .dataset_quality import branch_stats
            else:
                from dataset_quality import branch_stats
            groups[-1].update(branch_stats(branches, offsets, mass, cfg))
    result = dict(**entry, initial=initial, audit=qa, files=files, groups=groups, skipped_branches=skipped, trajectory_attempts=trajectory_attempt + 1, pbm_steps_executed=pbm_steps, final_temperature_c=float(temperatures[-1]), initial_total_solute=total)
    if cfg.get('design_revision', 1) == 2:
        if __package__:
            from .dataset_quality import coverage_stats
        else:
            from dataset_quality import coverage_stats
        result['state_coverage'] = coverage_stats(a, mass, plant.grid.pivots_m, cfg)
    return result
