"""Coverage statistics, response metrics and numerical-resolution checks."""
from __future__ import annotations
import csv
from dataclasses import replace
from pathlib import Path
import time
import numpy as np

def _histogram(values, edges):
    values, edges = (np.asarray(values), np.asarray(edges))
    indices = np.searchsorted(edges, values, side='right')
    indices[values == edges[-1]] = len(edges) - 1
    return np.bincount(indices.ravel(), minlength=len(edges) + 1).tolist()

def _indices(values, edges):
    values, edges = (np.asarray(values), np.asarray(edges))
    indices = np.searchsorted(edges, values, side='right')
    indices[values == edges[-1]] = len(edges) - 1
    return indices

def _cross(saturation, rates, settings):
    s_edges, r_edges = (settings['saturation_edges'], settings['rate_edges'])
    s = _indices(saturation, s_edges)
    r = _indices(rates, r_edges)
    result = np.zeros((len(s_edges) + 1, len(r_edges) + 1), dtype=np.int64)
    np.add.at(result, (s, r), 1)
    return result.tolist()

def _stats(values):
    a = np.asarray(values, dtype=float)
    return dict(count=int(a.size), minimum=float(a.min()), maximum=float(a.max()), sum=float(a.sum()))

def coverage_stats(a, particle_mass, pivots, cfg):
    settings = cfg['quality']['coverage']
    n, t, s = (a['counts'], a['temperature'], a['saturation'])
    number = n.sum(-1)
    per_bin_mass = n * particle_mass
    mass = per_bin_mass.sum(-1)
    small = pivots < settings['small_size_um'] * 1e-06
    large = pivots >= settings['large_size_um'] * 1e-06
    rates = np.diff(t)
    features = dict(temperature=t, concentration=a['concentration'], saturation=s, saturation_excess=s - 1, number=number, crystal_mass=mass, rate=rates, small_number_fraction=n[:, small].sum(-1) / np.maximum(number, 1e-30), large_mass_fraction=per_bin_mass[:, large].sum(-1) / np.maximum(mass, 1e-30), first_bin_number_fraction=n[:, 0] / np.maximum(number, 1e-30), last_bin_mass_fraction=per_bin_mass[:, -1] / np.maximum(mass, 1e-30))
    return dict(features={key: _stats(value) for key, value in features.items()}, histograms={key: _histogram(value, settings[key + '_edges']) for key, value in (('temperature', t), ('saturation', s), ('rate', rates))}, saturation_rate_counts=_cross(s[:-1], rates, settings))

def branch_stats(branches, offsets, particle_mass, cfg):
    baseline = branches[offsets.index(0.0)]
    responses = []
    for a, offset in zip(branches, offsets):
        if offset == 0:
            continue
        row = dict(offset_c=float(offset), next_rate_c_min=float(a['temperature'][1] - a['temperature'][0]))
        for name, index in (('first', 1), ('end', -1)):
            error = a['counts'][index] - baseline['counts'][index]
            row[name + '_number_response_relative_l1'] = float(np.abs(error).sum() / max(baseline['counts'][index].sum(), 1e-30))
            row[name + '_mass_response_relative_l1'] = float((np.abs(error) * particle_mass).sum() / max((baseline['counts'][index] * particle_mass).sum(), 1e-30))
            row[name + '_concentration_delta'] = float(a['concentration'][index] - baseline['concentration'][index])
        row['end_temperature_c'] = float(a['temperature'][-1])
        responses.append(row)
    rates = np.array([a['temperature'][1] - a['temperature'][0] for a in branches])
    saturation = np.full(len(branches), float(baseline['saturation'][0]))
    return dict(response_strength=responses, query_saturation=float(saturation[0]), query_temperature_c=float(baseline['temperature'][0]), action_coverage=_cross(saturation, rates, cfg['quality']['coverage']))

def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or (list(rows[0]) if rows else ['status'])
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def _interval_names(edges):
    return [f'<{edges[0]}'] + [f"[{a},{b}{(']' if i == len(edges) - 2 else ')')}" for i, (a, b) in enumerate(zip(edges[:-1], edges[1:]))] + [f'>{edges[-1]}']

def write_coverage(out, manifest):
    if __package__:
        from .dataset import ROLES, write_json
    else:
        from dataset import ROLES, write_json
    cfg = manifest['resolved_config']
    if cfg.get('design_revision', 1) != 2:
        return None
    settings = cfg['quality']['coverage']
    feature_rows, histogram_rows, cross_rows, response_rows, plan_rows = ([], [], [], [], [])
    for role in ROLES:
        parents = [p for p in manifest['completed'] if p['role'] == role]
        if not parents:
            continue
        for feature in parents[0]['state_coverage']['features']:
            stats = [p['state_coverage']['features'][feature] for p in parents]
            count = sum((v['count'] for v in stats))
            feature_rows.append(dict(role=role, feature=feature, observations=count, minimum=min((v['minimum'] for v in stats)), maximum=max((v['maximum'] for v in stats)), mean=sum((v['sum'] for v in stats)) / count))
        for feature in ('temperature', 'saturation', 'rate'):
            values = np.sum([p['state_coverage']['histograms'][feature] for p in parents], axis=0)
            histogram_rows.extend((dict(role=role, feature=feature, interval=label, observations=int(count)) for label, count in zip(_interval_names(settings[feature + '_edges']), values)))
        for scope in ('mother_intervals', 'branch_first_actions'):
            shape = (len(settings['saturation_edges']) + 1, len(settings['rate_edges']) + 1)
            counts = np.zeros(shape, dtype=np.int64)
            for p in parents:
                if scope == 'mother_intervals':
                    counts += np.asarray(p['state_coverage']['saturation_rate_counts'])
                else:
                    for g in p['groups']:
                        counts += np.asarray(g['action_coverage'])
            cross_rows.extend((dict(role=role, scope=scope, saturation_interval=slabel, rate_interval=rlabel, observations=int(counts[i, j])) for i, slabel in enumerate(_interval_names(settings['saturation_edges'])) for j, rlabel in enumerate(_interval_names(settings['rate_edges']))))
        for p in parents:
            for g in p['groups']:
                context = dict(parent_id=p['id'], role=role, family=p['family'], file=g['file'], origin_min=g['origin_min'], horizon_steps=g['horizon_steps'], pattern=g['pattern'], duration_steps=g['duration_steps'], terminal=g['terminal'])
                plan_rows.append(dict(**context, branches=len(g['offsets_c']), query_temperature_c=g['query_temperature_c'], query_saturation=g['query_saturation']))
                response_rows.extend((dict(**context, **row) for row in g['response_strength']))
    write_csv(out / 'coverage_features.csv', feature_rows)
    write_csv(out / 'coverage_histograms.csv', histogram_rows)
    write_csv(out / 'saturation_action_coverage.csv', cross_rows)
    write_csv(out / 'branch_plans.csv', plan_rows)
    write_csv(out / 'response_strength.csv', response_rows)
    summary = dict(role_features=feature_rows, histogram_edges=settings, plan_count=len(plan_rows), response_pair_count=len(response_rows), note='Parent states and first branch actions are counted separately. Correlated times and shared baselines are not independent samples. Empty cells identify coverage gaps but do not imply that every combination is required. Boundary occupancy is not a boundary flux.')
    write_json(out / 'coverage_summary.json', summary)
    return summary

def aggregate_fine_grid(counts, particle_mass, fine_edges, coarse_edges):
    """Integrate absolute counts and native crystal mass over common size edges; do not recompute fine-grid mass on coarse pivots."""
    counts = np.asarray(counts)
    fine_edges, coarse_edges = (np.asarray(fine_edges), np.asarray(coarse_edges))
    if counts.shape[-1] != len(fine_edges) - 1 or np.asarray(particle_mass).shape != (counts.shape[-1],):
        raise ValueError('Fine-grid count or particle-mass dimensions are inconsistent.')
    if np.any(np.diff(fine_edges) <= 0) or np.any(np.diff(coarse_edges) <= 0):
        raise ValueError('Grid edges must be strictly increasing.')
    indices = []
    for edge in coarse_edges:
        matching = np.flatnonzero(np.isclose(fine_edges, edge, rtol=1e-12, atol=1e-18))
        if len(matching) != 1:
            raise ValueError('Grid comparisons require nested common edges; interpolation cannot manufacture bin correspondence.')
        indices.append(int(matching[0]))
    if indices[0] != 0 or indices[-1] != counts.shape[-1] or np.any(np.diff(indices) <= 0):
        raise ValueError('Coarse/fine supports differ or aggregation indices are out of order.')
    number = np.stack([counts[..., a:b].sum(-1) for a, b in zip(indices[:-1], indices[1:])], -1)
    masses = counts * np.asarray(particle_mass)
    mass = np.stack([masses[..., a:b].sum(-1) for a, b in zip(indices[:-1], indices[1:])], -1)
    return (number, mass)

def refined_grid_config(grid, coarse_edges, factor):
    """The PBM configuration specifies end pivots rather than bin edges; recompute refined pivots to retain common edges."""
    edges = np.asarray(coarse_edges)
    bins = grid.bins * factor
    fine_log_step = np.log(edges[-1] / edges[0]) / bins
    return replace(grid, bins=bins, min_um=float(edges[0] * np.exp(fine_log_step / 2) * 1000000.0), max_um=float(edges[-1] * np.exp(-fine_log_step / 2) * 1000000.0))

def numerical_parent_ids(cfg, mode, plan):
    if cfg.get('design_revision', 1) != 2:
        return []
    n = cfg['quality']['numerical']
    if not n['enabled'] or mode not in n['modes']:
        return []
    return [p['id'] for p in plan if p['role'] == 'train' and p['seed'][-1] < n['parents_per_family']]

def _replay(api, base, cfg, entry, temperatures):
    if __package__:
        from .dataset import initial_plant, sample_arrays, audit_arrays
    else:
        from dataset import initial_plant, sample_arrays, audit_arrays
    plant, _ = initial_plant(api, base, cfg, np.random.default_rng(entry['seed']), entry.get('nominal_initial', False))
    samples = [plant.current]
    for t in temperatures[1:]:
        samples.append(plant.advance(float(t)))
    a = sample_arrays(samples)
    pm = plant.cfg.vessel.crystal_density_kg_m3 * plant.cfg.vessel.volume_shape_factor * plant.grid.volumes_m3
    audit = audit_arrays(a, pm, plant._total_solute, cfg['quality']['mass_relative_tolerance'])
    if not audit['valid']:
        raise ValueError('Numerical refinement produced invalid states, which cannot be used as valid labels.')
    return (a, pm, plant.grid.length_edges_m, audit)

def numerical_check(api, base, cfg, entry, out):
    if __package__:
        from .dataset import write_json
    else:
        from dataset import write_json
    start = time.perf_counter()
    with np.load(out / (entry['id'] + '.npz'), allow_pickle=False) as saved:
        truth = {k: saved[k].copy() for k in saved.files}
    ncfg = cfg['quality']['numerical']
    numerical = replace(base.numerical, max_internal_step_s=base.numerical.max_internal_step_s * ncfg['time_step_factor'], mechanism_coupling_substep_s=base.numerical.mechanism_coupling_substep_s * ncfg['time_step_factor'])
    cases = [('time_refined', replace(base, numerical=numerical)), ('grid_refined', replace(base, grid=refined_grid_config(base.grid, truth['size_edges_m'], ncfg['grid_factor'])))]
    baseline_mass = truth['counts'] * truth['particle_mass_kg']
    rows, comparisons = ([], [])
    for label, reference_cfg in cases:
        reference_cfg.validate()
        print(f"Numerical comparison {entry['id']}：{label}, identical actual-temperature path, 180 steps...", flush=True)
        a, pm, edges, audit = _replay(api, reference_cfg, cfg, entry, truth['temperature'])
        counts, mass = aggregate_fine_grid(a['counts'], pm, edges, truth['size_edges_m'])
        number_error = np.abs(counts - truth['counts']).sum(-1) / np.maximum(truth['counts'].sum(-1), 1e-30)
        mass_error = np.abs(mass - baseline_mass).sum(-1) / np.maximum(baseline_mass.sum(-1), 1e-30)
        concentration_error = a['concentration'] - truth['concentration']
        for minute in range(181):
            rows.append(dict(parent_id=entry['id'], comparison=label, minute=minute, number_relative_l1=float(number_error[minute]), mass_relative_l1=float(mass_error[minute]), concentration_delta=float(concentration_error[minute]), total_number_relative_delta=float((counts[minute].sum() - truth['counts'][minute].sum()) / max(truth['counts'][minute].sum(), 1e-30))))
        comparisons.append(dict(comparison=label, bins=reference_cfg.grid.bins, pivot_min_um=reference_cfg.grid.min_um, pivot_max_um=reference_cfg.grid.max_um, common_edge_min_um=float(edges[0] * 1000000.0), common_edge_max_um=float(edges[-1] * 1000000.0), max_internal_step_s=reference_cfg.numerical.max_internal_step_s, mechanism_coupling_substep_s=reference_cfg.numerical.mechanism_coupling_substep_s, number_relative_l1_max=float(number_error.max()), number_relative_l1_mean=float(number_error.mean()), mass_relative_l1_max=float(mass_error.max()), mass_relative_l1_mean=float(mass_error.mean()), concentration_abs_max=float(np.abs(concentration_error).max()), audit=audit))
    stem = entry['id'] + '_numerical'
    write_csv(out / (stem + '.csv'), rows)
    result = dict(parent_id=entry['id'], family=entry['family'], status='numerical_differences_reported_convergence_not_determined', comparisons=comparisons, elapsed_seconds=time.perf_counter() - start, note='Refine time and space separately without changing kinetics. Fine grids retain common support edges with recomputed end pivots; counts and native mass are integrated separately. Grid differences include seed discretization and end-pivot changes. These two comparisons do not establish continuous-PBE convergence; no unconfirmed automatic accuracy gate is applied.')
    write_json(out / (stem + '.json'), result)
    return (result, [stem + '.json', stem + '.csv'])
