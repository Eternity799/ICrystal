"""Frozen full-CSD, recursive and temperature-response checks using saved PBM labels only."""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .data import read_manifest, load_dataset
from .io import sha256, write_json, write_csv, save_npz
from .inference import Predictor


def cqa(n, pm, edges):
    w = np.asarray(n)*pm
    if not np.isfinite(w).all() or w.sum() <= 0:
        return np.full(4, np.nan)
    cdf = np.cumsum(w)/w.sum()
    sizes = []
    for p in (.1, .5, .9):
        i = min(int(np.searchsorted(cdf, p)), len(w)-1)
        lower = 0. if i == 0 else cdf[i-1]
        width = cdf[i]-lower
        if width <= 0:
            return np.full(4, np.nan)
        fraction = (p-lower)/width
        sizes.append(float(np.exp(np.log(edges[i])+fraction*np.log(edges[i+1]/edges[i]))*1e6))
    return np.array([*sizes, (sizes[2]-sizes[0])/sizes[1]])


def metrics(out, truth_n, truth_c, row):
    if 'invalid_reason' in out:
        return dict(valid=False, invalid_reason=out['invalid_reason'])
    n = out['counts']; pm = row['particle_mass_kg']; total = float(row['initial_total_solute'])
    if not np.isfinite(n).all() or not np.isfinite(out['concentration']):
        return dict(valid=False, invalid_reason='nonfinite prediction')
    delta_cqa = cqa(n, pm, row['size_edges_m'])-cqa(truth_n, pm, row['size_edges_m'])
    if not np.isfinite(delta_cqa).all():
        return dict(valid=False, invalid_reason='zero crystal mass or invalid quantiles')
    e = n-truth_n; ntotal = truth_n.sum(); mass = truth_n@pm
    return dict(valid=True, number_l1=float(np.abs(e).sum()/ntotal), mass_l1=float(np.abs(e)@pm/mass),
                total_number_relative_error=float(e.sum()/ntotal), crystal_mass_relative_error=float(e@pm/mass),
                concentration_error=float(out['concentration']-truth_c), d10_error_um=float(delta_cqa[0]),
                d50_error_um=float(delta_cqa[1]), d90_error_um=float(delta_cqa[2]), span_error=float(delta_cqa[3]),
                alpha=float(out.get('alpha', 1.)), mass_closure_relative=float(abs(out['concentration']+n@pm-total)/total))


def true_history(row, origin, length):
    indices = np.maximum(np.arange(origin-length+1, origin+1), 0)
    return {key:row[source][indices].copy() for key,source in [('n', 'counts'), ('c', 'concentration'), ('t', 'temperature')]}


def summarize(records):
    groups = {}
    for r in records:
        groups.setdefault((r['kind'], r['origin_min']), []).append(r)
    result = []
    keys = ('number_l1', 'mass_l1', 'total_number_relative_error', 'crystal_mass_relative_error',
            'concentration_error', 'd50_error_um', 'span_error')
    for (kind, origin), selected in groups.items():
        parents = sorted({r['parent_id'] for r in selected})
        summary = dict(kind=kind, origin_min=origin, parents=len(parents), samples=len(selected),
                       invalid_count=sum(not r['valid'] for r in selected),
                       projection_count=sum(r.get('alpha', 1.) < 1-1e-12 for r in selected if r['valid']))
        for k in keys:
            means = [np.mean([abs(r[k]) for r in selected if r['parent_id'] == p and r['valid']])
                     for p in parents if any(r['parent_id'] == p and r['valid'] for r in selected)]
            summary[k] = float(np.mean(means)) if means else None
            summary[k+'_parent_p95'] = float(np.quantile(means, .95)) if means else None
        result.append(summary)
    return result


def summarize_response(records):
    result = []
    for variable in ('d50_um', 'span', 'number', 'mass'):
        horizons = sorted({r['horizon'] for r in records if r['variable'] == variable})
        for h in horizons:
            selected = [r for r in records if r['variable'] == variable and r['horizon'] == h]
            valid = [r for r in selected if r['valid']]
            eligible = [r for r in valid if r.get('direction_eligible')]
            parents = sorted({r['parent_id'] for r in valid})
            means = [np.mean([r['delta_error'] for r in valid if r['parent_id'] == p]) for p in parents]
            result.append(dict(variable=variable, horizon=h, probes=len(selected), invalid_count=len(selected)-len(valid),
                               delta_error_parent_mean=float(np.mean(means)) if means else None,
                               direction_eligible_count=len(eligible),
                               direction_correct=sum(r['direction_agreement'] for r in eligible),
                               direction_agreement=float(np.mean([r['direction_agreement'] for r in eligible])) if eligible else None))
    return result


def evaluate_run(run, role='validation', dataset=None, checkpoint='best', device='auto', show=False):
    run = Path(run).resolve()
    predictor = Predictor.from_run(run, checkpoint, device)
    saved = predictor.saved
    root = Path(dataset).resolve() if dataset else Path(saved['config']['dataset'])
    manifest = read_manifest(root)
    if sha256(root/'manifest.json') != saved['fingerprint']['dataset_manifest']:
        raise ValueError('The evaluation manifest must match the frozen training run; relocating identical data is permitted, replacing data is not.')
    rows = load_dataset(root, (role,))[role]
    for row in rows:
        if any(not np.array_equal(row[k], np.asarray(saved['grid'][k])) for k in saved['grid']):
            raise ValueError('Evaluation grid differs from the model grid.')
    out = run/'evaluation'/role/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f_utc')
    out.mkdir(parents=True, exist_ok=False)
    config = saved['config']['evaluation']; records = []; responses = []; start = time.perf_counter()
    for index,row in enumerate(rows, 1):
        total = float(row['initial_total_solute']); pm = row['particle_mass_kg']; edges = row['size_edges_m']
        print(f'{role} {index}/{len(rows)}, {row["id"]}: true-history one-step and free rollout...', flush=True)
        n = np.full_like(row['counts'], np.nan); c = np.full_like(row['concentration'], np.nan)
        n[0], c[0] = row['counts'][0], row['concentration'][0]
        for k in range(180):
            history = true_history(row, k, predictor.history)
            try:
                pred = predictor.step(history, float(row['temperature'][k+1]), total)
                n[k+1], c[k+1] = pred['counts'], pred['concentration']
            except (ValueError, FloatingPointError) as exc:
                pred = dict(invalid_reason=str(exc))
            baseline = dict(counts=row['counts'][k], concentration=row['concentration'][k])
            for kind,result in [('one_step', pred), ('persistence_one_step', baseline)]:
                records.append(dict(parent_id=row['id'], family=row['family'], role=role, kind=kind, origin_min=-1,
                                    endpoint_min=k+1, **metrics(result, row['counts'][k+1], row['concentration'][k+1], row)))
        save_npz(out/f'{row["id"]}_one_step.npz', counts=n, concentration=c,
                 truth_counts=row['counts'], truth_concentration=row['concentration'], temperature=row['temperature'],
                 time_s=row['time_s'], size_pivots_m=row['size_pivots_m'])
        for origin in config['origins']:
            history = true_history(row, origin, predictor.history)
            forecast = predictor.rollout(history, row['temperature'][origin+1:], total)
            for j,result in enumerate(forecast['details'], 1):
                baseline = dict(counts=row['counts'][origin], concentration=row['concentration'][origin])
                for kind,pred in [('recursive', result), ('persistence_recursive', baseline)]:
                    records.append(dict(parent_id=row['id'], family=row['family'], role=role, kind=kind, origin_min=origin,
                                        endpoint_min=origin+j, **metrics(pred, row['counts'][origin+j], row['concentration'][origin+j], row)))
            save_npz(out/f'{row["id"]}_recursive_{origin}.npz', counts=forecast['counts'], concentration=forecast['concentration'],
                     valid_mask=forecast['valid_mask'], truth_counts=row['counts'][origin:],
                     truth_concentration=row['concentration'][origin:], temperature=row['temperature'][origin:],
                     time_s=row['time_s'][origin:], size_pivots_m=row['size_pivots_m'])
        for branch in row['groups']:
            if len(branch['offsets_c']) <= 1:
                continue
            origin = branch['origin']; history = true_history(row, origin, predictor.history)
            forecasts = [predictor.rollout(history, ts[1:], total) for ts in branch['temperature']]
            for h in config['response_horizons']:
                if h > branch['horizon']:
                    continue  
                for i,offset in enumerate(branch['offsets_c'][1:], 1):
                    pn, bn = forecasts[i]['counts'][h], forecasts[0]['counts'][h]
                    tn, tb = branch['counts'][i, h], branch['counts'][0, h]
                    pred_cqa = cqa(pn, pm, edges)-cqa(bn, pm, edges)
                    truth_cqa = cqa(tn, pm, edges)-cqa(tb, pm, edges)
                    dn = pn-bn; dt = tn-tb
                    for variable, pv, tv, tolerance in [('d50_um', pred_cqa[1], truth_cqa[1], config['direction_d50_tolerance']),
                                                        ('span', pred_cqa[3], truth_cqa[3], config['direction_span_tolerance'])]:
                        valid = bool(np.isfinite(pv) and np.isfinite(tv)); eligible = valid and abs(tv) > tolerance
                        responses.append(dict(parent_id=row['id'], origin_min=origin, horizon=h, offset_c=float(offset),
                            variable=variable, valid=valid, predicted_delta=float(pv) if valid else None, truth_delta=float(tv),
                            delta_error=float(abs(pv-tv)) if valid else None, direction_eligible=eligible,
                            direction_agreement=bool(np.sign(pv) == np.sign(tv)) if eligible else None))
                    for variable, weight, floor in [('number', np.ones(200), saved['config']['response']['number_floor']),
                                                     ('mass', pm, saved['config']['response']['mass_floor'])]:
                        denominator = max(float(np.abs(dt)@weight), float(tb@weight)*floor, 1e-30)
                        valid = bool(np.isfinite(dn).all())
                        responses.append(dict(parent_id=row['id'], origin_min=origin, horizon=h, offset_c=float(offset),
                            variable=variable, valid=valid, delta_error=float(np.abs(dn-dt)@weight/denominator) if valid else None,
                            direction_eligible=False, direction_agreement=None))
            save_npz(out/f'{row["id"]}_response_{origin}.npz',
                     counts=np.stack([f['counts'] for f in forecasts]), truth_counts=branch['counts'],
                     concentration=np.stack([f['concentration'] for f in forecasts]), truth_concentration=branch['concentration'],
                     temperature=branch['temperature'], offsets_c=branch['offsets_c'], size_pivots_m=row['size_pivots_m'])
        elapsed = time.perf_counter()-start
        print(f"Checked {index}/{len(rows)}; elapsed {elapsed/60:.1f} min; estimated remaining {elapsed/index*(len(rows)-index)/60:.1f} min from completed trajectories.", flush=True)
    groups = summarize(records); response_summary = summarize_response(responses)
    write_csv(out/'per_minute.csv', records); write_csv(out/'response.csv', responses)
    write_csv(out/'summary.csv', groups); write_csv(out/'response_summary.csv', response_summary)
    summary = dict(status='frozen_csd_evaluation_complete_not_closed_loop_validated', role=role, mode=saved['config']['mode'],
        checkpoint=checkpoint, checkpoint_epoch=saved['epoch'], checkpoint_sha256=sha256(run/f'{checkpoint}.pt'),
        dataset_manifest_sha256=sha256(root/'manifest.json'), parents=len(rows), seconds=time.perf_counter()-start,
        groups=groups, response=response_summary,
        note='Known actual temperatures; no future true-state feedback. Base trajectories are the independent units, not time samples. Test states are read only by explicit selection; previously inspected smoke tests are development data. No PBM, optimization or MPC is run.')
    write_json(out/'summary.json', summary)
    lines = ['# Frozen full-CSD prediction evaluation', '', f'Role: {role}; training mode: {saved["config"]["mode"]}; checkpoint: {checkpoint}, epoch {saved["epoch"]}.',
             'One-step prediction uses true history through the preceding minute. Free rollout uses true history only at its origin and then feeds back predictions. Both use the same actual temperature path.', '',
             '|Scenario|Origin|Number relative L1|Mass relative L1|D50 MAE/um|Span MAE|Invalid steps|Projected steps|', '|---|---|---|---|---|---|---|---|']
    for s in groups:
        lines.append('|'+ '|'.join(str(s.get(k)) for k in ('kind','origin_min','number_l1','mass_l1','d50_error_um','span_error','invalid_count','projection_count'))+'|')
    lines += ['', '## Temperature response', '', '|Metric|H|Response error|Direction agreement/eligible|Invalid|', '|---|---|---|---|---|']
    for r in response_summary:
        lines.append(f'|{r["variable"]}|{r["horizon"]}|{r["delta_error_parent_mean"]}|{r["direction_correct"]}/{r["direction_eligible_count"]}|{r["invalid_count"]}|')
    lines += ['', 'Inspect full number/mass CSD errors and invalid states before response direction and amplitude. Accurate D50/Span do not establish accurate full CSD.',
              'Inventory closure follows from the output parameterization and does not independently establish dynamic accuracy. Inspect projection activations alongside the raw-inventory penalty.',
              'Metrics are averaged within each base trajectory and then across trajectories. Invalid states are excluded from accuracy means and counted separately; missing values are not zero error.',
              'Direction is scored only when the PBM D50/Span response exceeds its tolerance. Full-CSD response columns report normalized amplitude errors.',
              'The persistence baseline tests whether dynamics were learned. There is no automatic MPC qualification gate.', summary['note']]
    (out/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    from .plotting import evaluation_plots
    evaluation_plots(out, records, responses, rows, show)
    print(out, flush=True); print(summary['status'], flush=True)
    return out
