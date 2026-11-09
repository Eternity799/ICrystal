"""Plot saved numerical results; use the first ID per family rather than selecting favorable cases."""
import csv
import json
from pathlib import Path

import numpy as np


def setup(show=False):
    import matplotlib
    matplotlib.use('TkAgg' if show else 'Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
        'axes.unicode_minus':False, 'pdf.fonttype':42, 'svg.fonttype':'none', 'font.size':9,
        'axes.spines.top':False, 'axes.spines.right':False, 'legend.frameon':False})
    return plt


def save(plt, fig, directory, name, show=False):
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ('pdf', 'svg', 'png', 'tiff'):
        fig.savefig(directory/f'{name}.{extension}', dpi=300, bbox_inches='tight')
    if not show:
        plt.close(fig)


def training_plots(out, history):
    plt = setup()
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout='constrained')
    for ax,key,label in zip(axes.flat, ('validation_number', 'validation_mass', 'validation_response', 'validation_score'),
                           ('Full number-CSD relative L1', 'Full mass-CSD relative L1', 'Full-CSD temperature-response component', 'Fixed validation selection score')):
        ax.plot([r['epoch'] for r in history], [np.nan if r.get(key) is None else r[key] for r in history], color='#3676B5')
        for i,r in enumerate(history):
            if i and r['stage'] != history[i-1]['stage']:
                ax.axvline(r['epoch'], color='.6', lw=.7, ls='--')
                ax.text(r['epoch'], .98, f'H={r["horizon"]}', transform=ax.get_xaxis_transform(), va='top')
        ax.set(xlabel='Total epoch (0 = random initialization)', ylabel=label); ax.grid(alpha=.2)
    fig.suptitle('Fixed validation set and score across training horizons')
    save(plt, fig, Path(out)/'figures', 'training')
    (Path(out)/'figures'/'figure_notes.txt').write_text('Source data: ../history.csv; only the fixed validation set is plotted.\n'
        'Dashed lines mark training-horizon changes, not validation horizons. Curves describe short windows and paired responses, not full-length rollouts.\n'
        'Invalid scores are omitted; smoke runs support interface checks only.\n', encoding='utf-8')


def evaluation_plots(out, records, responses, rows, show=False):
    out = Path(out); plt = setup(show); figures = out/'figures'
    selected = []
    for family in sorted({r['family'] for r in rows}):
        selected.append(sorted([r for r in rows if r['family'] == family], key=lambda r:r['id'])[0])
    for row in selected:
        parent = row['id']
        fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout='constrained')
        for ax,key,label in zip(axes.flat, ('number_l1','mass_l1','d50_error_um','span_error'),
                               ('Full number-CSD relative L1','Full mass-CSD relative L1','D50 error (um)','Span error')):
            for kind,origin,color,legend in [('one_step',-1,'#3676B5','True-history one-step'),
                                           ('recursive',0,'#D16A43','Recursive rollout from minute 0'),
                                           ('persistence_recursive',0,'.6','Initial-state persistence')]:
                rs = [r for r in records if r['parent_id'] == parent and r['kind'] == kind and r['origin_min'] == origin]
                ax.plot([r['endpoint_min'] for r in rs], [r.get(key, np.nan) if r['valid'] else np.nan for r in rs],
                        color=color, ls='--' if 'persistence' in kind else '-', label=legend)
            ax.set(xlabel='Prediction endpoint (min)', ylabel=label); ax.grid(alpha=.2)
            if key.endswith('l1'):
                ax.set_yscale('symlog', linthresh=.001)
            else:
                ax.axhline(0, color='.5', lw=.6)
        axes.flat[0].legend(); fig.suptitle(parent+': known actual temperature')
        save(plt, fig, figures, parent+'_errors', show)
        origins = sorted({r['origin_min'] for r in records if r['parent_id'] == parent and r['kind'] == 'recursive'})
        if origins:
            origin = 10 if 10 in origins else origins[0]
            with np.load(out/f'{parent}_recursive_{origin}.npz', allow_pickle=False) as z:
                prediction, truth = z['counts'].copy(), z['truth_counts'].copy()
                endpoint = (z['time_s']/60).astype(int)
            minutes = [m for m in [10, 30, 60, 120, 170, 180] if m >= origin]
            for mass_plot in (False, True):
                fig, axes = plt.subplots(2, 3, figsize=(12, 6), layout='constrained')
                weight = row['particle_mass_kg'] if mass_plot else np.ones(200)
                for ax,minute in zip(axes.flat, minutes):
                    ix = int(np.flatnonzero(endpoint == minute)[0]); size = row['size_pivots_m']*1e6
                    ax.plot(size, truth[ix]*weight, color='.25', label='Saved PBM reference')
                    ax.plot(size, prediction[ix]*weight, color='#D16A43', ls='--', label='Surrogate free rollout')
                    ax.set(xscale='log', xlabel='Particle size (um)', title=f"Minute {minute}")
                    if mass_plot:
                        ax.set(ylabel='Crystal mass per bin (kg/kg water)')
                    else:
                        ax.set(ylabel='Particles per bin (1/kg water)')
                        ax.set_yscale('symlog', linthresh=1.)
                    ax.grid(alpha=.2)
                for ax in list(axes.flat)[len(minutes):]:
                    ax.set_visible(False)
                axes.flat[0].legend(); fig.suptitle(parent+f": absolute full CSD from minute {origin}")
                save(plt, fig, figures, parent+('_mass_csd' if mass_plot else '_number_csd'), show)
        origins = sorted({r['origin_min'] for r in responses if r['parent_id'] == parent and r['variable'] == 'd50_um' and r['horizon'] == 1})
        if origins:
            fig, axes = plt.subplots(2, len(origins), figsize=(max(6, 3*len(origins)), 6), layout='constrained', squeeze=False)
            for col,origin in enumerate(origins):
                for ax,variable,label in zip(axes[:, col], ('d50_um','span'), ('D50 response (um)','Span response')):
                    rs = [r for r in responses if r['parent_id'] == parent and r['origin_min'] == origin and
                          r['horizon'] == 1 and r['variable'] == variable]
                    xs = sorted([0.]+[r['offset_c'] for r in rs])
                    by_offset = {r['offset_c']:r for r in rs}
                    for key,color,legend in [('truth_delta','.25','PBM'), ('predicted_delta','#D16A43','Surrogate')]:
                        ys = [0. if x == 0 else (by_offset[x][key] if by_offset[x]['valid'] else np.nan) for x in xs]
                        ax.plot(xs, ys, marker='o', ms=3, color=color, label=legend)
                    ax.axhline(0, color='.7', lw=.6); ax.grid(alpha=.2)
                    ax.set(xlabel='Temperature branch offset (degC)', ylabel=label, title=f"Minute {origin}, H=1")
            axes.flat[0].legend(); fig.suptitle(parent+': temperature response from the same history')
            save(plt, fig, figures, parent+'_response', show)
    figures.mkdir(exist_ok=True)
    (figures/'figure_notes.txt').write_text('Per-minute source: ../per_minute.csv; absolute counts/concentration: ../*_recursive_*.npz and *_one_step.npz.\n'
        'Response source: ../response.csv and *_response_*.npz. Zero-offset response is zero relative to baseline, not an additional independent sample.\n'
        'The first ID in each family is plotted. All trajectories enter CSV/NPZ and summary outputs, without accuracy-based selection.\n'
        'Number and mass CSD are absolute per-bin quantities, without shape normalization or division by bin width.\n'
        'PBM references are saved labels. Predictions know actual temperatures; these figures do not test execution error, PAT error, parameter drift or closed-loop performance.\n'
        'Invalid predictions remain as gaps. Multiple minutes within a trajectory are not independent experimental repeats.\n', encoding='utf-8')
    if show:
        plt.show()


def _read_result_csv(path):
    """Restore CSV types so the string False cannot be mistaken for a valid prediction."""
    strings = {'parent_id', 'family', 'role', 'kind', 'variable', 'invalid_reason'}
    integers = {'origin_min', 'endpoint_min', 'horizon'}
    booleans = {'valid', 'direction_eligible', 'direction_agreement'}
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            if key in strings:
                continue
            if value == '':
                row[key] = None
            elif key in booleans:
                if value not in ('True', 'False'):
                    raise ValueError(f"Invalid saved Boolean: {key}={value}")
                row[key] = value == 'True'
            else:
                row[key] = int(value) if key in integers else float(value)
    return rows


def plot_saved_evaluation(directory, show=False, dataset=None):
    """Replot saved evaluations without creating a predictor or changing CSV/NPZ, scores or checkpoints."""
    from .data import load_dataset
    from .io import sha256

    out = Path(directory).resolve()
    if not (out/'summary.json').is_file() or not (out/'per_minute.csv').is_file():
        raise ValueError('Replotting requires existing summary.json and per_minute.csv; specify the evaluation directory.')
    run = out.parents[2]
    summary = json.loads((out/'summary.json').read_text(encoding='utf-8'))
    role = summary['role']
    if role not in ('validation', 'test') or out.parent.name != role or out.parent.parent.name != 'evaluation':
        raise ValueError('The replot directory must be a timestamped evaluation/validation or evaluation/test directory under a training run.')
    config = json.loads((run/'config_resolved.json').read_text(encoding='utf-8'))
    root = Path(dataset).resolve() if dataset else Path(config['dataset'])
    if sha256(root/'manifest.json') != summary['dataset_manifest_sha256']:
        raise ValueError('The replot dataset manifest differs from the saved evaluation.')
    checkpoint = summary['checkpoint']
    if checkpoint not in ('best', 'last') or sha256(run/f'{checkpoint}.pt') != summary['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from the saved evaluation; old predictions cannot be relabeled as new weights.')
    rows = load_dataset(root, (role,))[role]
    records = _read_result_csv(out/'per_minute.csv')
    responses = _read_result_csv(out/'response.csv') if (out/'response.csv').is_file() else []
    expected = {row['id']:row['family'] for row in rows}
    if len(rows) != summary['parents'] or {r['parent_id'] for r in records} != set(expected):
        raise ValueError('Saved base-trajectory IDs differ from the frozen dataset.')
    if any(r['family'] != expected[r['parent_id']] for r in records):
        raise ValueError('Saved trajectory families differ from the frozen dataset.')
    print(f"Reading {len(rows)} saved evaluation trajectories; replotting only, without new predictions...", flush=True)
    evaluation_plots(out, records, responses, rows, show)
    print(f'Replot complete: {out/"figures"}', flush=True)
    return out
