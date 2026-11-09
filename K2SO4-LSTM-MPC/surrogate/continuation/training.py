"""Continuation training with inherited data roles, losses and preprocessing."""
import copy
import hashlib
import json
import math
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from core.data import GRID_KEYS, read_manifest
from core.io import save_checkpoint, sha256, write_csv, write_json
from core.model import ARCHITECTURE, FullCSDLSTM, rollout
from core.mpc.data import MPCSampleFactory, load_selected_dataset, select_parent_ids
from core.mpc.training import (WORKFLOW, coverage, fixed_validation, loss_value,
                               paired_value, payload, run_fingerprint, source_hashes)
from core.training import choose_device, restore_rng, rng_state
from .config import BASE

CONTINUATION_VERSION = 'mpc_best_continuation_v1'
ORDINARY_MIXTURE = {'short': .70, 'long': .25, 'terminal': 0.}
TERMINAL_MIXTURE = {'short': 0., 'long': 0., 'terminal': 1.}


def extension_hashes():
    files = [BASE / 'continue_mpc.py', *sorted((BASE / 'continuation').glob('*.py'))]
    return {p.relative_to(BASE).as_posix(): sha256(p) for p in files}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode('utf-8')).hexdigest()


def validate_parent(saved, run_meta, settings, current_sources, manifest_sha):
    """Verify the parent weights, arm, data and implementation without modifying the parent run."""
    if saved.get('workflow') != WORKFLOW or saved.get('architecture') != ARCHITECTURE:
        raise ValueError('The parent workflow/architecture is not the full-CSD control-oriented training interface.')
    if (run_meta.get('workflow') != WORKFLOW or run_meta.get('arm') != 'replay_csd'
            or run_meta.get('mode') != 'formal'):
        raise ValueError('Only a formal replay_csd parent is supported.')
    if settings.get('checkpoint') != 'best':
        raise ValueError('Continuation starts from best.')
    if (not isinstance(saved.get('config'), dict) or type(saved.get('epoch')) is not int
            or saved['epoch'] < 0):
        raise ValueError('Parent checkpoint lacks configuration or epoch.')
    expected = run_fingerprint(saved['config'], 'replay_csd', manifest_sha, current_sources)
    if saved.get('fingerprint') != expected or run_meta.get('fingerprint') != expected:
        raise ValueError('Parent configuration, dataset manifest or frozen source differs; incompatible runs cannot be mixed.')
    required = {'model', 'scaler', 'grid', 'control_scales', 'quality_domain', 'objective_snapshot'}
    if not required <= set(saved):
        raise ValueError('Parent checkpoint lacks model, scaler, grid or objective snapshot.')


def draw_pairs(factory, horizon, rng, device):
    available = factory.availability
    if not available['terminal']['groups'] or not any(available[k]['groups'] for k in ('short', 'long')):
        raise ValueError('Continuation requires both ordinary response and full-terminal groups; neither is silently skipped.')
    ordinary = factory.sample_pairs(1, horizon, rng, device, mixture=ORDINARY_MIXTURE,
                                    full_terminal=False)
    terminal = factory.sample_pairs(1, horizon, rng, device, mixture=TERMINAL_MIXTURE,
                                    full_terminal=True)
    return ordinary + terminal


def train_update(model, optimizer, factory, edges, scales, config, plan, rng, device):
    """One update: ordinary window loss plus the mean of two original paired losses, without applying response weight twice."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    sample = factory.sample(plan['batch_size'], plan['horizon'], rng, device)
    prediction = rollout(model, sample)
    value, parts = loss_value(prediction, sample, model, edges, scales, config, 'replay_csd')
    if not bool(torch.isfinite(value)):
        raise FloatingPointError('Nonfinite ordinary window loss; retain the previous complete epoch and do not commit this one.')
    value.backward()
    result = {'loss': float(value.detach()), **{k: float(v.detach()) for k, v in parts.items()}}
    del prediction, value, parts
    pair_losses = []
    for pair in draw_pairs(factory, plan['horizon'], rng, device):
        pair_value, _ = paired_value(model, pair, edges, scales, config, 'replay_csd')
        if not bool(torch.isfinite(pair_value)):
            raise FloatingPointError('Nonfinite paired loss; epoch not committed.')
        (pair_value / 2.).backward()
        pair_losses.append(float(pair_value.detach()))
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['sampling']['grad_clip'])
    if not bool(torch.isfinite(norm)):
        raise FloatingPointError('Nonfinite continuation gradient; epoch not committed.')
    optimizer.step()
    result.update(pair_loss=sum(pair_losses) / 2., ordinary_pair_loss=pair_losses[0],
                  terminal_pair_loss=pair_losses[1], grad_norm=float(norm.detach()))
    return result


def selected_ids(manifest, parent_ids, settings):
    if set(parent_ids) != {'train', 'validation', 'development'}:
        raise ValueError('Parent selected_parents.json must contain train, validation and development.')
    for key, ids in parent_ids.items():
        if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)):
            raise ValueError('Empty or duplicated parent set: ' + key)
    if set(parent_ids['train']) & set(parent_ids['validation']):
        raise ValueError('Parent train and validation sets overlap.')
    if not set(parent_ids['development']) <= set(parent_ids['validation']):
        raise ValueError('The parent development set is not a subset of validation.')
    if settings['mode'] == 'formal':
        return copy.deepcopy(parent_ids)
    result = {}
    for i, role in enumerate(('train', 'validation', 'development')):
        allowed = parent_ids['train'] if role == 'train' else (
            parent_ids['development'] if role == 'validation' else result['validation'])
        allowed = set(allowed)
        subset = {**manifest, **{key: [e for e in manifest[key] if e['id'] in allowed]
                                for key in ('plan', 'completed')}}
        result[role] = select_parent_ids(subset, 'train' if role == 'train' else 'validation',
                                        settings['plan'][role + '_per_family'], settings['seed'] + i)
    return result


def _factory_state(factory):
    return dict(parents=dict(factory.seen_parents), groups=dict(factory.seen_groups), draws=factory.draws,
                mixture=dict(factory.last_mixture))


def _restore_factory(factory, state):
    factory.seen_parents.update(state['parents'])
    factory.seen_groups.update(state['groups'])
    factory.draws = state['draws']
    factory.last_mixture = dict(state['mixture'])


def _coverage(factory, updates):
    result = coverage(factory)
    result.update(completed_updates=updates, ordinary_response_draws=updates, terminal_response_draws=updates,
                  pair_policy='One ordinary plus one full-terminal group per update; average the two losses',
                  ordinary_requested_mixture=ORDINARY_MIXTURE,
                  note='effective_mixture records the latest terminal sample; the overall allocation follows the fixed two-group policy.')
    return result


def _record_validation(record, metrics):
    record.update(validation_score=metrics['score'],
                  **{'validation_' + k: v for k, v in metrics.items() if k != 'score'})


def _write_report(out, run, history):
    baseline = history[0]['validation_score']
    improvement = baseline - run['best_score']
    (out / 'report.md').write_text(
        '# Continuation of the best full-CSD model\n\n'
        f"""Status: {run["status"]}. Original parent epoch: {run["parent_epoch"]}; continuation epochs restart at zero.

"""
        f'Completed {run["completed_epochs"]} epochs and {run["completed_updates"]}/{run["planned_updates"]} updates; '
        f'best continuation epoch {run["best_epoch"]}, fixed development score {run["best_score"]:.8g}; '
        f"""parent score on the same set {baseline:.8g}, reduction {improvement:.8g} (best=0 retains the parent).

"""
        'Formal mode inherits parent train/validation/development sets. Smoke uses smaller sets and its score is not directly comparable to formal.'
        'Scaler, grid, inventory, network, CSD loss, response loss and control objective are inherited; no new direct Span loss is added.\n\n'
        'Each update adds the ordinary parent-window loss to the mean original paired losses from one ordinary and one complete terminal branch.'
        'Ordinary short:long groups use normalized 0.70:0.25 weights; terminal groups always reach minute 180.'
        'Future truth is used only in losses; rollout feeds back predicted states.\n\n'
        'A new AdamW and random sequence start from parent best weights; this does not restore the optimizer from the parent best epoch.'
        "last restores this continuation run's optimizer/RNG and last complete epoch. Interrupted partial epochs do not count as completed updates.\n\n"
        'best uses the same fixed development score, including the epoch-zero parent; full-validation scores are tracked separately.'
        'Continuation sources have their own fingerprint; frozen inference sources are unchanged. No test states, PBM or closed loop are used.'
        'Changes in long-rollout D50/Span and action direction still require frozen evaluation; continuation does not automatically qualify MPC.\n',
        encoding='utf-8')


def plot_history(out, history, show=False):
    from core.plotting import setup, save
    plt = setup(show)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout='constrained')
    keys = ('validation_score', 'validation_absolute', 'validation_response', 'validation_terminal')
    labels = ('Fixed development total score', 'Fixed development state error', 'Fixed development response error', 'Fixed development full-terminal error')
    for ax, key, label in zip(axes.flat, keys, labels):
        rows = [r for r in history if r.get(key) is not None]
        ax.plot([r['epoch'] for r in rows], [r[key] for r in rows], 'o-', ms=3, color='#3676B5')
        if history[0].get(key) is not None:
            ax.axhline(history[0][key], color='.5', ls='--', lw=1, label='Parent model (continuation epoch 0)')
        ax.set(xlabel='Continuation epoch (0 = parent)', ylabel=label)
        ax.grid(alpha=.2)
    axes.flat[0].legend()
    fig.suptitle('Bounded continuation on a fixed development set; closed loop not yet validated')
    save(plt, fig, out / 'figures', 'continuation_training', show)
    write_csv(out / 'figures' / 'continuation_source_data.csv', history)
    if show:
        plt.show()
        plt.close(fig)


def run(settings, config_path, resume=None, no_plot=False, show=False):
    parent = Path(settings['parent_run'])
    if not (parent / 'best.pt').is_file() or not (parent / 'run.json').is_file():
        raise ValueError('Parent directory must contain best.pt and run.json from a formal replay_csd arm.')
    saved = torch.load(parent / 'best.pt', map_location='cpu', weights_only=False)
    parent_meta = json.loads((parent / 'run.json').read_text(encoding='utf-8'))
    root = Path(saved['config']['dataset'])
    sources = source_hashes()
    manifest_sha = sha256(root / 'manifest.json')
    validate_parent(saved, parent_meta, settings, sources, manifest_sha)
    manifest = read_manifest(root)
    parent_selection = json.loads((parent / 'selected_parents.json').read_text(encoding='utf-8'))
    ids = selected_ids(manifest, parent_selection, settings)
    lineage = dict(version=CONTINUATION_VERSION, parent_run=str(parent.resolve()),
                   parent_checkpoint='best.pt', parent_sha256=sha256(parent / 'best.pt'),
                   parent_epoch=saved['epoch'], parent_fingerprint=saved['fingerprint'],
                   parent_selection_sha256=sha256(parent / 'selected_parents.json'),
                   parent_run_sha256=sha256(parent / 'run.json'), settings=settings,
                   source=extension_hashes(), config_sha256=sha256(config_path),
                   selection_sha256=digest(ids), optimizer_reset=True)
    config = copy.deepcopy(saved['config'])
    config['continuation'] = lineage
    fingerprint = run_fingerprint(config, 'replay_csd', manifest_sha, sources)
    checkpoint = None
    if resume is not None:
        out = Path(resume).resolve()
        if not (out / 'last.pt').is_file():
            raise ValueError('Resume requires this continuation arm directory containing last.pt.')
        checkpoint = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
        if (checkpoint.get('fingerprint') != fingerprint or checkpoint.get('workflow') != WORKFLOW
                or checkpoint.get('continuation') != lineage):
            raise ValueError('Continuation settings, parent weights, parent sets or source changed; incompatible runs cannot be resumed together.')
        if 'best_checkpoint' not in checkpoint:
            raise ValueError('Resume checkpoint lacks the best-weight snapshot; preservation of the parent/best model cannot be guaranteed.')
    else:
        out = Path(settings['output_root']) / settings['mode'] / datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f_utc') / 'replay_csd'
        out.mkdir(parents=True, exist_ok=False)
    lock = out / 'RUNNING.lock'
    try:
        with lock.open('x', encoding='utf-8') as stream:
            stream.write('Continuation running; after abnormal exit confirm the process stopped before removing this lock.\n')
    except FileExistsError as exc:
        raise ValueError('RUNNING.lock already exists; do not resume the same run concurrently.') from exc
    try:
        rows = load_selected_dataset(root, ('train', 'validation'),
                                     {k: ids[k] for k in ('train', 'validation')}, manifest)
        dev = [r for r in rows['validation'] if r['id'] in set(ids['development'])]
        if len(dev) != len(ids['development']):
            raise ValueError('The fixed development parent set did not load completely.')
        if any(not np.array_equal(row[k], np.asarray(saved['grid'][k]))
               for selected in rows.values() for row in selected for k in GRID_KEYS):
            raise ValueError('Dataset grid/physical properties differ from the parent checkpoint.')
        plan = settings['plan']
        device = choose_device(settings['device'])
        random.seed(settings['seed']); np.random.seed(settings['seed']); torch.manual_seed(settings['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(settings['seed'])
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
        model = FullCSDLSTM(saved['scaler'], saved['grid'], config['model']).to(device)
        model.load_state_dict(saved['model'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=plan['lr'],
                                      weight_decay=config['sampling']['weight_decay'])
        rng = np.random.default_rng(settings['seed'] + 1)
        factory = MPCSampleFactory(rows['train'], model.history, time_edges=config['sampling']['time_edges'])
        if not factory.availability['terminal']['groups'] or not any(factory.availability[k]['groups'] for k in ('short', 'long')):
            raise ValueError('Selected training parents lack ordinary or terminal groups; continuation will not start.')
        edges = torch.tensor(saved['grid']['size_edges_m'], dtype=torch.float64, device=device)
        history, cursor, best_score, best_epoch = [], 0, math.inf, 0
        best_checkpoint = None
        if checkpoint is not None:
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            restore_rng(checkpoint['rng'], rng)
            history, cursor = checkpoint['history'], checkpoint['cursor']
            best_score, best_epoch = checkpoint['best_score'], checkpoint['best_epoch']
            _restore_factory(factory, checkpoint['factory_state'])
            best_checkpoint = checkpoint['best_checkpoint']
            # Restore the best checkpoint as well as the latest resumption state.
            save_checkpoint(out / 'best.pt', best_checkpoint)
        def pack():
            result = payload(model, saved['scaler'], saved['grid'], config, saved['control_scales'],
                             saved['quality_domain'], saved['objective_snapshot'], fingerprint,
                             cursor, best_score if math.isfinite(best_score) else None, best_epoch)
            result['continuation'] = lineage
            return result
        def save_last():
            value = pack()
            value.update(optimizer=optimizer.state_dict(), rng=rng_state(rng), history=history,
                         cursor=cursor, completed_updates=cursor * plan['updates_per_epoch'],
                         factory_state=_factory_state(factory), best_checkpoint=best_checkpoint)
            save_checkpoint(out / 'last.pt', value)
        def validate(selected):
            return fixed_validation(model, selected, config, saved['control_scales'],
                                    saved['objective_snapshot'], saved['quality_domain'], device)
        write_json(out / 'config_resolved.json', config)
        write_json(out / 'continuation_plan.json', lineage)
        write_json(out / 'selected_parents.json', ids)
        write_json(out / 'control_scales.json', saved['control_scales'])
        write_json(out / 'objective_snapshot.json', dict(objective=saved['objective_snapshot'],
                                                        quality_domain=saved['quality_domain']))
        shutil.copy2(config_path, out / 'config_original.yaml')
        for rel in {**sources, **lineage['source']}:
            target = out / 'source' / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(BASE / rel, target)
        if checkpoint is None:
            print(f'Parent original epoch {saved["epoch"]}; evaluating the same fixed development set as continuation epoch 0...', flush=True)
            start = time.perf_counter()
            baseline = validate(dev)
            if baseline['score'] is None or not math.isfinite(baseline['score']):
                raise ValueError('Parent model has no valid score on this fixed development set; continuation stopped.')
            best_score = baseline['score']
            record = dict(epoch=0, stage=0, horizon=0, seconds=time.perf_counter() - start, completed_updates=0)
            _record_validation(record, baseline)
            history.append(record)
            best_checkpoint = pack()
            save_last()
            save_checkpoint(out / 'best.pt', best_checkpoint)
        run_meta = dict(workflow=WORKFLOW, arm='replay_csd', mode=settings['mode'], device=device,
                        continuation_version=CONTINUATION_VERSION, parent_epoch=saved['epoch'],
                        parent_recorded_best_score=saved.get('best_score'),
                        remeasured_parent_score=history[0]['validation_score'],
                        parent_run=str(parent.resolve()), fingerprint=fingerprint, dataset=str(root),
                        planned_epochs=plan['epochs'], planned_updates=plan['epochs'] * plan['updates_per_epoch'],
                        train_parents=len(rows['train']), validation_parents=len(rows['validation']),
                        development_parents=len(dev), test_state_files_opened=False,
                        completed_epochs=cursor, completed_updates=cursor * plan['updates_per_epoch'],
                        best_epoch=best_epoch, best_score=best_score, status='full_csd_continuation_running_unvalidated')
        write_json(out / 'run.json', run_meta)
        session_start = time.perf_counter()
        durations = []
        while cursor < plan['epochs']:
            print(f'Continuation epoch {cursor + 1}/{plan["epochs"]}: ordinary H={plan["horizon"]}, ordinary plus full-terminal response; {plan["updates_per_epoch"]} updates.', flush=True)
            start = last_progress = time.perf_counter()
            sums = {}
            for update in range(plan['updates_per_epoch']):
                metrics = train_update(model, optimizer, factory, edges, saved['control_scales'], config, plan, rng, device)
                for key, value in metrics.items():
                    sums[key] = sums.get(key, 0.) + value
                if time.perf_counter() - last_progress >= 30:
                    elapsed = time.perf_counter() - start
                    print(f'  Completed {update + 1}/{plan["updates_per_epoch"]} updates; estimated update time remaining '
                          f'{elapsed / (update + 1) * (plan["updates_per_epoch"] - update - 1) / 60:.1f} min, excluding validation.', flush=True)
                    last_progress = time.perf_counter()
            cursor += 1
            update_seconds = time.perf_counter() - start
            record = dict(epoch=cursor, stage=0, horizon=plan['horizon'], full_terminal=True,
                          completed_updates=cursor * plan['updates_per_epoch'], update_seconds=update_seconds,
                          **{'train_' + k: v / plan['updates_per_epoch'] for k, v in sums.items()})
            end = cursor == plan['epochs']
            validation_start = time.perf_counter()
            if cursor % plan['every_epochs'] == 0 or end:
                score = validate(dev)
                _record_validation(record, score)
                if score['score'] is not None and score['score'] < best_score:
                    best_score, best_epoch = score['score'], cursor
                    best_checkpoint = pack()
            if cursor % plan['full_every_epochs'] == 0 or end:
                full = validate(rows['validation'])
                write_json(out / f'full_validation_epoch_{cursor:04d}.json', full)
                record['full_validation_score'] = full['score']
            record.update(validation_seconds=time.perf_counter() - validation_start, seconds=time.perf_counter() - start)
            history.append(record)
            
            save_last()
            save_checkpoint(out / 'best.pt', best_checkpoint)
            write_csv(out / 'history.csv', history)
            write_json(out / 'sampling_coverage.json', _coverage(factory, cursor * plan['updates_per_epoch']))
            run_meta.update(completed_epochs=cursor, completed_updates=cursor * plan['updates_per_epoch'],
                            best_epoch=best_epoch, best_score=best_score, session_seconds=time.perf_counter() - session_start)
            write_json(out / 'run.json', run_meta)
            durations.append(record['seconds'])
            print(f'Update {update_seconds:.1f}s, validation {record["validation_seconds"]:.1f}s; cumulative '
                  f'{run_meta["completed_updates"]}/{run_meta["planned_updates"]} updates; best continuation epoch {best_epoch}; '
                  f'estimated remaining {np.mean(durations) * (plan["epochs"] - cursor) / 60:.1f} min from completed epochs (full validation costs differ).', flush=True)
        run_meta['status'] = ('continuation_smoke_complete_no_improvement_claim' if settings['mode'] == 'smoke'
                              else 'bounded_continuation_complete_pending_independent_prediction_and_control')
        write_json(out / 'run.json', run_meta)
        write_csv(out / 'history.csv', history)
        write_json(out / 'sampling_coverage.json', _coverage(factory, cursor * plan['updates_per_epoch']))
        _write_report(out, run_meta, history)
        if not no_plot:
            plot_history(out, history, show)
        print(str(out), flush=True)
        print(run_meta['status'], flush=True)
        return out
    finally:
        lock.unlink(missing_ok=True)
