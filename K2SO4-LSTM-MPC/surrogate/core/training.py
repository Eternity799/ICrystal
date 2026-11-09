"""From-scratch full-CSD training and checkpoint recovery."""
import copy
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .config import BASE
from .data import GRID_KEYS, load_dataset, read_manifest, fit_scaler, SampleFactory, stack_samples
from .io import sha256, write_json, write_csv, save_checkpoint
from .losses import full_csd_loss, weighted, pair_losses
from .model import ARCHITECTURE, FullCSDLSTM, rollout


def source_hashes():
    files = sorted((BASE/'core').glob('*.py')) + [BASE/'train.py', BASE/'evaluate.py']
    return {str(p.relative_to(BASE)).replace('\\', '/'):sha256(p) for p in files}


def rng_state(rng):
    return dict(sampler=copy.deepcopy(rng.bit_generator.state), python=random.getstate(),
                numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state, rng):
    rng.bit_generator.state = state['sampler']
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def choose_device(request):
    if request == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA was requested but is unavailable.')
    return 'cuda' if request == 'cuda' or (request == 'auto' and torch.cuda.is_available()) else 'cpu'


def persistence_output(sample, model):
    n = sample['n'][:, -1:, :].expand_as(sample['target_n'])
    mass = (n*model.particle_mass).sum(-1)
    total = sample['total'][:, None].expand_as(mass)
    return dict(counts=n, concentration=total-mass, mass=mass, total=total,
                raw_mass=mass, alpha=torch.ones_like(mass), current_n=n)


def fixed_validation(model, rows, config, device, persistence=False):
    """Average within each base trajectory, then across trajectories; validation sets, losses and response scales remain fixed across stages."""
    factory = SampleFactory(rows, model.history)
    was_training = model.training
    model.eval()
    parent_scores, parent_responses, parent_pair_scores, numbers, masses = [], [], [], [], []
    projection, steps = 0, 0
    last_progress = time.perf_counter()
    try:
        with torch.no_grad():
            for parent_index,row in enumerate(rows, 1):
                values, ns, ms, responses, pair_scores = [], [], [], [], []
                for h in config['selection']['horizons']:
                    samples = [factory.window(row, origin, min(h, 180-origin))
                               for origin in config['selection']['origins']]
                    # Batch truncated windows by their available future length.
                    for length in sorted({s['future_t'].shape[0] for s in samples}):
                        selected = [s for s in samples if s['future_t'].shape[0] == length]
                        s = stack_samples(selected, device)
                        out = persistence_output(s, model) if persistence else rollout(model, s)
                        parts = full_csd_loss(out, s['target_n'], s['target_c'], model)
                        value = weighted(parts, config['loss'])
                        if not torch.isfinite(value):
                            return dict(score=None, invalid=True, reason='nonfinite validation-window prediction')
                        # Weight each length-group mean by its number of windows.
                        values.extend([float(value)]*len(selected))
                        ns.extend([float(parts['number'])]*len(selected)); ms.extend([float(parts['mass'])]*len(selected))
                        projection += int((out['alpha'] < 1-1e-12).sum()); steps += out['alpha'].numel()
                    for branch in row['groups']:
                        if len(branch['offsets_c']) <= 1:
                            continue
                        s = factory.pair(row, branch, h, device)
                        if persistence:
                            from .losses import response_loss
                            out = persistence_output(s, model)
                            pn = out['counts']; tn = s['target_n']
                            pair_pred = torch.stack((pn[1:], pn[0:1].expand_as(pn[1:])), 1)
                            pair_true = torch.stack((tn[1:], tn[0:1].expand_as(tn[1:])), 1)
                            resp = response_loss(pair_pred, pair_true, model, config['response']['number_floor'], config['response']['mass_floor'])
                            absolute_parts = full_csd_loss(out, s['target_n'], s['target_c'], model)
                        else:
                            absolute_parts, resp, out = pair_losses(model, s, config['response'])
                        pair_score = weighted(absolute_parts, config['loss'])
                        response = sum(resp.values())
                        if not torch.isfinite(response) or not torch.isfinite(pair_score):
                            return dict(score=None, invalid=True, reason='nonfinite validation-response prediction')
                        responses.append(float(response))
                        pair_scores.append(float(pair_score))
                        projection += int((out['alpha'] < 1-1e-12).sum()); steps += out['alpha'].numel()
                if not responses:
                    raise ValueError(f"Validation base trajectory {row['id']} has no comparable temperature-response group.")
                parent_scores.append(float(np.mean(values))); parent_responses.append(float(np.mean(responses)))
                parent_pair_scores.append(float(np.mean(pair_scores)))
                numbers.append(float(np.mean(ns))); masses.append(float(np.mean(ms)))
                if time.perf_counter()-last_progress >= 30:
                    print(f"  Fixed validation trajectory {parent_index}/{len(rows)} (persistence baseline)" if persistence else
                          f"  Fixed validation trajectory {parent_index}/{len(rows)}", flush=True)
                    last_progress = time.perf_counter()
        absolute = float(np.mean(parent_scores)); response = float(np.mean(parent_responses))
        pair_absolute = float(np.mean(parent_pair_scores))
        return dict(score=absolute+config['selection']['response_weight']*(pair_absolute+response), invalid=False,
                    absolute=absolute, pair_absolute=pair_absolute, response=response, number=float(np.mean(numbers)),
                    mass=float(np.mean(masses)), projection_count=projection, prediction_steps=steps)
    finally:
        model.train(was_training)


def make_payload(model, scaler, grid, config, epoch, best_score, best_epoch, fingerprint):
    return dict(format_version=1, architecture=ARCHITECTURE,
                model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                scaler=scaler, grid=grid, config=config, epoch=epoch,
                best_score=best_score, best_epoch=best_epoch, fingerprint=fingerprint)


def run_training(config, config_path, resume=None):
    device = choose_device(config['device'])
    print('Reading complete nominal data and checking train/validation files, branches and inventory closure; test states are not opened...', flush=True)
    rows = load_dataset(config['dataset'], ('train', 'validation'))
    manifest = read_manifest(config['dataset'])
    profile = config['profiles'][config['mode']]
    for role, key in [('train', 'min_train_parents'), ('validation', 'min_validation_parents')]:
        if len(rows[role]) < profile[key]:
            raise ValueError(f"Only {len(rows[role])} {role} base trajectories; this mode requires {profile[key]}.")
    if config['mode'] == 'formal' and manifest['mode'] != 'formal':
        raise ValueError('Formal training cannot use smoke data. Specify a complete formal dataset with --dataset.')
    if any(not any(len(b['offsets_c']) > 1 for b in r['groups']) for r in rows['validation']):
        raise ValueError('Fixed validation requires at least one nonzero temperature branch per base trajectory.')
    if not all(SampleFactory(rows[r], config['model']['history']).paired for r in ('train', 'validation')):
        raise ValueError('Train and validation must both have usable paired temperature branches.')
    fingerprint = dict(configuration=hashlib.sha256(json.dumps(config, sort_keys=True).encode('utf-8')).hexdigest(),
                       dataset_manifest=sha256(Path(config['dataset'])/'manifest.json'), source=source_hashes())
    if resume:
        out = Path(resume).resolve()
        if not (out/'run.json').is_file() or not (out/'last.pt').is_file():
            raise ValueError('Resumption requires this workflow run.json and last.pt; legacy models cannot be parent weights.')
        previous = json.loads((out/'run.json').read_text(encoding='utf-8'))
        if previous['fingerprint'] != fingerprint:
            raise ValueError('Configuration, dataset manifest or source changed; start a new from-scratch run instead of mixing implementations.')
    else:
        out = Path(config['output_root'])/config['mode']/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f_utc')
        out.mkdir(parents=True, exist_ok=False)
        previous = None
    lock = out/'RUNNING.lock'
    with lock.open('x', encoding='utf-8') as f:
        f.write(f'pid={os.getpid()}\n')
    run = None
    try:
        random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config['seed'])
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
        rng = np.random.default_rng(config['seed']+1)
        grid = {k:rows['train'][0][k].tolist() for k in GRID_KEYS}
        scaler = fit_scaler(rows['train'], config['model']['solubility']) if not resume else None
        checkpoint = torch.load(out/'last.pt', map_location='cpu', weights_only=False) if resume else None
        if checkpoint:
            if checkpoint['architecture'] != ARCHITECTURE or checkpoint['fingerprint'] != fingerprint:
                raise ValueError('Invalid resumed checkpoint architecture or fingerprint.')
            scaler = checkpoint['scaler']
            if scaler['train_parent_ids'] != [r['id'] for r in rows['train']] or grid != checkpoint['grid']:
                raise ValueError('Resumed training base trajectories or physical grid differ.')
        model = FullCSDLSTM(scaler, grid, config['model']).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=profile['stages'][0]['lr'], weight_decay=config['sampling']['weight_decay'])
        factory = SampleFactory(rows['train'], config['model']['history'], config['sampling']['time_edges'])
        history, epoch_done, best_score, best_epoch = [], 0, math.inf, None
        timings = {}
        if checkpoint:
            model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
            for state in optimizer.state.values():
                for k,v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            restore_rng(checkpoint['rng'], rng)
            history = checkpoint['history']; epoch_done = checkpoint['epoch']
            best_score, best_epoch = checkpoint['best_score'], checkpoint['best_epoch']
            timings = checkpoint['timings']
        else:
            shutil.copyfile(config_path, out/'config_original.yaml')
            for relative in fingerprint['source']:
                dest = out/'source'/relative; dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(BASE/relative, dest)
            write_json(out/'config_resolved.json', config)
            write_json(out/'scaler.json', scaler)
        plan = [(stage_index, local_epoch, stage)
                for stage_index,stage in enumerate(profile['stages']) for local_epoch in range(1, stage['epochs']+1)]
        if epoch_done > len(plan):
            raise ValueError('Checkpoint epoch exceeds the training plan.')
        run = previous or dict(architecture=ARCHITECTURE, initialization='random_all_parameters_trainable',
            fingerprint=fingerprint, config=config, dataset=config['dataset'], mode=config['mode'],
            train_parent_ids=[r['id'] for r in rows['train']], validation_parent_ids=[r['id'] for r in rows['validation']],
            test_state_files_opened=False, parameters=sum(p.numel() for p in model.parameters()),
            selection='fixed validation horizons and response groups; no test',
            scope='ideal actual-temperature prediction; no PBM training calls, no MPC',
            environment=dict(python=sys.version, pytorch=torch.__version__, numpy=np.__version__,
                             cuda=torch.version.cuda, tf32=False,
                             accelerator=torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'))
        run.update(status='training_from_scratch', device=device, planned_epochs=len(plan), completed_epochs=epoch_done)
        write_json(out/'run.json', run)
        print(f'Output: {out}\nFrom-scratch training; train {len(rows["train"])} trajectories, validation {len(rows["validation"])} trajectories; test states are not read.\n'
              f'Device {device}; trainable parameters {run["parameters"]:,}; planned epochs {len(plan)}.', flush=True)

        def save_last(epoch):
            payload = make_payload(model, scaler, grid, config, epoch, best_score, best_epoch, fingerprint)
            payload.update(optimizer=optimizer.state_dict(), rng=rng_state(rng), history=history, timings=timings)
            save_checkpoint(out/'last.pt', payload)

        if not resume:
            baseline = fixed_validation(model, rows['validation'], config, device, persistence=True)
            score = fixed_validation(model, rows['validation'], config, device)
            write_json(out/'persistence_validation.json', baseline)
            history.append(dict(epoch=0, stage=-1, horizon=0, validation_score=score['score'],
                                validation_number=score.get('number'), validation_mass=score.get('mass'),
                                validation_response=score.get('response'), invalid_validation=score['invalid'], seconds=0.))
            if score['score'] is not None:
                best_score, best_epoch = score['score'], 0
                save_checkpoint(out/'best.pt', make_payload(model, scaler, grid, config, 0, best_score, best_epoch, fingerprint))
            write_csv(out/'history.csv', history); save_last(0)
            print(f'Epoch-0 random-initialization score: {score["score"]}; persistence baseline: {baseline["score"]}', flush=True)

        started = time.perf_counter()
        for epoch in range(epoch_done+1, len(plan)+1):
            stage_index, local_epoch, stage = plan[epoch-1]
            for group in optimizer.param_groups:
                group['lr'] = stage['lr']
            model.train(); sums = {}; epoch_start = time.perf_counter()
            print(f'Stage {stage_index+1}, H={stage["horizon"]}, epoch {local_epoch}/{stage["epochs"]}: updating...', flush=True)
            last_progress = epoch_start
            for update in range(stage['updates_per_epoch']):
                optimizer.zero_grad(set_to_none=True)
                sample = factory.sample(config['sampling']['batch_size'], stage['horizon'], rng, device)
                out_prediction = rollout(model, sample)
                parts = full_csd_loss(out_prediction, sample['target_n'], sample['target_c'], model)
                loss = weighted(parts, config['loss'])
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite trajectory loss; stopping at the previous fully committed epoch.')
                loss.backward()
                sums['train_absolute'] = sums.get('train_absolute', 0.)+float(loss.detach())
                for k,v in parts.items():
                    sums['train_'+k] = sums.get('train_'+k, 0.)+float(v.detach())
                pair_samples = factory.sample_pairs(config['response']['groups_per_update'], stage['horizon'], rng, device)
                pair_value = 0.
                for sample in pair_samples:
                    absolute, response, _ = pair_losses(model, sample, config['response'])
                    value = weighted(absolute, config['loss'])+sum(response.values())
                    if not torch.isfinite(value):
                        raise FloatingPointError('Nonfinite paired-response loss; stopping.')
                    (config['response']['weight']*value/len(pair_samples)).backward()
                    pair_value += float(sum(response.values()).detach())/len(pair_samples)
                sums['train_response'] = sums.get('train_response', 0.)+pair_value
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['sampling']['grad_clip'])
                if not torch.isfinite(norm):
                    raise FloatingPointError('Nonfinite training gradient; stopping.')
                optimizer.step()
                if time.perf_counter()-last_progress >= 30:
                    elapsed = time.perf_counter()-epoch_start
                    print(f'  Update {update+1}/{stage["updates_per_epoch"]}; elapsed this epoch {elapsed/60:.1f} min; '
                          f'estimated remaining update time {elapsed/(update+1)*(stage["updates_per_epoch"]-update-1)/60:.1f} min from the current mean.', flush=True)
                    last_progress = time.perf_counter()
            print('Scoring fixed validation windows and responses...', flush=True)
            validation = fixed_validation(model, rows['validation'], config, device)
            seconds = time.perf_counter()-epoch_start
            record = dict(epoch=epoch, stage=stage_index, horizon=stage['horizon'], lr=stage['lr'],
                          **{k:v/stage['updates_per_epoch'] for k,v in sums.items()},
                          validation_score=validation['score'], validation_number=validation.get('number'),
                          validation_mass=validation.get('mass'), validation_response=validation.get('response'),
                          invalid_validation=validation['invalid'], projection_count=validation.get('projection_count'),
                          prediction_steps=validation.get('prediction_steps'), seconds=seconds)
            history.append(record)
            if validation['score'] is not None and validation['score'] < best_score:
                best_score, best_epoch = validation['score'], epoch
                save_checkpoint(out/'best.pt', make_payload(model, scaler, grid, config, epoch, best_score, best_epoch, fingerprint))
            key = str(stage_index); timings.setdefault(key, []).append(seconds)
            write_csv(out/'history.csv', history); save_last(epoch)
            run.update(completed_epochs=epoch, best_epoch=best_epoch, best_score=best_score if math.isfinite(best_score) else None,
                       session_seconds=time.perf_counter()-started)
            write_json(out/'run.json', run)
            remaining = 0.; estimated_stages = []
            for si, _, future in plan[epoch:]:
                if str(si) in timings:
                    remaining += float(np.mean(timings[str(si)]))
                else:
                    # Estimate unvisited stages from their update counts and rollout horizons.
                    remaining += seconds*(future['updates_per_epoch']/stage['updates_per_epoch'])*(future['horizon']/stage['horizon'])
                    estimated_stages.append(si+1)
            print(f'Epoch {epoch}/{len(plan)}; number error {validation.get("number")}; mass error {validation.get("mass")}; '
                  f'fixed score {validation["score"]}; epoch time {seconds/60:.1f} min; estimated remaining {remaining/60:.1f} min'
                  + (' (unvisited stages extrapolated by horizon).' if estimated_stages else '。'), flush=True)
        status = 'smoke_complete_no_accuracy_conclusion' if config['mode'] == 'smoke' else 'full_csd_training_complete_not_rollout_or_closed_loop_validated'
        if not (out/'best.pt').is_file():
            status = 'training_finished_all_validation_invalid_no_candidate'
        run.update(status=status, completed_epochs=len(plan), best_epoch=best_epoch,
                   best_score=best_score if math.isfinite(best_score) else None)
        write_json(out/'run.json', run)
        lines = ['# Full-CSD from-scratch training report', '', f"Status: {status}",
                 f"Architecture: {ARCHITECTURE}; all parameters trainable; best epoch: {best_epoch}.",
                 'Random initialization; every scale is fitted to train trajectories. Fixed validation scoring remains the same across horizon stages; test states are not read.',
                 'Recursive feedback uses predicted CSD/concentration; future states enter losses only. Candidate temperatures define known actual interval paths.',
                 'best.pt minimizes the fixed validation score; last.pt stores resumption state. No legacy model is loaded.', '',
                 '|Epoch|Training H|Validation number relative L1|Validation mass relative L1|Response component|Fixed score|Invalid|', '|---|---|---|---|---|---|---|']
        for r in history:
            lines.append('|'+ '|'.join(str(r.get(k)) for k in ('epoch','horizon','validation_number','validation_mass',
                         'validation_response','validation_score','invalid_validation'))+'|')
        lines += ['', 'Number/mass errors are trajectory-balanced means over fixed short windows, not 180-step rollout accuracy. Response includes the full-200-bin temperature-response amplitude and is not direction accuracy.',
                  'Smoke checks interfaces only. Learn uses development samples, not independent generalization evidence. Run evaluate.py for one-step, full rollout, temperature-response and persistence checks.',
                  'This run does not replace a controller, run PBM/MPC, or add robustness metrics.']
        (out/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        from .plotting import training_plots
        try:
            training_plots(out, history)
        except (ImportError, RuntimeError, OSError) as exc:
            run['plot_error'] = f'{type(exc).__name__}: {exc}'
            write_json(out/'run.json', run)
            print(f"Weights and training CSV were saved, but plotting failed: {exc}", flush=True)
        print(out, flush=True); print(status, flush=True)
        return out
    except BaseException as exc:
        if run is not None:
            run.update(status='training_interrupted_resume_previous_complete_epoch', error=f'{type(exc).__name__}: {exc}')
            write_json(out/'run.json', run)
        raise
    finally:
        lock.unlink(missing_ok=True)
