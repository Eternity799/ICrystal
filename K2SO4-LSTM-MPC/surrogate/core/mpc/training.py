"""Control-oriented training, fixed validation and checkpoint recovery."""
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

from ..data import GRID_KEYS, fit_scaler, read_manifest, stack_samples
from ..io import save_checkpoint, sha256, write_csv, write_json
from ..losses import full_csd_loss, response_loss
from ..model import ARCHITECTURE, FullCSDLSTM, rollout
from ..training import choose_device, persistence_output, rng_state, restore_rng
from .config import BASE, QUALITY_KEYS
from .data import MPCSampleFactory, load_selected_dataset, select_parent_ids
from .quality import fit_quality_scales, quality_components, quality_response_components, quality_tensor
from .ranking import compare_candidates, freeze_objective, trajectory_score


WORKFLOW = 'mpc_oriented_training_v2'


def source_hashes():
    files = list((BASE/'core').glob('*.py')) + list((BASE/'core'/'mpc').glob('*.py'))
    files += [BASE/name for name in ('train.py','evaluate.py','train_mpc.py','evaluate_mpc.py')]
    return {str(p.relative_to(BASE)).replace('\\','/'):sha256(p) for p in sorted(files)}


def run_fingerprint(config, arm, dataset_hash, sources):
    frozen = copy.deepcopy(config)
    frozen.pop('arms', None)
    frozen['active_arm'] = arm
    digest = hashlib.sha256(json.dumps(frozen, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()
    return dict(configuration=digest, dataset_manifest=dataset_hash, source=sources)


def weighted(parts, weights):
    return sum(parts[key]*float(value) for key,value in weights.items() if value != 0)


def loss_value(out, sample, model, edges, scales, config, arm):
    csd = full_csd_loss(out, sample['target_n'], sample['target_c'], model)
    quality = quality_components(out, sample, model.particle_mass, edges, scales)
    if arm == 'replay_csd':
        value = weighted(csd, config['loss'])
    elif arm == 'control_priority':
        value = weighted(quality, config['quality_weights'])+weighted(csd, config['auxiliary_loss'])
    else:
        raise ValueError('Unknown training arm.')
    return value, {**{'csd_'+k:v for k,v in csd.items()}, **{'quality_'+k:v for k,v in quality.items()}}


def paired_value(model, sample, edges, scales, config, arm):
    out = rollout(model, sample)
    absolute, parts = loss_value(out, sample, model, edges, scales, config, arm)
    qr = quality_response_components(out, sample, model.particle_mass, edges, scales)
    if arm == 'control_priority':
        value = absolute+weighted(qr, config['quality_weights'])
        factor = config['response']['weight']
    else:
        pn,tn = out['counts'],sample['target_n']
        predicted = torch.stack((pn[1:], pn[0:1].expand_as(pn[1:])), 1)
        truth = torch.stack((tn[1:], tn[0:1].expand_as(tn[1:])), 1)
        legacy = config['legacy_response']
        response = response_loss(predicted, truth, model, legacy['number_floor'], legacy['mass_floor'])
        value = absolute+sum(response.values())
        factor = legacy['weight']
    parts.update({'response_'+k:v for k,v in qr.items()})
    return factor*value, parts


def domain_from_train(rows):
    lower = np.full(6, np.inf); upper = np.full(6, -np.inf)
    for row in rows:
        q = quality_tensor(torch.tensor(row['counts'])[None], torch.tensor(row['concentration'])[None],
                           torch.tensor(row['particle_mass_kg']), torch.tensor(row['size_edges_m']),
                           torch.tensor(float(row['concentration'][0]),dtype=torch.float64))
        values = np.stack([q[k][0].detach().numpy() for k in
                          ('concentration','yield_fraction','d10_um','d50_um','d90_um','span')], -1)
        lower = np.minimum(lower, values.min(0)); upper = np.maximum(upper, values.max(0))
    return [lower.tolist(), upper.tolist()]


def forecast_ranking(row, branch, out, objective, domain, tolerance):
    origin = branch['origin']; h = branch['horizon']
    meta = dict(terminal=bool(branch.get('terminal', False)), origin=origin, horizon=h)
    previous = float(row['temperature'][origin]-row['temperature'][max(0,origin-1)])
    predicted, truth = [], []
    for j in range(len(branch['offsets_c'])):
        kwargs = dict(initial_concentration=float(row['concentration'][0]),
                      particle_mass=row['particle_mass_kg'], edges=row['size_edges_m'], objective=objective,
                      previous_rate=previous, quality_domain=domain, metadata=meta)
        nn = np.concatenate((row['counts'][origin:origin+1], out['counts'][j].detach().cpu().numpy()))
        cc = np.r_[row['concentration'][origin], out['concentration'][j].detach().cpu().numpy()]
        predicted.append(trajectory_score(nn, cc, branch['temperature'][j], **kwargs))
        truth.append(trajectory_score(branch['counts'][j], branch['concentration'][j], branch['temperature'][j], **kwargs))
    return compare_candidates(predicted, truth, tolerance)


def ranking_selection_penalty(rank):
    """Only PBM truth determines ranking eligibility; predicted ties or false rejection retain their penalty."""
    if rank['truth_feasible_count'] < 2 or rank['truth_tie']:
        return None
    value=rank.get('normalized_regret')
    if not rank['informative'] or value is None or not math.isfinite(value):
        return 1.
    return min(1.,max(0.,float(value)))


def fixed_validation(model, rows, config, scales, objective, domain, device, persistence=False):
    """Fixed origins, horizons and branches; average within parents first, with the same quality score for both arms."""
    factory = MPCSampleFactory(rows, model.history)
    edges = torch.tensor(rows[0]['size_edges_m'], dtype=torch.float64, device=device)
    was_training = model.training; model.eval()
    parent_values, parent_response, parent_terminal, parent_regret = [], [], [], []
    ns,ms = [], []
    invalid = projection = steps = terminal_groups = informative_groups = false_feasible = 0
    false_rejected = ranked_candidates = truth_informative_groups = 0
    last = time.perf_counter()
    try:
        with torch.no_grad():
            for i,row in enumerate(rows,1):
                values, responses, terminals, regrets = [], [], [], []
                try:
                    for h in config['selection']['horizons']:
                        windows = [factory.window(row,o,min(h,180-o)) for o in config['selection']['origins']]
                        for length in sorted({len(s['future_t']) for s in windows}):
                            group = [s for s in windows if len(s['future_t']) == length]
                            sample = stack_samples(group,device)
                            out = persistence_output(sample,model) if persistence else rollout(model,sample)
                            q = quality_components(out,sample,model.particle_mass,edges,scales)
                            csd = full_csd_loss(out,sample['target_n'],sample['target_c'],model)
                            value = float(weighted(q,config['quality_weights']))
                            if not math.isfinite(value):
                                raise ValueError('Nonfinite validation quality loss')
                            values.extend([value]*len(group))
                            ns.append(float(csd['number'])); ms.append(float(csd['mass']))
                            projection += int((out['alpha'] < 1-1e-12).sum()); steps += out['alpha'].numel()
                    ordinary = sorted([b for b in row['groups'] if len(b['offsets_c']) > 1 and not b.get('terminal',False)],
                                      key=lambda b:(b['origin'],b['file']))[:config['selection']['response_groups_per_parent']]
                    for b in ordinary:
                        for h in config['selection']['horizons']:
                            if h > b['horizon']:
                                continue
                            sample = factory.pair(row,b,h,device)
                            out = persistence_output(sample,model) if persistence else rollout(model,sample)
                            qr = quality_response_components(out,sample,model.particle_mass,edges,scales)
                            response = float(weighted(qr,config['quality_weights']))
                            if not math.isfinite(response):
                                raise ValueError('Nonfinite validation response')
                            responses.append(response)
                    for b in row['groups']:
                        if not b.get('terminal',False) or len(b['offsets_c']) <= 1:
                            continue
                        terminal_groups += 1
                        sample = factory.pair(row,b,b['horizon'],device)
                        out = persistence_output(sample,model) if persistence else rollout(model,sample)
                        endpoint_out = {k:v[:,-1:] for k,v in out.items() if isinstance(v,torch.Tensor)}
                        endpoint_sample = dict(sample, target_n=sample['target_n'][:,-1:], target_c=sample['target_c'][:,-1:])
                        q = quality_components(endpoint_out,endpoint_sample,model.particle_mass,edges,scales)
                        terminal_value=float(weighted(q,config['quality_weights']))
                        if not math.isfinite(terminal_value):raise ValueError('Nonfinite terminal quality error')
                        terminals.append(terminal_value)
                        rank = forecast_ranking(row,b,out,objective,domain,config['selection']['tie_tolerance'])
                        false_feasible += int(rank.get('false_feasible_count',0))
                        false_rejected += int(rank.get('false_rejected_feasible_count',0))
                        ranked_candidates += int(rank.get('candidate_count',0))
                        penalty=ranking_selection_penalty(rank)
                        if penalty is not None:
                            regrets.append(penalty);truth_informative_groups+=1
                        if rank.get('informative'):informative_groups+=1
                    if not values:
                        raise ValueError('No validation windows')
                    parent_values.append(float(np.mean(values)))
                    if responses: parent_response.append(float(np.mean(responses)))
                    if terminals: parent_terminal.append(float(np.mean(terminals)))
                    if regrets: parent_regret.append(float(np.mean(regrets)))
                except (ValueError,FloatingPointError) as exc:
                    invalid += 1
                    if i == 1:
                        print('Validation failure: '+str(exc),flush=True)
                if time.perf_counter()-last >= 30:
                    print(f"Fixed quality validation {i}/{len(rows)}; terminal groups {terminal_groups}",flush=True); last=time.perf_counter()
    finally:
        model.train(was_training)
    absolute = float(np.mean(parent_values)) if parent_values else None
    response = float(np.mean(parent_response)) if parent_response else None
    terminal = float(np.mean(parent_terminal)) if parent_terminal else None
    regret = float(np.mean(parent_regret)) if parent_regret else None
    score = None
    if invalid == 0 and absolute is not None and response is not None:
        score = absolute+config['selection']['response_weight']*response
        if terminal is not None: score += config['selection']['terminal_weight']*terminal
        if regret is not None: score += config['selection']['ranking_weight']*regret
        score += float(false_feasible+false_rejected)/max(ranked_candidates,1)
        if not math.isfinite(score): score=None
    return dict(score=score, absolute=absolute, response=response, terminal=terminal, normalized_regret=regret,
                invalid_parents=invalid, parents=len(rows), terminal_groups=terminal_groups,
                informative_terminal_groups=informative_groups, false_feasible_count=false_feasible,
                truth_informative_terminal_groups=truth_informative_groups,
                false_rejected_feasible_count=false_rejected,ranked_candidates=ranked_candidates,
                number_l1=float(np.mean(ns)) if ns else None,mass_l1=float(np.mean(ms)) if ms else None,
                projection_count=projection,prediction_steps=steps,
                scope='Fixed development quality score with finite terminal candidates; not MPC qualification or global optimality')


def coverage(factory):
    return dict(parent_draws=dict(factory.seen_parents),
                group_draws={str(k):v for k,v in factory.seen_groups.items()},
                total_draws=factory.draws, availability=factory.availability,
                effective_mixture=getattr(factory,'last_mixture',{}))


def payload(model,scaler,grid,config,scales,domain,objective,fingerprint,epoch,best_score,best_epoch):
    return dict(format_version=1,architecture=ARCHITECTURE,workflow=WORKFLOW,
                model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                scaler=scaler,grid=grid,config=config,control_scales=scales,quality_domain=domain,
                objective_snapshot=objective,fingerprint=fingerprint,epoch=epoch,
                best_score=best_score,best_epoch=best_epoch)


def train_arm(config,arm,rows,dev_rows,grid,scaler,scales,domain,objective,fingerprint,out,resume=False,no_plot=False):
    device = choose_device(config['device']); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    lock = out/'RUNNING.lock'
    with lock.open('x',encoding='utf-8') as f: f.write(f'arm={arm}\n')
    try:
        random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config['seed']); torch.backends.cudnn.benchmark=False
            torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        model=FullCSDLSTM(scaler,grid,config['model']).to(device)
        edges=torch.tensor(grid['size_edges_m'],dtype=torch.float64,device=device)
        profile=config['profiles'][config['mode']]
        optimizer=torch.optim.AdamW(model.parameters(),lr=profile['stages'][0]['lr'],weight_decay=config['sampling']['weight_decay'])
        rng=np.random.default_rng(config['seed']+1)
        factory=MPCSampleFactory(rows['train'],model.history,time_edges=config['sampling']['time_edges'])
        if any(s['full_terminal'] for s in profile['stages']) and not factory.availability['terminal']['groups']:
            raise ValueError('The full-terminal training stage has no terminal groups; inspect selected parents instead of substituting short branches.')
        plan=[(si,local,stage) for si,stage in enumerate(profile['stages']) for local in range(1,stage['epochs']+1)]
        history=[]; cursor=0; best_score=math.inf; best_epoch=None; stale=full_stale=0; stage_best=full_best=math.inf
        stage_seen=-1; latest=None; actual_epochs=0
        if resume:
            checkpoint=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
            if checkpoint.get('fingerprint') != fingerprint or checkpoint.get('workflow') != WORKFLOW:
                raise ValueError('Resume configuration, data, objective or source changed; start a separate run.')
            model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
            restore_rng(checkpoint['rng'],rng)
            cursor=checkpoint['cursor'];history=checkpoint['history'];best_score=checkpoint['best_score'];best_epoch=checkpoint['best_epoch']
            if best_score is None:best_score=math.inf
            stale,full_stale,stage_best,full_best,stage_seen,actual_epochs=[checkpoint[k] for k in
                 ('stale','full_stale','stage_best','full_best','stage_seen','actual_epochs')]
            state=checkpoint.get('factory_state',{})
            factory.seen_parents.update(state.get('parents',{}));factory.seen_groups.update(state.get('groups',{}))
            factory.draws=state.get('draws',0);factory.last_mixture=state.get('mixture',{})
        write_json(out/'config_resolved.json',config);write_json(out/'selected_parents.json',
                   dict(train=[r['id'] for r in rows['train']],validation=[r['id'] for r in rows['validation']],development=[r['id'] for r in dev_rows]))
        write_json(out/'control_scales.json',scales);write_json(out/'objective_snapshot.json',dict(objective=objective,quality_domain=domain))
        run=dict(workflow=WORKFLOW,arm=arm,mode=config['mode'],dataset=config['dataset'],device=device,
                 fingerprint=fingerprint,planned_epochs=len(plan),planned_updates=sum(s['updates_per_epoch'] for _,_,s in plan),
                 train_parents=len(rows['train']),validation_parents=len(rows['validation']),development_parents=len(dev_rows),
                 test_state_files_opened=False,status='control_oriented_training_running_unvalidated')
        for rel in fingerprint['source']:
            target=out/'source'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/rel,target)
        def save_last():
            value=payload(model,scaler,grid,config,scales,domain,objective,fingerprint,cursor,
                          best_score if math.isfinite(best_score) else None,best_epoch)
            value.update(optimizer=optimizer.state_dict(),rng=rng_state(rng),cursor=cursor,history=history,
                         stale=stale,full_stale=full_stale,stage_best=stage_best,full_best=full_best,
                         stage_seen=stage_seen,actual_epochs=actual_epochs,
                         factory_state=dict(parents=dict(factory.seen_parents),groups=dict(factory.seen_groups),
                                            draws=factory.draws,mixture=getattr(factory,'last_mixture',{})))
            save_checkpoint(out/'last.pt',value)
        if not resume:
            print(arm+': checking the random initial model and persistence baseline...',flush=True)
            baseline=fixed_validation(model,dev_rows,config,scales,objective,domain,device,persistence=True)
            write_json(out/'persistence_validation.json',baseline)
            latest=fixed_validation(model,dev_rows,config,scales,objective,domain,device)
            history.append(dict(epoch=0,stage=-1,horizon=0,seconds=0.,validation_score=latest['score'],**{'validation_'+k:v for k,v in latest.items() if k!='score'}))
            if latest['score'] is not None:
                best_score=latest['score'];best_epoch=0
                save_checkpoint(out/'best.pt',payload(model,scaler,grid,config,scales,domain,objective,fingerprint,0,best_score,0))
            save_last()
        write_json(out/'run.json',run)
        started=time.perf_counter(); stage_seconds={}
        while cursor < len(plan):
            si,local,stage=plan[cursor]
            if si != stage_seen:
                stage_seen=si;stale=full_stale=0;stage_best=full_best=math.inf
            for g in optimizer.param_groups: g['lr']=stage['lr']
            model.train();began=time.perf_counter(); sums={};last=began
            print(f'{arm}: stage {si+1}, H={stage["horizon"]}, epoch {local}/{stage["epochs"]}, full terminal backpropagation={stage["full_terminal"]}',flush=True)
            for update in range(stage['updates_per_epoch']):
                optimizer.zero_grad(set_to_none=True)
                sample=factory.sample(config['sampling']['batch_size'],stage['horizon'],rng,device)
                prediction=rollout(model,sample)
                value,parts=loss_value(prediction,sample,model,edges,scales,config,arm)
                if not torch.isfinite(value): raise FloatingPointError('Nonfinite training loss; stopping and retaining the previous complete epoch.')
                value.backward()
                n_groups=1 if stage['full_terminal'] else config['response']['groups_per_update']
                mixture={'short':0.,'long':0.,'terminal':1.} if stage['full_terminal'] else config['response']['mixture']
                pairs=factory.sample_pairs(n_groups,stage['horizon'],rng,device,mixture=mixture,full_terminal=stage['full_terminal'])
                response_total=0.
                for pair in pairs:
                    pair_value,pair_parts=paired_value(model,pair,edges,scales,config,arm)
                    if not torch.isfinite(pair_value): raise FloatingPointError('Nonfinite paired loss.')
                    (pair_value/len(pairs)).backward();response_total+=float(pair_value.detach())/len(pairs)
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['sampling']['grad_clip'])
                if not torch.isfinite(norm): raise FloatingPointError('Nonfinite training gradient.')
                optimizer.step()
                for key,val in parts.items(): sums[key]=sums.get(key,0.)+float(val.detach())
                sums['loss']=sums.get('loss',0.)+float(value.detach())
                sums['pair_loss']=sums.get('pair_loss',0.)+response_total
                if time.perf_counter()-last>=30:
                    elapsed=time.perf_counter()-began
                    print(f'  Update {update+1}/{stage["updates_per_epoch"]}; estimated remaining updates {elapsed/(update+1)*(stage["updates_per_epoch"]-update-1)/60:.1f} min, excluding validation.',flush=True);last=time.perf_counter()
            cursor+=1;actual_epochs+=1;update_seconds=time.perf_counter()-began
            stage_end=cursor==len(plan) or plan[cursor][0]!=si
            record=dict(epoch=cursor,stage=si,horizon=stage['horizon'],full_terminal=stage['full_terminal'],
                        update_seconds=update_seconds,**{'train_'+k:v/stage['updates_per_epoch'] for k,v in sums.items()})
            validation_seconds=0.
            if cursor%config['selection']['every_epochs']==0 or stage_end:
                tick=time.perf_counter();latest=fixed_validation(model,dev_rows,config,scales,objective,domain,device)
                validation_seconds+=time.perf_counter()-tick
                record.update(validation_score=latest['score'],**{'validation_'+k:v for k,v in latest.items() if k!='score'})
                score=latest['score']
                if score is not None and score<stage_best:stage_best=score;stale=0
                else:stale+=1
                if score is not None and score<best_score:
                    best_score=score;best_epoch=cursor
                    save_checkpoint(out/'best.pt',payload(model,scaler,grid,config,scales,domain,objective,fingerprint,cursor,best_score,best_epoch))
            if cursor%config['selection']['full_every_epochs']==0 or stage_end:
                tick=time.perf_counter();full=fixed_validation(model,rows['validation'],config,scales,objective,domain,device)
                validation_seconds+=time.perf_counter()-tick
                write_json(out/f'full_validation_epoch_{cursor:04d}.json',full)
                record['full_validation_score']=full['score']
                if full['score'] is not None and full['score']<full_best:full_best=full['score'];full_stale=0
                else:full_stale+=1
            record.update(validation_seconds=validation_seconds,seconds=time.perf_counter()-began,
                          unique_train_parents=len(factory.seen_parents),unique_pair_groups=len(factory.seen_groups))
            history.append(record);stage_seconds.setdefault(si,[]).append(record['seconds'])
            patience=profile['patience']
            if patience and local>=5 and stale>=patience and full_stale>=2:
                skipped=0
                while cursor<len(plan) and plan[cursor][0]==si:cursor+=1;skipped+=1
                print(f"Development and full validation stalled; skip {skipped} remaining epochs in this stage and continue the curriculum. This is not qualification.",flush=True)
            write_csv(out/'history.csv',history);write_json(out/'sampling_coverage.json',coverage(factory));save_last()
            run.update(completed_epochs=actual_epochs,plan_cursor=cursor,best_epoch=best_epoch,
                       best_score=best_score if math.isfinite(best_score) else None,session_seconds=time.perf_counter()-started)
            write_json(out/'run.json',run)
            remaining=sum(np.mean(stage_seconds.get(sidx,[record['seconds']*s['horizon']/stage['horizon']*s['updates_per_epoch']/stage['updates_per_epoch']])) for sidx,_,s in plan[cursor:])
            print(f'Update {update_seconds:.1f}s, validation {validation_seconds:.1f}s; fixed quality score {record.get("validation_score")}; rough remaining {remaining/60:.1f} min (unmeasured stages extrapolated).',flush=True)
        run.update(status='control_training_smoke_complete_no_accuracy_claim' if config['mode']=='smoke' else 'control_oriented_training_complete_pending_independent_prediction_and_control')
        run.update(best_epoch=best_epoch,best_score=best_score if math.isfinite(best_score) else None,completed_epochs=actual_epochs)
        if best_epoch is None:run['status']='training_finished_no_valid_validation_candidate'
        write_json(out/'run.json',run);write_csv(out/'history.csv',history)
        write_json(out/'sampling_coverage.json',coverage(factory))
        (out/'report.md').write_text(f"""# Control-oriented surrogate training

Arm: {arm}; mode: {config["mode"]}.

"""
            f"""Status: {run["status"]}; best planned epoch: {best_epoch}; actual epochs: {actual_epochs}.

"""
            'Both arms share architecture, data, initialization and sampling. Concentration and yield are not weighted twice. Full CSD is the internal state; the control-priority arm adds quality supervision.\n\n'
            'True states enter only at the origin; future CSD/C labels only score losses. Terminal backpropagation uses saved full branches without true-state resets.\n\n'
            'Best weights use the fixed development quality/response/terminal score. Full validation covers all selected validation parents. No test states, PBM or closed loop were used.\n\n'
            'Terminal branches cover nominal initial conditions and finite action templates. Ties and infeasible plans are reported separately; these are not global optimization or MPC qualification.\n',encoding='utf-8')
        if not no_plot:
            from .reporting import plot_training
            plot_training(out,history)
        print(str(out),flush=True);print(run['status'],flush=True)
        return out
    finally:
        lock.unlink(missing_ok=True)


def run_suite(config,config_path,resume=None,no_plot=False):
    if config['mode']=='formal' and len(config['arms'])!=1:
        raise ValueError('Select one arm explicitly for formal training.')
    manifest=read_manifest(config['dataset'])
    if config['mode']=='formal' and manifest['mode']!='formal':raise ValueError('Formal training requires a complete formal dataset.')
    profile=config['profiles'][config['mode']]
    ids={role:select_parent_ids(manifest,role,profile[role+'_per_family'],config['seed']+i)
         for i,role in enumerate(('train','validation'))}
    rows=load_selected_dataset(config['dataset'],('train','validation'),ids,manifest)
    dev_manifest=dict(manifest,completed=[e for e in manifest['completed'] if e['id'] in set(ids['validation'])],
                      plan=[e for e in manifest['plan'] if e['id'] in set(ids['validation'])])
    dev_ids=set(select_parent_ids(dev_manifest,'validation',profile['development_per_family'],config['seed']+7))
    dev_rows=[r for r in rows['validation'] if r['id'] in dev_ids]
    scaler=fit_scaler(rows['train'],config['model']['solubility'])
    scales=fit_quality_scales(rows['train'],config['scale_floors']['absolute'],config['scale_floors']['response'])
    label_scale_statistics=copy.deepcopy(scales)
    if config['absolute_scale_mode']=='process':
        scales['absolute']=copy.deepcopy(config['scale_floors']['absolute'])
    grid={k:rows['train'][0][k].tolist() for k in GRID_KEYS};domain=domain_from_train(rows['train'])
    objective=freeze_objective(config['objective_config']);objective['coefficients']=config['model']['solubility']
    frozen=copy.deepcopy(config);frozen['frozen_objective']=objective
    hashes=source_hashes();dataset_hash=sha256(Path(config['dataset'])/'manifest.json')
    if resume:
        out=Path(resume).resolve()
        if not (out/'run.json').is_file() or not (out/'last.pt').is_file():raise ValueError('Resume requires one arm directory containing run.json and last.pt.')
        arm=json.loads((out/'run.json').read_text(encoding='utf-8'))['arm']
        if arm not in config['arms']:raise ValueError('The resumed arm differs from --arm.')
        fingerprint=run_fingerprint(frozen,arm,dataset_hash,hashes)
        return train_arm(frozen,arm,rows,dev_rows,grid,scaler,scales,domain,objective,fingerprint,out,True,no_plot)
    suite=Path(config['output_root'])/config['mode']/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f_utc')
    suite.mkdir(parents=True,exist_ok=False);shutil.copy2(config_path,suite/'config_original.yaml')
    write_json(suite/'label_scale_statistics.json',label_scale_statistics)
    write_json(suite/'selected_parents.json',dict(ids,development=[r['id'] for r in dev_rows]))
    arms=[]
    for arm in config['arms']:
        fingerprint=run_fingerprint(frozen,arm,dataset_hash,hashes)
        arms.append(str(train_arm(frozen,arm,rows,dev_rows,grid,scaler,scales,domain,objective,fingerprint,suite/arm,False,no_plot)))
    write_json(suite/'suite.json',dict(mode=config['mode'],arms=arms,dataset=config['dataset'],status='training_finished_pending_independent_validation'))
    print('Arm output directories: '+str(suite),flush=True)
    return suite
