"""Configuration, prediction, offline cooling optimization and four-policy comparison."""
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
LABELS = ('MPC', 'linear', 'empirical', 'B3')
OBJECTIVE_KEYS = ('objective','target','constraints','normalization','weights','empirical_nodes')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def json_default(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=json_default)+'\n', encoding='utf-8')


def write_csv(path, rows):
    if not rows: return
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader(); writer.writerows(rows)


def output_folder(kind):
    folder = ROOT/'outputs'/kind/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f_utc')
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def verify_release():
    """Check release-local bytes without training, inference or PBM advancement."""
    manifest = json.loads((ROOT/'release_manifest.json').read_text(encoding='utf-8'))
    for row in manifest['files']:
        path = (ROOT/row['path']).resolve()
        if not path.is_relative_to(ROOT) or digest(path) != row['sha256']:
            raise ValueError('Release file missing or modified: '+row['path'])
    return manifest


def baseline_source_hashes():
    paths=[*sorted((ROOT/'physics').glob('*.py')),ROOT/'control/controller.py',
        ROOT/'control/objective.py',ROOT/'control/offline.py',ROOT/'release_runtime.py']
    return {path.relative_to(ROOT).as_posix():digest(path) for path in paths}


def write_baseline(folder, arrays, control, summary, status):
    """Export a newly optimized baseline with its numeric and source identity."""
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    if status not in ('search_completed','budget_stop_with_incumbent'):
        raise ValueError('Only a completed search or budget-limited feasible incumbent can be exported.')
    if not summary.get('valid') or not summary.get('feasible'):
        raise ValueError('A valid feasible baseline is required.')
    curve=np.asarray(arrays['temperature_c'],float)
    if curve.shape!=(181,) or np.asarray(arrays['quality']).shape!=(181,6):
        raise ValueError('A baseline needs the full 181-minute-state temperature and quality arrays.')
    validate_commands(curve,control)
    np.testing.assert_allclose(arrays['quality'][-1],summary['final_quality'],rtol=1e-12,atol=1e-12)
    np.savez_compressed(folder/'candidate.npz',**arrays)
    write_json(folder/'candidate_summary.json',summary)
    write_json(folder/'frozen_schedule.json',dict(nodes=np.column_stack((np.arange(181),curve)),
        optimized_on='nominal PBM only'))
    write_json(folder/'manifest.json',dict(format='release_nominal_b3_v1',status=status,
        control={key:copy.deepcopy(control[key]) for key in OBJECTIVE_KEYS},
        physical_config_sha256=digest(ROOT/'configs/pbm.yaml'),source_hashes=baseline_source_hashes(),
        files={name:digest(folder/name) for name in ('candidate.npz','candidate_summary.json','frozen_schedule.json')}))


def load_inputs(target='main', b3_run=None):
    from physics.config import load_config
    from physics.grid import make_log_grid
    filename = 'control_700.yaml' if target == 'target700' else 'control.yaml'
    control = yaml.safe_load((ROOT/'configs'/filename).read_text(encoding='utf-8'))
    config = load_config(ROOT/'configs/pbm.yaml')
    grid = make_log_grid(config)
    if b3_run is None:
        metadata=json.loads((ROOT/'assets'/f'b3_{target}.json').read_text(encoding='utf-8'))
        candidate=ROOT/'assets'/f'b3_{target}.npz'
    else:
        folder=Path(b3_run).resolve()
        manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('format')!='release_nominal_b3_v1' or manifest.get('status') not in ('search_completed','budget_stop_with_incumbent'):
            raise ValueError('The selected baseline is not a usable completed release B3 search.')
        if manifest.get('physical_config_sha256')!=digest(ROOT/'configs/pbm.yaml') or manifest.get('source_hashes')!=baseline_source_hashes():
            raise ValueError('Selected baseline physical configuration or numerical source differs.')
        expected={'candidate.npz','candidate_summary.json','frozen_schedule.json'}
        if set(manifest.get('files',{}))!=expected:
            raise ValueError('The selected baseline is missing required output identities.')
        for name,checksum in manifest['files'].items():
            if digest(folder/name)!=checksum:
                raise ValueError('Selected baseline file modified: '+name)
        summary=json.loads((folder/'candidate_summary.json').read_text(encoding='utf-8'))
        if not summary.get('valid') or not summary.get('feasible'):
            raise ValueError('The selected baseline has no valid feasible incumbent.')
        metadata=dict(control=manifest['control'],summary=summary)
        candidate=folder/'candidate.npz'
    if any(control[key] != metadata['control'][key] for key in OBJECTIVE_KEYS):
        raise ValueError('Frozen B3 and control objective/constraints differ. Re-optimize B3 for a changed objective.')
    with np.load(candidate, allow_pickle=False) as z:
        curve = z['temperature_c'].copy()
        np.testing.assert_allclose(z['quality'][-1],metadata['summary']['final_quality'],rtol=1e-12,atol=1e-12)
    if b3_run is not None:
        schedule=json.loads((folder/'frozen_schedule.json').read_text(encoding='utf-8'))
        if schedule.get('optimized_on')!='nominal PBM only':
            raise ValueError('The selected baseline was not optimized on the nominal PBM.')
        nodes=np.asarray(schedule['nodes'],float)
        if nodes.shape!=(181,2) or not np.array_equal(nodes[:,0],np.arange(181)):
            raise ValueError('The selected baseline schedule must span minutes 0..180.')
        np.testing.assert_allclose(nodes[:,1],curve,rtol=0,atol=1e-10)
    domain = json.loads((ROOT/'assets/quality_domain.json').read_text(encoding='utf-8'))
    lower, upper = np.asarray(domain['quality_min']), np.asarray(domain['quality_max'])
    if lower.shape != (6,) or upper.shape != (6,) or np.any(upper<lower):
        raise ValueError('The frozen six-dimensional scoring domain is invalid.')
    if curve.shape != (181,) or not np.isfinite(curve).all():
        raise ValueError('Frozen B3 must contain 181 finite minute temperatures.')
    validate_commands(curve,control)
    return config,grid,control,curve,(lower,upper),metadata


def load_predictor(device='cpu', observed=False, model_run=None):
    """Retain strict source identity for the explicitly exported checkpoint."""
    from surrogate.core.mpc.inference import MPCPredictor
    location=(ROOT/'assets/model') if model_run is None else Path(model_run).resolve()
    if location.resolve()==(ROOT/'assets/model').resolve():
        provenance=json.loads((ROOT/'assets/model/provenance.json').read_text(encoding='utf-8'))
        if digest(location/'best.pt')!=provenance['exported_checkpoint_sha256']:
            raise ValueError('The portable frozen checkpoint differs from its export provenance.')
        metadata=None
    else:
        metadata=json.loads((location/'run.json').read_text(encoding='utf-8'))
        finished={'control_training_smoke_complete_no_accuracy_claim',
            'control_oriented_training_complete_pending_independent_prediction_and_control',
            'continuation_smoke_complete_no_improvement_claim',
            'bounded_continuation_complete_pending_independent_prediction_and_control'}
        if (location/'RUNNING.lock').exists() or metadata.get('status') not in finished:
            raise ValueError('Select a complete training/continuation run, not a running or incomplete model.')
    cls = MPCPredictor
    if observed:
        class ObservedPredictor(MPCPredictor):
            def prepare_history(self, history, total):
                if set(history) != {'n','c','t'}:
                    raise ValueError('Observed history requires n, c and t.')
                n,c,t = [np.asarray(history[k],float) for k in ('n','c','t')]
                if n.ndim != 2 or n.shape[1]!=200 or not len(n) or c.shape!=(len(n),) or t.shape!=c.shape:
                    raise ValueError('Observed history requires nonempty K-by-200 counts and matching C/T.')
                if not all(np.isfinite(v).all() for v in (n,c,t)) or np.any(n<0) or np.any(c<0) or not np.isfinite(total) or total<=0:
                    raise ValueError('Invalid observations or fixed initial inventory; no clipping or repair applied.')
                pad=max(0,self.history-len(n))
                return {k:np.concatenate((np.repeat(v[:1],pad,axis=0),v),axis=0)[-self.history:].copy()
                        for k,v in (('n',n),('c',c),('t',t))}
        cls = ObservedPredictor
    predictor=cls.from_run(location,'best',device)
    if metadata is not None and metadata.get('fingerprint')!=predictor.saved.get('fingerprint'):
        raise ValueError('Model run metadata and best-checkpoint identities differ.')
    return predictor


def validate_commands(curve, control):
    values=np.asarray(curve,float)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Commands must be a nonempty finite temperature vector.')
    c=control['constraints']; rates=np.diff(values)
    if (np.any(values<c['min_temperature_c']-1e-7) or np.any(values>c['max_temperature_c']+1e-7)
        or np.any(rates<c['min_rate_c_min']-1e-7) or np.any(rates>c['max_rate_c_min']+1e-7)):
        raise ValueError('Command temperature/rate bounds violated.')


def quality(counts, concentration, initial_c, grid):
    from physics.metrics import volume_percentiles
    counts=np.asarray(counts,float); concentration=np.asarray(concentration,float)
    if counts.ndim!=2 or counts.shape[1]!=200 or concentration.shape!=(len(counts),) or not np.isfinite(initial_c) or initial_c<=0:
        raise ValueError('Quality requires H-by-200 counts, H concentrations and positive global C0.')
    result=np.full((len(counts),6),np.nan)
    for j,(n,c) in enumerate(zip(counts,concentration)):
        if not np.isfinite(n).all() or np.any(n<0) or not np.isfinite(c) or c<0 or float(n@grid.volumes_m3)<=0:
            continue
        result[j]=np.r_[c,(initial_c-c)/initial_c,volume_percentiles(n,grid)]
    return result


def optimize_baseline(target='main', max_evaluations=300):
    """Original six-free-node SLSQP design, using only the nominal PBM."""
    from scipy.optimize import minimize
    from control.offline import NodeProblem,PBMPredictor,seed_curves,command_margins
    if type(max_evaluations) is not int or max_evaluations<1:
        raise ValueError('max_evaluations must be a positive integer.')
    cfg,grid,control,_,domain,_=load_inputs(target)
    settings=yaml.safe_load((ROOT/'configs/offline.yaml').read_text(encoding='utf-8'))
    starts=seed_curves(control,cfg.batch.start_temperature_c)
    predictor=PBMPredictor(cfg,starts['linear'][:11])
    nodes=np.asarray(settings['node_minutes'],dtype=int)
    problem=NodeProblem(predictor,control,nodes,cfg.batch.start_temperature_c,*domain)
    folder=output_folder('offline'); started=time.perf_counter(); stopped=False
    class BudgetStop(Exception): pass
    def evaluate(x):
        if len(problem.trace)>=max_evaluations or time.perf_counter()-started>=settings['solver']['max_seconds']:
            raise BudgetStop('Offline search budget reached.')
        return problem.evaluate(x)
    for name in settings['starts']:
        print('Optimizing B3 start:',name,flush=True)
        try:
            minimize(lambda x:evaluate(x)['objective'],starts[name][nodes[1:]],method='SLSQP',
                bounds=[(control['constraints']['min_temperature_c'],control['constraints']['max_temperature_c'])]*(len(nodes)-1),
                constraints=[dict(type='ineq',fun=lambda x:np.r_[command_margins(problem.curve(x),control),evaluate(x)['margins']])],
                options=dict(maxiter=settings['solver']['max_iter_per_start'],eps=settings['solver']['finite_difference_c'],ftol=settings['solver']['ftol']))
        except BudgetStop:
            stopped=True; break
        finally:
            write_json(folder/'search_history.json',problem.trace)
    if problem.best is None:
        write_json(folder/'summary.json',dict(status='no_feasible_incumbent',evaluations=len(problem.trace)))
        raise RuntimeError('No feasible nominal B3 incumbent was found within the search budget.')
    curve=np.r_[starts['linear'][:11],problem.best['temperatures']]
    replay=PBMPredictor(cfg,starts['linear'][:11])
    replay(curve[11:])
    arrays=replay.last
    np.testing.assert_allclose(arrays['quality'][11:],problem.best['quality'],rtol=1e-10,atol=1e-9)
    summary=dict(valid=True,feasible=True,objective=problem.best['objective'],
        final_quality=arrays['quality'][-1].tolist(),minimum_margin=float(np.min(problem.best['margins'])),
        evaluations=len(problem.trace),validation_replay_evaluations=1,elapsed_seconds=time.perf_counter()-started,
        optimized_on='nominal PBM only',global_optimality_claim=False)
    write_baseline(folder,arrays,control,summary,'budget_stop_with_incumbent' if stopped else 'search_completed')
    write_json(folder/'summary.json',summary)
    print('Best-found offline schedule saved:',folder)
    return folder


def score(qualities, commands, control, domain):
    from control.objective import terminal_components
    components=terminal_components(qualities[-1],control)
    rates=np.diff(commands)[10:]; lower,upper=domain
    n,w=control['normalization'],control['weights']
    excursion=(np.maximum(lower-qualities[11:],0)+np.maximum(qualities[11:]-upper,0))/np.maximum(upper-lower,1e-12)
    components.update(move=float(w['move']*np.mean((rates/n['rate_c_min'])**2)),
        smooth=float(w['smooth']*np.mean((np.diff(np.r_[-40/180,rates])/n['rate_change_c_min'])**2)),
        domain=float(w['training_domain']*np.mean(excursion**2)))
    return dict(total=float(sum(components.values())),components=components)


def streams(case, seed, settings, minutes=180):
    """Original paper random streams; all policies share the same arrays."""
    from control.disturbances import noise_sequence, temperature_errors
    if case not in ('nominal','combined') or type(seed) is not int or seed<0:
        raise ValueError('Use nominal/combined and a nonnegative integer seed.')
    if case=='nominal':
        return np.zeros((minutes+1,2)),np.zeros(minutes+1),np.ones((minutes,2))
    limit=float(settings['temperature_limit_c'])
    if not np.isfinite(limit) or limit<=0:
        raise ValueError('A positive execution-error limit is required for combined disturbances.')
    sensors=noise_sequence(minutes+1,settings['measurement_temperature_std_c'],settings['measurement_concentration_relative_std'],seed)
    tracking=np.clip(temperature_errors(minutes+1,limit/2,settings['process_rho'],seed),-limit,limit)
    lower,upper=float(settings['minimum_kinetic_factor']),float(settings['maximum_kinetic_factor'])
    if not 0<lower<upper: raise ValueError('Kinetic bounds require 0 < lower < upper.')
    factors=np.random.default_rng(np.random.SeedSequence([seed,4545])).uniform(lower,upper,size=(minutes,2))
    return sensors,tracking,factors


def bridge(predictor, history, total, initial_c, grid):
    """Bind known observations; future plant states/errors never enter prediction."""
    known={k:np.array(v,float,copy=True) for k,v in history.items()}
    def predict(temperatures):
        forecast=predictor.rollout(known,np.asarray(temperatures,float),total)
        q=quality(forecast['counts'][1:],forecast['concentration'][1:],initial_c,grid)
        return dict(quality=q,diagnostics=dict(valid=forecast['valid_mask'][1:] & np.isfinite(q).all(-1)))
    return predict


def predict_example(device='cpu'):
    predictor=load_predictor(device)
    with np.load(ROOT/'assets/example_history.npz',allow_pickle=False) as z:
        history={k:z[k].copy() for k in ('n','c','t')}
        future=z['future_t'].copy(); total=float(z['initial_total'])
    result=predictor.rollout(history,future,total)
    folder=output_folder('prediction')
    np.savez_compressed(folder/'forecast.npz',counts=result['counts'],concentration=result['concentration'],valid_mask=result['valid_mask'],future_t=future)
    write_json(folder/'summary.json',dict(horizon=len(future),all_valid=bool(result['valid_mask'].all()),
        example_role='train',note='Bounded real-history interface example, not independent validation.'))
    print('Prediction saved:',folder)
    return folder


def simulate(minutes=2):
    from physics.simulation import BatchPlant
    if type(minutes) is not int or not 1<=minutes<=180: raise ValueError('minutes must be 1..180.')
    cfg,grid,control,*_=load_inputs()
    plant=BatchPlant.initialize(cfg)
    for temperature in np.linspace(60,20,181)[1:minutes+1]: plant.advance(float(temperature))
    r=plant.result(); folder=output_folder('pbm')
    np.savez_compressed(folder/'trajectory.npz',time_min=r.time_s/60,temperature_c=r.temperature_c,
        counts=r.counts_per_kg_water,concentration=r.concentration_kg_kg_water,
        quality=quality(r.counts_per_kg_water,r.concentration_kg_kg_water,plant.inventory.initial_concentration_kg_kg_water,grid),
        relative_mass_error=r.relative_mass_error)
    print('PBM trajectory saved:',folder)
    return folder


def compare(mode='smoke',case='nominal',target='main',device='auto',seed=None,model_run=None,b3_run=None):
    from physics.simulation import BatchPlant
    from control.controller import TrialProblem,solve_trial
    from control.control_helpers import shifted_seed,choose_action,cooling_profile
    from control.disturbances import perturbed_config,advance_disturbed
    cfg,grid,control,b3,domain,b3_meta=load_inputs(target,b3_run)
    control=copy.deepcopy(control)
    if mode not in ('smoke','run'): raise ValueError('Comparison mode must be smoke or run.')
    stop=12 if mode=='smoke' else 180
    if mode=='smoke': control['solver']['max_iter']=1
    settings=yaml.safe_load((ROOT/'configs/disturbance.yaml').read_text(encoding='utf-8'))
    seed=int(settings['seed']) if seed is None else seed
    sensor,tracking,factors=streams(case,seed,settings)
    predictor=load_predictor(device,observed=(case!='nominal'),model_run=model_run)
    np.testing.assert_allclose(predictor.model.pivots.detach().cpu().numpy(),grid.pivots_m,rtol=1e-12)
    coefficient=cfg.vessel.crystal_density_kg_m3*cfg.vessel.volume_shape_factor
    np.testing.assert_allclose(predictor.model.particle_mass.detach().cpu().numpy(),coefficient*grid.volumes_m3,rtol=1e-12)
    if model_run is not None:
        objective=predictor.saved['objective_snapshot']
        if objective.get('objective')!=control['objective']:
            raise ValueError('The selected model used a different objective family.')
        for section in ('constraints','normalization','weights'):
            for key,value in control[section].items():
                if objective[section].get(key)!=value:
                    raise ValueError('Model/control protocol mismatch: '+section+'.'+key)
        for key in ('yield_fraction','span_upper'):
            if objective['target'].get(key)!=control['target'][key]:
                raise ValueError('Model/control protocol mismatch: target.'+key)
    plants={name:BatchPlant.initialize(cfg) for name in LABELS}
    linear=np.linspace(60,20,181)
    curves=dict(linear=linear,empirical=cooling_profile(control['empirical_nodes'],60,control['constraints']),B3=b3)
    if any(not np.allclose(curve[:11],linear[:11],rtol=0,atol=1e-8) for curve in curves.values()):
        raise ValueError('Every policy must share the first ten linear-cooling minutes.')
    observations={name:dict(n=[],c=[],t=[]) for name in LABELS}
    commands={name:[60.] for name in LABELS}
    initial_c=plants['MPC'].inventory.initial_concentration_kg_kg_water
    total=float(plants['MPC'].current.concentration_kg_kg_water+plants['MPC'].current.crystal_mass_kg_kg_water)
    def observe(name,sample,index):
        c=float(sample.concentration_kg_kg_water*(1+sensor[index,1])); t=float(sample.temperature_c+sensor[index,0])
        if not np.isfinite([c,t]).all() or c<0: raise ValueError('Invalid observation; no clipping or state repair.')
        observations[name]['n'].append(sample.counts_per_kg_water.copy())
        observations[name]['c'].append(c); observations[name]['t'].append(t)
    for name,p in plants.items(): observe(name,p.current,0)
    folder=output_folder('comparison'); decisions=[]; previous=None; started=time.perf_counter()
    state=dict(status='running',mode=mode,case=case,target=target,seed=seed,stop_minute=stop,
        model_source='bundled_paper_model' if model_run is None else str(Path(model_run).resolve()),
        model_checkpoint_sha256=digest((ROOT/'assets/model' if model_run is None else Path(model_run))/'best.pt'),
        baseline_source='bundled_paper_baseline' if b3_run is None else str(Path(b3_run).resolve()),
        protocol='portable paper-core orchestration; original numerical kernels and random streams',
        note='Frozen LSTM/B3; exact CSD feedback, noisy T/C only in combined. No future errors/kinetics or solver-latency plant model.')
    write_json(folder/'manifest.json',state); write_json(folder/'control.json',control)
    np.savez_compressed(folder/'disturbance_sequence.npz',sensor=sensor,tracking=tracking,kinetic_factors=factors)
    def save():
        arrays=dict(time_min=plants['MPC'].result().time_s/60,pivots_um=grid.pivots_m*1e6)
        records=[]
        for name,p in plants.items():
            r=p.result(); q=quality(r.counts_per_kg_water,r.concentration_kg_kg_water,initial_c,grid)
            arrays.update({name+'_quality':q,name+'_counts':r.counts_per_kg_water,
                name+'_temperature_c':r.temperature_c,name+'_command_temperature_c':np.asarray(commands[name]),
                name+'_saturation':r.supersaturation_ratio,name+'_mass_error':r.relative_mass_error})
            for j,row in enumerate(q):
                records.append(dict(policy=name,minute=j,command_temperature_c=commands[name][j],temperature_c=r.temperature_c[j],
                    concentration=row[0],yield_fraction=row[1],D10_um=row[2],D50_um=row[3],D90_um=row[4],Span=row[5],
                    supersaturation=r.supersaturation_ratio[j],relative_mass_error=r.relative_mass_error[j]))
        np.savez_compressed(folder/'trajectories.npz',**arrays)
        write_csv(folder/'trajectories.csv',records); write_json(folder/'solver_history.json',decisions)
        return arrays
    try:
        save()
        for minute in range(stop):
            command=float(linear[minute+1])
            if minute>=10:
                known={k:np.asarray(v) for k,v in observations['MPC'].items()}
                anchor=commands['MPC'][-1]
                problem=TrialProblem(bridge(predictor,known,total,initial_c,grid),anchor,
                    anchor-commands['MPC'][-2],180-minute,control,*domain)
                seed_rates=shifted_seed(previous,problem.matrix,-40/180)
                start=time.perf_counter()
                answer=solve_trial(problem,seed_rates,budget=control['solver']['max_seconds'],
                    maxiter=control['solver']['max_iter'],fd_step=control['solver']['finite_difference_step'])
                elapsed=time.perf_counter()-start
                candidate=answer['candidate']; rate,action=choose_action(candidate,anchor,control['constraints'],elapsed)
                command=float(anchor+rate)
                previous=problem.matrix@candidate['rates'] if action=='feasible_incumbent' else None
                record={k:v for k,v in answer.items() if k not in ('baseline','candidate')}
                record.update(minute=minute,horizon_steps=180-minute,decision_seconds=elapsed,action=action,
                    command_temperature_c=command,rate_c_min=rate,
                    candidate_objective=None if candidate is None else candidate['objective'])
                decisions.append(record)
                if candidate is not None:
                    (folder/'decisions').mkdir(exist_ok=True)
                    np.savez_compressed(folder/'decisions'/f'{minute:03d}.npz',temperatures=candidate['temperatures'],
                        quality=candidate['quality'],rates=candidate['rates'],applied=np.asarray(action=='feasible_incumbent'))
            validate_commands(np.r_[commands['MPC'],command],control)
            for name,plant in plants.items():
                issued=command if name=='MPC' else float(curves[name][minute+1])
                try:
                    plant.cfg=perturbed_config(cfg,float(factors[minute,0]),float(factors[minute,1]))
                    sample=advance_disturbed(plant,issued,float(tracking[minute+1]))
                finally:
                    plant.cfg=cfg
                commands[name].append(issued); observe(name,sample,minute+1)
            save()
            if minute>=10:
                print(f'Minute {minute+1}/{stop}: command {command:.3f} C; decision {elapsed:.2f}s; elapsed {(time.perf_counter()-started)/60:.1f} min.',flush=True)
        arrays=save(); rows=[]
        for name in LABELS:
            q=arrays[name+'_quality']; row=dict(policy=name,observed_minute=stop,completed_full_batch=(stop==180),
                D50_um=float(q[-1,3]),Span=float(q[-1,5]),yield_fraction=float(q[-1,1]),
                D50_error_um=float(q[-1,3]-control['target']['d50_um']),
                maximum_relative_mass_error=float(np.max(np.abs(arrays[name+'_mass_error']))))
            if stop==180: row['objective']=score(q,np.asarray(commands[name]),control,domain)['total']
            rows.append(row)
        if stop==180 and case=='nominal':
            np.testing.assert_allclose(arrays['B3_quality'][-1],b3_meta['summary']['final_quality'],rtol=1e-9,atol=1e-8)
            np.testing.assert_allclose(next(r['objective'] for r in rows if r['policy']=='B3'),b3_meta['summary']['objective'],rtol=2e-6,atol=1e-5)
        if digest((ROOT/'assets/model' if model_run is None else Path(model_run))/'best.pt')!=state['model_checkpoint_sha256']:
            raise ValueError('The selected model checkpoint changed during the comparison.')
        write_csv(folder/'comparison.csv',rows); write_csv(folder/'solver_history.csv',decisions)
        write_json(folder/'summary.json',dict(completed_full_batch=(stop==180),comparison=rows,
            decision_count=len(decisions),accepted_mpc_actions=sum(r['action']=='feasible_incumbent' for r in decisions),
            hold_fallback_count=sum(r['action']=='hold_fallback' for r in decisions),
            elapsed_seconds=time.perf_counter()-started,note='All products are PBM truth. Smoke stops at 12 minutes and makes no final-product claim.'))
        state['status']='completed'
    except BaseException as error:
        state.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        state['elapsed_seconds']=time.perf_counter()-started
        write_json(folder/'manifest.json',state)
        print('Comparison output:',folder,flush=True)
    return folder
