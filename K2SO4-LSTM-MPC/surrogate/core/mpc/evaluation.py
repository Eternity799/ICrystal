"""Single-step, recursive and action-response evaluation of saved models."""
import json
import math
import time
from datetime import datetime,timezone
from pathlib import Path

import numpy as np

from ..data import GRID_KEYS,read_manifest
from ..evaluation import cqa, metrics, summarize
from ..io import save_npz,sha256,write_csv,write_json
from .data import load_selected_dataset,select_parent_ids
from .inference import MPCPredictor
from .ranking import compare_candidates,trajectory_score
from .training import ranking_selection_penalty


def row_metrics(out,truth_n,truth_c,row):
    result=metrics(out,truth_n,truth_c,row)
    if result['valid']:
        result['yield_error_pp']=-100*float(out['concentration']-truth_c)/float(row['concentration'][0])
    return result


def response_statistics(records):
    result=[]
    for variable,horizon in sorted({(r['variable'],r['horizon']) for r in records}):
        selected=[r for r in records if r['variable']==variable and r['horizon']==horizon]
        valid=[r for r in selected if r['valid']]
        parents=sorted({r['parent_id'] for r in valid})
        means=[float(np.mean([abs(r['delta_error']) for r in valid if r['parent_id']==p])) for p in parents]
        eligible=[r for r in valid if r['direction_eligible']]
        parent_sign=[float(np.mean([bool(r['direction_agreement']) for r in eligible if r['parent_id']==p]))
                     for p in parents if any(r['parent_id']==p for r in eligible)]
        result.append(dict(variable=variable,horizon=horizon,probes=len(selected),parents=len(parents),invalid_count=len(selected)-len(valid),
                           delta_mae_parent_mean=float(np.mean(means)) if means else None,
                           delta_mae_parent_p95=float(np.quantile(means,.95)) if means else None,
                           direction_eligible_count=len(eligible),
                           direction_agreement_parent_mean=float(np.mean(parent_sign)) if parent_sign else None))
    return result


def quality_summary(records):
    result=[]
    for family in ['all']+sorted({r['family'] for r in records}):
        selected=records if family=='all' else [r for r in records if r['family']==family]
        for summary in summarize(selected):
            terminal_rows=[r for r in selected if r['kind']==summary['kind'] and r['origin_min']==summary['origin_min'] and r['endpoint_min']==180]
            group=[r for r in selected if r['kind']==summary['kind'] and r['origin_min']==summary['origin_min'] and r['valid']]
            parents=sorted({r['parent_id'] for r in group})
            errors=[float(np.mean([abs(r['yield_error_pp']) for r in group if r['parent_id']==p])) for p in parents]
            endpoints=[r for r in group if r['endpoint_min']==180]
            summary.update(family=family,yield_error_pp_parent_mean=float(np.mean(errors)) if errors else None,
                           terminal_total_count=len(terminal_rows),terminal_valid_count=sum(r['valid'] for r in terminal_rows),
                           terminal_invalid_count=sum(not r['valid'] for r in terminal_rows),
                           yield_error_pp_parent_p95=float(np.quantile(errors,.95)) if errors else None,
                           terminal_d50_mae_um=float(np.mean([abs(r['d50_error_um']) for r in endpoints])) if endpoints else None,
                           terminal_span_mae=float(np.mean([abs(r['span_error']) for r in endpoints])) if endpoints else None,
                           terminal_yield_mae_pp=float(np.mean([abs(r['yield_error_pp']) for r in endpoints])) if endpoints else None)
            result.append(summary)
    return result


def evaluate_run(run,role='validation',checkpoint='best',device='auto',per_family=None,show=False,no_plot=False):
    if role not in ('validation','test'):raise ValueError('Evaluation role must be validation or test.')
    run=Path(run).resolve();predictor=MPCPredictor.from_run(run,checkpoint,device);saved=predictor.saved
    root=Path(saved['config']['dataset']);manifest=read_manifest(root)
    if sha256(root/'manifest.json')!=saved['fingerprint']['dataset_manifest']:
        raise ValueError('Dataset manifest differs from the training snapshot.')
    selection=json.loads((run/'selected_parents.json').read_text(encoding='utf-8'))
    if per_family is None and role=='validation':
        ids=selection['validation']
    else:
        ids=select_parent_ids(manifest,role,0 if per_family is None else per_family,saved['config']['seed']+17)
    rows=load_selected_dataset(root,(role,),{role:ids},manifest)[role]
    if any(not np.array_equal(row[key],np.asarray(saved['grid'][key])) for row in rows for key in GRID_KEYS):
        raise ValueError('Evaluation grid or material properties differ from the training checkpoint.')
    out=run/'evaluation_mpc'/role/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f_utc')
    out.mkdir(parents=True,exist_ok=False);records=[];responses=[];rankings=[];candidates=[];started=time.perf_counter()
    objective=saved['objective_snapshot'];domain=saved['quality_domain'];config=saved['config']['evaluation']
    def append_record(parent,kind,origin,endpoint,predicted):
        record=dict(parent_id=parent['id'],role=role,family=parent['family'],kind=kind,origin_min=origin,endpoint_min=endpoint)
        record.update(row_metrics(predicted,parent['counts'][endpoint],parent['concentration'][endpoint],parent));records.append(record)
    for index,row in enumerate(rows,1):
        total=float(row['initial_total_solute']);pm=row['particle_mass_kg'];edges=row['size_edges_m'];c0=float(row['concentration'][0])
        one_n=np.full_like(row['counts'],np.nan);one_c=np.full_like(row['concentration'],np.nan)
        one_n[0]=row['counts'][0];one_c[0]=row['concentration'][0]
        for minute in range(1,181):
            history=dict(n=row['counts'][:minute],c=row['concentration'][:minute],t=row['temperature'][:minute])
            try:
                prediction=predictor.step(history,float(row['temperature'][minute]),total)
                one_n[minute]=prediction['counts'];one_c[minute]=prediction['concentration']
            except (ValueError,FloatingPointError) as exc:prediction=dict(invalid_reason=str(exc))
            append_record(row,'one_step',-1,minute,prediction)
        save_npz(out/(row['id']+'_one_step.npz'),counts=one_n,concentration=one_c,time_s=row['time_s'],truth_counts=row['counts'],truth_concentration=row['concentration'])
        for origin in config['origins']:
            history=dict(n=row['counts'][:origin+1],c=row['concentration'][:origin+1],t=row['temperature'][:origin+1])
            forecast=predictor.rollout(history,row['temperature'][origin+1:],total)
            for j,prediction in enumerate(forecast['details'],origin+1):append_record(row,'recursive',origin,j,prediction)
            save_npz(out/f'{row["id"]}_recursive_{origin}.npz',counts=forecast['counts'],concentration=forecast['concentration'],
                     valid_mask=forecast['valid_mask'],time_s=row['time_s'][origin:],truth_counts=row['counts'][origin:],truth_concentration=row['concentration'][origin:])
        for branch in row['groups']:
            origin=branch['origin'];h=branch['horizon'];forecasts=[]
            history=dict(n=row['counts'][:origin+1],c=row['concentration'][:origin+1],t=row['temperature'][:origin+1])
            for temperatures in branch['temperature']:
                forecasts.append(predictor.rollout(history,temperatures[1:],total))
            save_npz(out/(Path(branch['file']).stem+'_prediction.npz'),
                     counts=np.stack([f['counts'] for f in forecasts]),concentration=np.stack([f['concentration'] for f in forecasts]),
                     valid_mask=np.stack([f['valid_mask'] for f in forecasts]),truth_counts=branch['counts'],
                     truth_concentration=branch['concentration'],temperature=branch['temperature'],offsets_c=branch['offsets_c'])
            for horizon in config['response_horizons']:
                if horizon>h or len(forecasts)<2:continue
                for j,offset in enumerate(branch['offsets_c'][1:],1):
                    valid=bool(forecasts[0]['valid_mask'][horizon] and forecasts[j]['valid_mask'][horizon])
                    predicted_values=[];truth_values=[]
                    for f in (forecasts[0],forecasts[j]):
                        q=cqa(f['counts'][horizon],pm,edges)
                        predicted_values.append(dict(d50_um=q[1],span=q[3],concentration=f['concentration'][horizon],yield_fraction=(c0-f['concentration'][horizon])/c0))
                    for k in (0,j):
                        q=cqa(branch['counts'][k,horizon],pm,edges);c=branch['concentration'][k,horizon]
                        truth_values.append(dict(d50_um=q[1],span=q[3],concentration=c,yield_fraction=(c0-c)/c0))
                    for key,tolerance in config['direction_tolerance'].items():
                        delta_true=float(truth_values[1][key]-truth_values[0][key])
                        delta_pred=float(predicted_values[1][key]-predicted_values[0][key])
                        finite=valid and math.isfinite(delta_true) and math.isfinite(delta_pred)
                        eligible=finite and abs(delta_true)>tolerance
                        responses.append(dict(parent_id=row['id'],family=row['family'],role=role,group_file=branch['file'],origin_min=origin,horizon=horizon,
                            offset_c=float(offset),variable=key,valid=finite,truth_delta=delta_true,predicted_delta=delta_pred if finite else None,
                            delta_error=delta_pred-delta_true if finite else None,direction_eligible=eligible,
                            direction_agreement=bool(np.sign(delta_true)==np.sign(delta_pred)) if eligible else None))
            if branch.get('terminal',False) and len(forecasts)>1:
                prediction_scores=[];truth_scores=[]
                previous=float(row['temperature'][origin]-row['temperature'][max(0,origin-1)])
                kwargs=dict(initial_concentration=c0,particle_mass=pm,edges=edges,objective=objective,previous_rate=previous,quality_domain=domain,
                            metadata=dict(terminal=True,origin=origin,horizon=h))
                for j,forecast in enumerate(forecasts):
                    ps=trajectory_score(forecast['counts'],forecast['concentration'],branch['temperature'][j],**kwargs)
                    ts=trajectory_score(branch['counts'][j],branch['concentration'][j],branch['temperature'][j],**kwargs)
                    prediction_scores.append(ps);truth_scores.append(ts)
                    candidates.append(dict(parent_id=row['id'],family=row['family'],role=role,group_file=branch['file'],origin_min=origin,horizon=h,
                         candidate=j,offset_c=float(branch['offsets_c'][j]),predicted_valid=ps['valid'],predicted_feasible=ps['feasible'],predicted_objective=ps['objective'],
                         truth_valid=ts['valid'],truth_feasible=ts['feasible'],truth_objective=ts['objective'],predicted_reason=ps['reason'],truth_reason=ts['reason']))
                rank=compare_candidates(prediction_scores,truth_scores,saved['config']['selection']['tie_tolerance'])
                rankings.append(dict(parent_id=row['id'],family=row['family'],role=role,group_file=branch['file'],origin_min=origin,horizon=h,
                                     selection_penalty=ranking_selection_penalty(rank),**rank))
        elapsed=time.perf_counter()-started
        print(f"Checked {index}/{len(rows)}; elapsed {elapsed/60:.1f} min; estimated remaining {elapsed/index*(len(rows)-index)/60:.1f} min from completed trajectories, excluding plots.",flush=True)
    write_csv(out/'per_minute.csv',records);write_csv(out/'response.csv',responses)
    write_csv(out/'terminal_ranking.csv',rankings);write_csv(out/'candidate_scores.csv',candidates)
    groups=quality_summary(records);response_summary=response_statistics(responses)
    write_csv(out/'summary.csv',groups);write_csv(out/'response_summary.csv',response_summary)
    informative=[r for r in rankings if r['informative']]
    summary=dict(role=role,parents=len(rows),parent_ids=ids,checkpoint=checkpoint,checkpoint_sha256=sha256(run/f'{checkpoint}.pt'),
        dataset_manifest_sha256=saved['fingerprint']['dataset_manifest'],groups=groups,response=response_summary,
        terminal_groups=len(rankings),informative_terminal_groups=len(informative),
        truth_informative_terminal_groups=sum(r['selection_penalty'] is not None for r in rankings),
        predicted_tie_groups=sum(r['predicted_tie'] for r in rankings),
        no_predicted_feasible_groups=sum(r['predicted_feasible_count']==0 for r in rankings),
        selected_truth_infeasible_groups=sum(r['selected_truth_infeasible'] for r in rankings),
        ranking_mismatch_count=sum(bool(r['mismatch']) for r in informative),
        false_feasible_count=sum(r['false_feasible_count'] for r in rankings),
        false_rejected_feasible_count=sum(r['false_rejected_feasible_count'] for r in rankings),
        note='Frozen nominal surrogate, known actual temperatures and saved PBM labels. No future CSD/C feedback, new PBM or control optimization. Finite nominal terminal candidates do not establish closed-loop qualification or global optimality.')
    write_json(out/'summary.json',summary)
    (out/'report.md').write_text('# Control-oriented prediction evaluation\n\n'+f"Role: {role}; base trajectories: {len(rows)}; terminal candidate groups: {len(rankings)}; informative groups: {len(informative)}.\n\n"
        'Inspect one-step and terminal rollout results in summary.csv, temperature responses in response_summary.csv, and candidate decisions in terminal_ranking.csv and candidate_scores.csv.\n\n'
        f'Ranking mismatches in informative groups: {summary["ranking_mismatch_count"]}; predicted tie groups: {summary["predicted_tie_groups"]}; groups without a predicted feasible candidate: {summary["no_predicted_feasible_groups"]}.\n\n'
        f'False accepted candidates: {summary["false_feasible_count"]}; falsely rejected feasible candidates: {summary["false_rejected_feasible_count"]}; selected candidates infeasible under PBM labels: {summary["selected_truth_infeasible_groups"]}.\n\n'
        'Near ties, groups without feasible candidates and invalid predictions are listed separately. Unplotted errors are not absent errors. Candidates within each group are correlated.\n\n'+summary['note']+'\n',encoding='utf-8')
    if not no_plot:
        from .reporting import plot_evaluation
        plot_evaluation(out,records,responses,rankings,show)
    print('Report, figures and CSV: '+str(out),flush=True)
    return out
