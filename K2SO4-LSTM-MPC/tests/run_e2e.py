"""Real, reduced-budget end-to-end interface exercise, with no mocked stages.

All nine PBM base trajectories retain the full 180-minute physical duration.
Formal *execution paths* use explicit tiny fixture counts and training budgets;
this is never a formal accuracy or robustness experiment.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import release_runtime as runtime


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',type=Path,help='Continue this audit after a failed interface stage; completed stages are retained')
    args=parser.parse_args()
    runtime.verify_release()
    work=runtime.output_folder('e2e') if args.resume is None else args.resume.resolve()
    configs=work/'configs'; configs.mkdir(exist_ok=True)
    initial=dict(status='running',purpose='real reduced-budget interface chain; no scientific performance conclusion',
        work=str(work),started=time.time(),stages=[],artifacts={},budget=dict(parents=9,
            parent_minutes=180,training_updates=2,continuation_updates=1,b3_search_evaluations=1,
            b3_verification_replays=1,control_stop_minute=12,mpc_forecast_endpoint_minute=180))
    state=initial if args.resume is None else json.loads((work/'e2e_summary.json').read_text(encoding='utf-8'))
    state['status']='running'; state.pop('error',None)
    def save(): runtime.write_json(work/'e2e_summary.json',state)
    def yaml_file(name,value):
        path=configs/name
        path.write_text(yaml.safe_dump(value,sort_keys=False),encoding='utf-8')
        return str(path)
    def call(name,args):
        if any(row['stage']==name and row['status']=='passed' for row in state['stages']):
            print('RETAINED:',name,flush=True)
            return
        args=list(map(str,args))
        print('START:',name,flush=True)
        began=time.perf_counter(); record=dict(stage=name,arguments=args,status='running',log=name+'.log')
        state['stages'].append(record); save()
        with (work/(name+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'run.py'),*map(str,args)],cwd=ROOT,
                env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLBACKEND='Agg'),
                stdout=log,stderr=subprocess.STDOUT)
        record.update(exit_code=result.returncode,seconds=time.perf_counter()-began,
                      status='passed' if result.returncode==0 else 'failed')
        save()
        if result.returncode:
            print((work/(name+'.log')).read_text(encoding='utf-8')[-12000:],flush=True)
            raise RuntimeError('End-to-end stage failed: '+name)
        print('PASS:',name,f'({record["seconds"]:.1f}s)',flush=True)
    def newest(path,pattern):
        choices=sorted(path.glob(pattern))
        if not choices: raise RuntimeError('Expected output not found: '+str(path/pattern))
        return choices[-1]
    def logged_directory(stage,prefix):
        records=[row for row in state['stages'] if row['stage']==stage and row['status']=='passed']
        log=work/records[-1].get('log',stage+'.log')
        matches=[line.split(prefix,1)[1].strip() for line in log.read_text(encoding='utf-8').splitlines() if prefix in line]
        if not matches: raise RuntimeError('Output directory not recorded for stage '+stage)
        path=Path(matches[-1]).resolve()
        if not path.is_relative_to(ROOT/'outputs') or not path.is_dir():
            raise ValueError('Unexpected audit output directory for '+stage)
        return path
    def record_comparison(stage,key):
        folder=logged_directory(stage,'Comparison output: ')
        summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        assert summary['decision_count']==2 and not summary['completed_full_batch']
        assert {row['policy'] for row in summary['comparison']}==set(runtime.LABELS)
        with runtime.np.load(folder/'trajectories.npz',allow_pickle=False) as z:
            assert len(z['time_min'])==13 and z['time_min'][-1]==12
            for policy in runtime.LABELS:
                assert z[policy+'_counts'].shape==(13,200)
                assert runtime.np.isfinite(z[policy+'_quality']).all()
        state['artifacts'][key]=str(folder)
        state[key+'_actions']=dict(decisions=summary['decision_count'],accepted=summary['accepted_mpc_actions'],
            fallbacks=summary['hold_fallback_count'])
        save()
        return summary
    try:
        g=yaml.safe_load((ROOT/'data_generation/dataset.yaml').read_text(encoding='utf-8'))
        g.update(pbm_source=str(ROOT/'physics'),pbm_config=str(ROOT/'configs/pbm.yaml'),output_root=str(work/'datasets'))
        g['counts']['formal']=dict(train=1,validation=1,test=1)
        g['branches'].update(origins_min=[10],strata_min=[0,180],origins_per_stratum=1,
            horizon_steps=3,offsets_c=[0.,-.05,.05],patterns=['offset'],durations_steps=[1],
            long_horizon_steps=3,long_groups_per_parent=0,terminal_every=1,terminal_origins_min=[170])
        g['quality']['numerical']['enabled']=False
        call('01_generate',['generate','--config',yaml_file('generate.yaml',g),'--mode','formal','--limit','0'])
        dataset=newest(work/'datasets/formal','*')
        generated=json.loads((dataset/'manifest.json').read_text(encoding='utf-8'))
        assert len(generated['completed'])==9 and generated['status']=='dataset_complete_not_trained'
        state['artifacts']['dataset']=str(dataset); save()

        base=yaml.safe_load((ROOT/'surrogate/config.yaml').read_text(encoding='utf-8'))
        base.update(dataset=str(dataset),output_root=str(work/'training'),device='cpu')
        base['sampling']['batch_size']=2
        config=yaml.safe_load((ROOT/'surrogate/mpc_training.yaml').read_text(encoding='utf-8'))
        config.update(base_config=yaml_file('base.yaml',base),dataset=str(dataset),output_root=str(work/'training'),
            objective_config=str(ROOT/'configs/control.yaml'),device='cpu',arms=['replay_csd'])
        config['profiles']['formal'].update(train_per_family=0,validation_per_family=0,development_per_family=1,patience=0,
            stages=[dict(horizon=1,epochs=1,updates_per_epoch=1,lr=.001,full_terminal=False),
                    dict(horizon=3,epochs=1,updates_per_epoch=1,lr=.0005,full_terminal=True)])
        config['selection'].update(horizons=[1,3],origins=[10,170],every_epochs=1,full_every_epochs=1,response_groups_per_parent=1)
        config['evaluation'].update(origins=[10],response_horizons=[1,3,10])
        training_config=yaml_file('training.yaml',config)
        call('02_train',['train','--config',training_config,'--mode','formal','--arm','replay_csd','--dataset',dataset,'--no-plot'])
        trained=newest(work/'training/formal','*/replay_csd')
        state['artifacts']['trained_model']=str(trained); save()
        call('03_validate',['evaluate','--config',training_config,'--run',trained,'--role','validation','--device','cpu','--no-plot'])

        continuation=yaml.safe_load((ROOT/'surrogate/mpc_continuation.yaml').read_text(encoding='utf-8'))
        continuation.update(parent_run=str(trained),output_root=str(work/'continued'),device='cpu')
        continuation['profiles']['formal'].update(epochs=1,updates_per_epoch=1,horizon=3,batch_size=2,
            every_epochs=1,full_every_epochs=1)
        call('04_continue',['continue-training','--config',yaml_file('continuation.yaml',continuation),'--mode','formal','--no-plot'])
        continued=newest(work/'continued/formal','*/replay_csd')
        state['artifacts']['continued_model']=str(continued); save()
        call('05_test',['evaluate','--config',training_config,'--run',continued,'--role','test','--per-family','1','--device','cpu','--no-plot'])

        call('06_offline',['optimize-baseline','--target','main','--max-evaluations','1'])
        baseline=logged_directory('06_offline','Best-found offline schedule saved: ')
        runtime.load_inputs('main',b3_run=baseline)
        state['artifacts']['new_baseline']=str(baseline); save()
        call('07_bundled_nominal',['compare','--mode','smoke','--case','nominal','--device','cpu'])
        bundled_summary=record_comparison('07_bundled_nominal','bundled_nominal')
        if not bundled_summary['accepted_mpc_actions']:
            raise RuntimeError('Bundled nominal comparison produced only fallback actions; optimized control not exercised.')
        save()
        call('08_new_model_nominal',['compare','--mode','smoke','--case','nominal','--device','cpu',
            '--model-run',continued,'--b3-run',baseline])
        record_comparison('08_new_model_nominal','new_model_nominal')
        call('09_new_model_combined',['compare','--mode','smoke','--case','combined','--device','cpu',
            '--model-run',continued,'--b3-run',baseline])
        record_comparison('09_new_model_combined','new_model_combined')
        state['status']='passed_reduced_budget_interface_chain'
    except BaseException as error:
        state.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        state['elapsed_seconds']=time.time()-state['started']; save()
        print('End-to-end audit:',work,flush=True)


if __name__=='__main__': main()
