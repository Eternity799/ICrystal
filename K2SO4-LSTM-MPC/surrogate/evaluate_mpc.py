"""Evaluate saved control-oriented surrogate models."""
import argparse
import json
from pathlib import Path

from core.mpc.config import DEFAULT_CONFIG,load_config
from core.mpc.evaluation import evaluate_run


def resolve_runs(run,latest,mode,arm,config,checkpoint='best'):
    if latest:
        root=Path(config['output_root'])/mode
        suites=sorted((p for p in root.glob('*') if (p/'suite.json').is_file()),reverse=True)
        if not suites:raise ValueError('No completed suite.json for this mode; use --run for a running/interrupted arm.')
        run=suites[0]
    run=Path(run).resolve()
    if (run/'suite.json').is_file():
        meta=json.loads((run/'suite.json').read_text(encoding='utf-8'))
        runs=[run/Path(p).name for p in meta['arms']]
    else:runs=[run]
    runs=[p for p in runs if arm=='all' or p.name==arm]
    if not runs or any(not (p/f'{checkpoint}.pt').is_file() for p in runs):raise ValueError(f"A single arm with saved {checkpoint}.pt, or a completed two-arm directory, is required.")
    return runs


def main():
    p=argparse.ArgumentParser(description=__doc__);source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--run',type=Path);source.add_argument('--latest',action='store_true')
    p.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    p.add_argument('--mode',choices=('smoke','pilot','formal'),default='pilot')
    p.add_argument('--arm',choices=('all','replay_csd','control_priority'),default='all')
    p.add_argument('--role',choices=('validation','test'),default='validation')
    p.add_argument('--checkpoint',choices=('best','last'),default='best')
    p.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    p.add_argument('--per-family',type=int,help='Parents per family; 0 means all. Validation defaults to the validation parents selected during training.')
    p.add_argument('--show',action='store_true');p.add_argument('--no-plot',action='store_true')
    a=p.parse_args()
    
    config=load_config(a.config,'smoke')
    for run in resolve_runs(a.run,a.latest,a.mode,a.arm,config,a.checkpoint):
        evaluate_run(run,a.role,a.checkpoint,a.device,a.per_family,a.show,a.no_plot)


if __name__=='__main__':main()
