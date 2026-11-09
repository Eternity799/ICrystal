"""Command-line entry point for the K2SO4 crystallization study."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    scripts={'generate':ROOT/'data_generation/generate_dataset.py',
        'train':ROOT/'surrogate/train_mpc.py','evaluate':ROOT/'surrogate/evaluate_mpc.py',
        'continue-training':ROOT/'surrogate/continue_mpc.py'}
    if argv and argv[0] in scripts:
        return subprocess.run([sys.executable,'-X','utf8',str(scripts[argv[0]]),*argv[1:]],check=True).returncode
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('check',help='Verify release-local source and frozen assets; no prediction, training or PBM')
    p=commands.add_parser('predict',help='Three-step prediction on one real saved training history')
    p.add_argument('--device',choices=('cpu','cuda','auto'),default='cpu')
    p=commands.add_parser('simulate',help='PBM linear-cooling example')
    p.add_argument('--minutes',type=int,default=2)
    p=commands.add_parser('compare',help='Frozen LSTM-MPC, B3, linear and slow-fast control')
    p.add_argument('--mode',choices=('smoke','run'),default='smoke')
    p.add_argument('--case',choices=('nominal','combined'),default='nominal')
    p.add_argument('--target',choices=('main','target700'),default='main')
    p.add_argument('--device',choices=('cpu','cuda','auto'),default='auto')
    p.add_argument('--seed',type=int)
    p.add_argument('--model-run',type=Path,help='A compatible completed training/continuation arm; default: bundled final paper weights')
    p.add_argument('--b3-run',type=Path,help='A release optimize-baseline output directory; default: bundled frozen paper baseline')
    p=commands.add_parser('optimize-baseline',help='Re-run the nominal six-free-node PBM search (expensive)')
    p.add_argument('--target',choices=('main','target700'),default='main')
    p.add_argument('--max-evaluations',type=int,default=300)
    for name in scripts:
        commands.add_parser(name,help='Use '+name+' --help for the dedicated English interface')
    args=parser.parse_args(argv)
    import release_runtime as runtime
    manifest=runtime.verify_release()
    if args.command=='check':
        runtime.load_inputs('main'); runtime.load_inputs('target700')
        print(f'CHECK PASSED: {len(manifest["files"])} release files; two frozen B3 targets; no inference/PBM/training.')
        return 0
    if args.command=='predict': runtime.predict_example(args.device)
    elif args.command=='simulate': runtime.simulate(args.minutes)
    elif args.command=='compare': runtime.compare(args.mode,args.case,args.target,args.device,args.seed,args.model_run,args.b3_run)
    elif args.command=='optimize-baseline': runtime.optimize_baseline(args.target,args.max_evaluations)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
