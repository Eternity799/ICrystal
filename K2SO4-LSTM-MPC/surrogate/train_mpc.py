"""Train control-oriented full-CSD surrogate models."""
import argparse
from pathlib import Path

from core.mpc.config import DEFAULT_CONFIG, load_config
from core.mpc.training import run_suite


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    p.add_argument('--mode',choices=('smoke','pilot','formal'),default='smoke')
    p.add_argument('--arm',choices=('all','replay_csd','control_priority'),default='all')
    p.add_argument('--dataset',type=Path,help='Existing complete dataset directory; no data regeneration')
    p.add_argument('--resume',type=Path,help='One arm directory, not the two-arm suite directory')
    p.add_argument('--no-plot',action='store_true')
    a=p.parse_args();config=load_config(a.config,a.mode,a.dataset,a.arm)
    print('Allow roughly 5-15 min for smoke, 0.5-2 h for two pilot arms, and 6-24 h for formal; re-estimate from local timings.',flush=True)
    print('Reads saved train/validation labels only; no PBM, test states or control optimization.',flush=True)
    run_suite(config,a.config.resolve(),a.resume,a.no_plot)


if __name__=='__main__':main()
