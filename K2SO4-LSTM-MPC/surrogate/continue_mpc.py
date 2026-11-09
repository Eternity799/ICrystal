"""Continue training from a completed replay_csd checkpoint."""
import argparse
from pathlib import Path

from continuation.config import DEFAULT_CONFIG, load_settings
from continuation.training import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--mode', choices=('smoke', 'formal'), default='smoke')
    parser.add_argument('--resume', type=Path, help='This continuation replay_csd directory with last.pt, not the parent model directory')
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    settings = load_settings(args.config, args.mode)
    plan = settings['plan']
    print(f'Continuation plan: {plan["epochs"]} epochs x {plan["updates_per_epoch"]} updates = {plan["epochs"] * plan["updates_per_epoch"]} optimizer updates; '
          'Trains ordinary and full-terminal responses together without a new loss.', flush=True)
    print('Allow roughly 3-10 min for smoke and 30-90 min for formal including validation; re-estimate the unmeasured mixture from actual logs.', flush=True)
    print('Reads saved train/validation labels; no PBM, test-state reads or closed loop.', flush=True)
    run(settings, args.config.resolve(), args.resume, args.no_plot, args.show)


if __name__ == '__main__':
    main()
