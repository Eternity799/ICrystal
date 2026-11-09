"""Evaluate saved full-CSD surrogate models with fixed data roles."""
import argparse
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    entry = p.add_mutually_exclusive_group(required=True)
    entry.add_argument('--run', type=Path, help='A complete new train.py run directory for fresh predictions')
    entry.add_argument('--plot-only', type=Path, help='Existing evaluation directory; redraw figures without rerunning inference')
    p.add_argument('--role', choices=['validation', 'test'], default='validation')
    p.add_argument('--dataset', type=Path, help='Relocated dataset directory; the frozen manifest checksum must match')
    p.add_argument('--checkpoint', choices=['best', 'last'], default='best')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--show', action='store_true')
    a = p.parse_args()
    if a.plot_only:
        from core.plotting import plot_saved_evaluation
        plot_saved_evaluation(a.plot_only, show=a.show, dataset=a.dataset)
    else:
        from core.evaluation import evaluate_run
        evaluate_run(a.run, a.role, a.dataset, a.checkpoint, a.device, a.show)


if __name__ == '__main__':
    main()
