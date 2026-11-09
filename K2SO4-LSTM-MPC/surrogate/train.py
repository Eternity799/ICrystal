"""Train a full-CSD LSTM from random initialization."""
import argparse
from pathlib import Path

from core.config import BASE, load_config
from core.training import run_training


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=BASE/'config.yaml')
    p.add_argument('--mode', choices=['smoke', 'learn', 'formal'], default='smoke')
    p.add_argument('--dataset', type=Path, help='Complete schema-1 dataset directory, relative to the current shell directory')
    p.add_argument('--resume', type=Path, help='Restore last.pt from this same training run, not an older-model fine-tune')
    a = p.parse_args()
    config = load_config(a.config, a.mode, a.dataset)
    run_training(config, a.config.resolve(), a.resume)


if __name__ == '__main__':
    main()
