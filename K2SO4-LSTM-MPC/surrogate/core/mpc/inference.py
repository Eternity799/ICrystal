"""Checkpoint loading for control-oriented full-CSD LSTM models."""
from pathlib import Path

import torch

from ..inference import Predictor
from ..model import ARCHITECTURE, FullCSDLSTM
from ..training import choose_device


class MPCPredictor(Predictor):
    @classmethod
    def from_run(cls,run,checkpoint='best',device='auto'):
        from .training import WORKFLOW, source_hashes
        if checkpoint not in ('best','last'):raise ValueError('checkpoint must be best or last.')
        saved=torch.load(Path(run)/f'{checkpoint}.pt',map_location='cpu',weights_only=False)
        if saved.get('workflow')!=WORKFLOW or saved.get('architecture')!=ARCHITECTURE:
            raise ValueError('A single-arm run from train_mpc.py is required; use evaluate.py for the earlier workflow.')
        if saved['fingerprint']['source']!=source_hashes():
            raise ValueError('Source differs from the training snapshot; restore run/source rather than mixing implementations.')
        device=choose_device(device)
        if device=='cuda':
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False
            torch.backends.cuda.matmul.allow_tf32=False
        model=FullCSDLSTM(saved['scaler'],saved['grid'],saved['config']['model'])
        model.load_state_dict(saved['model']);model.to(device)
        result=cls(model,saved['config']['model']['history']);result.saved=saved
        return result
