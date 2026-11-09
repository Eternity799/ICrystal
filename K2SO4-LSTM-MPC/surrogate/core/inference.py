"""One-step and recursive inference for full-CSD surrogate models."""
from pathlib import Path

import numpy as np
import torch

from .model import ARCHITECTURE, FullCSDLSTM


class Predictor:
    def __init__(self, model, history):
        self.model = model.eval()
        self.history = history
        self.device = next(model.parameters()).device

    @classmethod
    def from_run(cls, run, checkpoint='best', device='cpu'):
        if checkpoint not in ('best', 'last'):
            raise ValueError('checkpoint must be best or last.')
        path = Path(run)/f'{checkpoint}.pt'
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if saved.get('format_version') != 1 or saved.get('architecture') != ARCHITECTURE:
            raise ValueError('Only this full-CSD architecture is supported; legacy PI-AERNN or patched-LSTM weights are incompatible.')
        from .training import source_hashes, choose_device
        if saved['fingerprint']['source'] != source_hashes():
            raise ValueError('Current source differs from the training snapshot. Restore the saved source or retrain; implementations cannot be mixed.')
        device = choose_device(device)
        if device == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
        model = FullCSDLSTM(saved['scaler'], saved['grid'], saved['config']['model'])
        model.load_state_dict(saved['model']); model.to(device)
        result = cls(model, saved['config']['model']['history'])
        result.saved = saved
        return result

    def prepare_history(self, history, total):
        if set(history) != {'n', 'c', 't'}:
            raise ValueError('History requires n (absolute counts), c (concentration) and t (actual temperature).')
        n, c, t = [np.asarray(history[k], dtype=float) for k in ('n', 'c', 't')]
        if n.ndim != 2 or n.shape[1] != 200 or not len(n) or c.shape != (len(n),) or t.shape != c.shape:
            raise ValueError('History requires a nonempty time-by-200 CSD and matching one-dimensional C/T arrays.')
        if not all(np.isfinite(x).all() for x in (n, c, t)) or not np.isfinite(total) or total <= 0:
            raise ValueError('Inputs contain nonfinite states or a nonpositive inventory.')
        if np.any(n < 0) or np.any(c < 0):
            raise ValueError('The ideal-condition interface rejects negative counts or concentration.')
        pm = self.model.particle_mass.detach().cpu().numpy()
        if np.max(np.abs(c+n@pm-total))/total > 1e-8:
            raise ValueError('Observed history does not close the fixed global material inventory.')
        if len(n) < self.history:
            pad = self.history-len(n)
            n = np.concatenate((np.repeat(n[:1], pad, axis=0), n))
            c = np.r_[np.repeat(c[0], pad), c]; t = np.r_[np.repeat(t[0], pad), t]
        return {k:np.array(v[-self.history:], copy=True) for k,v in [('n', n), ('c', c), ('t', t)]}

    def step(self, history, next_temperature, initial_total):
        """The next temperature is the endpoint of a candidate actual path over a fixed 60-second linear interval."""
        h = self.prepare_history(history, initial_total)
        if not np.isfinite(next_temperature):
            raise ValueError('The next temperature must be finite.')
        values = [torch.tensor(h[k], dtype=torch.float64, device=self.device)[None] for k in ('n', 'c', 't')]
        next_t = torch.tensor([next_temperature], dtype=torch.float64, device=self.device)
        total = torch.tensor([initial_total], dtype=torch.float64, device=self.device)
        with torch.no_grad():
            output = self.model(*values, next_t, total)
        result = {k:v[0].detach().cpu().numpy() for k,v in output.items()}
        if not all(np.isfinite(v).all() for v in result.values()):
            raise FloatingPointError('Nonfinite full-CSD prediction.')
        if np.any(result['counts'] < 0) or result['concentration'] < 0:
            raise FloatingPointError('Negative predicted state.')
        return result

    def rollout(self, history, future_temperatures, initial_total):
        """True history is supplied only at the origin; all later CSD/C states are predictions with candidate actual temperatures."""
        h = self.prepare_history(history, initial_total)
        future = np.asarray(future_temperatures, dtype=float)
        if future.ndim != 1 or not np.isfinite(future).all():
            raise ValueError('Future temperatures must be a finite one-dimensional array.')
        n = np.full((len(future)+1, 200), np.nan)
        c = np.full(len(future)+1, np.nan)
        n[0], c[0] = h['n'][-1], h['c'][-1]
        details = []; failed = None
        for j, next_t in enumerate(future):
            if failed is None:
                try:
                    result = self.step(h, float(next_t), float(initial_total))
                    n[j+1], c[j+1] = result['counts'], result['concentration']
                    h = dict(n=np.concatenate((h['n'][1:], result['counts'][None])),
                             c=np.r_[h['c'][1:], result['concentration']], t=np.r_[h['t'][1:], next_t])
                except (ValueError, FloatingPointError) as exc:
                    failed = str(exc); result = dict(invalid_reason=failed)
            else:
                result = dict(invalid_reason='Earlier prediction failed; rollout stopped: '+failed)
            details.append(result)
        return dict(counts=n, concentration=c, details=details,
                    valid_mask=np.isfinite(n).all(-1) & np.isfinite(c))
