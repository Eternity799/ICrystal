"""Full-CSD LSTM with additive-retention outputs and global solute-inventory projection."""
import torch
from torch import nn
from torch.nn import functional as F

from .features import STATE_NAMES, ACTION_NAMES, state_features, action_features

ARCHITECTURE = 'full_csd_lstm_from_scratch_v1'


def append_state(n, c, t, out, next_t):
    return (torch.cat((n[:, 1:], out['counts'][:, None]), 1),
            torch.cat((c[:, 1:], out['concentration'][:, None]), 1),
            torch.cat((t[:, 1:], next_t[:, None]), 1))


class FullCSDLSTM(nn.Module):
    def __init__(self, scaler, grid, config):
        super().__init__()
        self.history = config['history']
        for name in ('count_scale', 'moment_scale', 'state_mean', 'state_std', 'action_mean',
                     'action_std', 'delta_q_scale'):
            self.register_buffer(name, torch.tensor(scaler[name], dtype=torch.float64))
        for name in ('drive_scale', 'concentration_scale'):
            self.register_buffer(name, torch.tensor(float(scaler[name]), dtype=torch.float64))
        self.register_buffer('particle_mass', torch.tensor(grid['particle_mass_kg'], dtype=torch.float64))
        self.register_buffer('pivots', torch.tensor(grid['size_pivots_m'], dtype=torch.float64))
        self.register_buffer('coefficients', torch.tensor(config['solubility'], dtype=torch.float64))
        self.encoder = nn.Sequential(nn.Linear(len(STATE_NAMES), config['embedding']), nn.SiLU())
        self.lstm = nn.LSTM(config['embedding'], config['hidden'], batch_first=True)
        self.direct = nn.Sequential(nn.Linear(len(STATE_NAMES)+len(ACTION_NAMES), config['direct_width']), nn.SiLU())
        self.fusion = nn.Sequential(nn.Linear(config['hidden']+config['direct_width'], config['head_width']), nn.SiLU())
        self.output = nn.Linear(config['head_width'], 400)
        nn.init.normal_(self.output.weight, mean=0., std=.001)
        with torch.no_grad():
            self.output.bias[:200].fill_(-config['init_logit'])
            self.output.bias[200:].fill_(config['init_logit'])

    def forward(self, n, c, t, next_t, total):
        if n.ndim != 3 or n.shape[1:] != (self.history, 200) or c.shape != n.shape[:2] or t.shape != c.shape:
            raise ValueError('Expected B-by-history-by-200 counts and matching concentration/temperature histories.')
        if next_t.shape != (n.shape[0],) or total.shape != next_t.shape:
            raise ValueError('Next temperature and global inventory must have shape [B].')
        n, c, t, next_t, total = [v.double() for v in (n, c, t, next_t, total)]
        x = state_features(n, c, t, total, self.count_scale, self.moment_scale, self.pivots,
                           self.particle_mass, self.coefficients, self.drive_scale)
        x = ((x-self.state_mean)/self.state_std).float()
        u = action_features(c[:, -1], t[:, -1], next_t, self.coefficients, self.drive_scale)
        u = ((u-self.action_mean)/self.action_std).float()
        h, _ = self.lstm(self.encoder(x))
        current = self.direct(torch.cat((x[:, -1], u), -1))
        logits = self.output(self.fusion(torch.cat((h[:, -1], current), -1))).double()
        added = self.count_scale*F.softplus(logits[:, :200])
        retained = torch.sigmoid(logits[:, 200:])
        raw_counts = n[:, -1]*retained + added
        raw_mass = (raw_counts*self.particle_mass).sum(-1)
        alpha = total/torch.maximum(total, raw_mass)
        counts = raw_counts*alpha[:, None]
        mass = (counts*self.particle_mass).sum(-1)
        # Recover concentration from the fixed inventory after projecting crystal mass.
        concentration = (total-mass).clamp_min(0)
        return dict(counts=counts, concentration=concentration, mass=mass, total=total,
                    raw_counts=raw_counts, raw_mass=raw_mass, alpha=alpha,
                    added_counts=added, retained_fraction=retained)


def rollout(model, sample):
    n, c, t = sample['n'], sample['c'], sample['t']
    outputs, current = [], []
    for j in range(sample['future_t'].shape[1]):
        current.append(n[:, -1])
        nt = sample['future_t'][:, j]
        out = model(n, c, t, nt, sample['total'])
        outputs.append(out)
        n, c, t = append_state(n, c, t, out, nt)
    result = {k:torch.stack([out[k] for out in outputs], 1) for k in outputs[0]}
    result['current_n'] = torch.stack(current, 1)
    return result
