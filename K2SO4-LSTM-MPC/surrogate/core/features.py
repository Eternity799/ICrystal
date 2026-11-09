"""State and candidate-temperature features for the full-CSD LSTM."""
import torch

STATE_NAMES = [f'log_bin_{i}' for i in range(200)] + [
    'concentration', 'temperature', 'current_drive', 'moment0', 'moment1',
    'moment2', 'moment3', 'initial_total', 'crystal_fraction', 'past_delta_t']
ACTION_NAMES = ['next_temperature', 'delta_t', 'drive_at_end_with_current_c',
                'drive_at_midpoint_with_current_c', 'saturation_with_current_c', 'next_solubility']


def solubility(t, coefficients):
    return coefficients[0] + coefficients[1]*t + coefficients[2]*t.square()


def signed_log(x, scale):
    return torch.sign(x)*torch.log1p(x.abs()/scale)


def state_features(n, c, t, total, count_scale, moment_scale, pivots, particle_mass,
                   coefficients, drive_scale):
    q = torch.log1p(n/count_scale)
    moments = torch.stack([(n*pivots.pow(j)).sum(-1) for j in range(4)], -1)
    logs = torch.log1p(moments/moment_scale)
    expanded_total = total[:, None].expand_as(c)
    past_rate = torch.cat((torch.zeros_like(t[:, :1]), t[:, 1:]-t[:, :-1]), 1)
    mass_fraction = (n*particle_mass).sum(-1)/expanded_total
    drive = signed_log(c-solubility(t, coefficients), drive_scale)
    return torch.cat((q, c[..., None], t[..., None], drive[..., None], logs,
                      expanded_total[..., None], mass_fraction[..., None], past_rate[..., None]), -1)


def action_features(c, t, next_t, coefficients, drive_scale):
    cs_end = solubility(next_t, coefficients)
    cs_mid = solubility((t+next_t)/2, coefficients)
    return torch.stack((next_t, next_t-t, signed_log(c-cs_end, drive_scale),
                        signed_log(c-cs_mid, drive_scale), c/cs_end, cs_end), -1)
