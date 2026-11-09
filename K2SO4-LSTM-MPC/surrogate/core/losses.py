"""Full-CSD reconstruction and paired-action response losses."""
import torch


def full_csd_loss(out, truth_n, truth_c, model):
    pm = model.particle_mass
    error = out['counts']-truth_n
    number = truth_n.sum(-1).clamp_min(1e-30)
    mass = (truth_n*pm).sum(-1).clamp_min(1e-30)
    q = torch.log1p(out['counts']/model.count_scale)
    target_q = torch.log1p(truth_n/model.count_scale)
    
    return dict(number=(error.abs().sum(-1)/number).mean(),
                mass=((error.abs()*pm).sum(-1)/mass).mean(),
                log=(q-target_q).square().mean(),
                change_scaled=((q-target_q)/model.delta_q_scale).square().mean(),
                concentration=((out['concentration']-truth_c)/model.concentration_scale).square().mean(),
                inventory=torch.relu((out['raw_mass']-out['total'])/out['total']).square().mean())


def weighted(parts, weights):
    if set(parts) != set(weights):
        raise ValueError('Loss components and configured weights differ.')
    return sum(parts[k]*weights[k] for k in parts)


def response_loss(predictions, truths, model, number_floor=1e-4, mass_floor=1e-4):
    """Shape: nonzero branches by 2 (alternative, baseline) by optional horizon by 200 bins. Global monotonicity is not assumed."""
    if predictions.shape != truths.shape or predictions.ndim < 3 or predictions.shape[1] != 2 or predictions.shape[-1] != 200:
        raise ValueError('Response loss requires paired full-200-bin predictions and labels.')
    dp = predictions[:, 0]-predictions[:, 1]
    dt = truths[:, 0]-truths[:, 1]
    baseline = truths[:, 1]
    number_scale = torch.maximum(dt.abs().sum(-1), number_floor*baseline.sum(-1)).clamp_min(1e-30)
    mass_scale = torch.maximum((dt.abs()*model.particle_mass).sum(-1),
                              mass_floor*(baseline*model.particle_mass).sum(-1)).clamp_min(1e-30)
    return dict(response_number=((dp-dt).abs().sum(-1)/number_scale).mean(),
                response_mass=(((dp-dt).abs()*model.particle_mass).sum(-1)/mass_scale).mean())


def pair_losses(model, sample, response_config):
    from .model import rollout
    out = rollout(model, sample)
    absolute = full_csd_loss(out, sample['target_n'], sample['target_c'], model)
    # Pair every alternative branch with the shared zero-offset baseline.
    pairs = torch.stack((out['counts'][1:], out['counts'][0:1].expand_as(out['counts'][1:])), 1)
    targets = torch.stack((sample['target_n'][1:], sample['target_n'][0:1].expand_as(sample['target_n'][1:])), 1)
    response = response_loss(pairs, targets, model, response_config['number_floor'], response_config['mass_floor'])
    return absolute, response, out
