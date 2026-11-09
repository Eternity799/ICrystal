"""Quality metrics and supervision derived from full CSD; no independent quality heads or PBM calls."""
from collections.abc import Mapping
import math

import numpy as np
import torch


QUALITY_KEYS = ('d50_um', 'span', 'concentration', 'yield_fraction')


def _real_tensor(value, name, reference=None):
    try:
        tensor = torch.as_tensor(value, device=reference.device if reference is not None else None)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} must be a real-valued array.") from exc
    if tensor.is_complex():
        raise ValueError(f"{name} must be a real-valued array.")
    dtype = reference.dtype if reference is not None else (
        tensor.dtype if isinstance(value, torch.Tensor) and tensor.is_floating_point() else torch.float64)
    
    return torch.as_tensor(value, dtype=dtype, device=tensor.device)


def _finite(tensor, name):
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains nonfinite values.")


def _initial_concentration(value, reference):
    initial = _real_tensor(value, 'initial dissolved concentration C0', reference)
    batch, horizon = reference.shape
    if initial.ndim == 0:
        initial = initial.expand(batch, horizon)
    elif initial.shape == (batch,):
        initial = initial[:, None].expand(batch, horizon)
    else:
        raise ValueError('Initial dissolved concentration C0 must be a scalar or a [B] vector.')
    _finite(initial, 'initial dissolved concentration C0')
    if bool((initial <= 0).any()):
        raise ValueError('Initial dissolved concentration C0 must be finite and positive.')
    return initial


def quality_tensor(counts, concentration, particle_mass, edges, initial_concentration):
    """Return B-by-H quality metrics using a particle-mass-weighted CDF and logarithmic size interpolation. Counts have shape B-by-H-by-bin, concentration B-by-H, and edges are in metres. Quantiles match core.evaluation.cqa; within-bin CDF gradients are retained, with piecewise differentiability across bin boundaries. Empty bins are allowed but total crystal mass must be positive. C0 is initial dissolved concentration, not the seed-inclusive initial_total_solute; dissolution can give negative yield."""
    counts = _real_tensor(counts, 'particle counts')
    if counts.ndim != 3 or any(size <= 0 for size in counts.shape):
        raise ValueError('Counts require a nonempty B-by-H-by-bin shape.')
    concentration = _real_tensor(concentration, 'concentration', counts)
    if concentration.shape != counts.shape[:2]:
        raise ValueError('Concentration must have shape B-by-H matching the counts.')
    particle_mass = _real_tensor(particle_mass, 'per-particle mass', counts)
    edges = _real_tensor(edges, 'size-bin edges', counts)
    bins = counts.shape[-1]
    if particle_mass.shape != (bins,) or edges.shape != (bins+1,):
        raise ValueError('Per-particle mass or size edges do not match the number of bins.')
    for name, tensor in (('particle counts', counts), ('concentration', concentration),
                         ('per-particle mass', particle_mass), ('size-bin edges', edges)):
        _finite(tensor, name)
    if bool((counts < 0).any()) or bool((concentration < 0).any()):
        raise ValueError('Counts and concentration cannot be negative.')
    if bool((particle_mass <= 0).any()):
        raise ValueError('Per-particle mass must be positive.')
    if bool((edges <= 0).any()) or bool((edges[1:] <= edges[:-1]).any()):
        raise ValueError('Size edges must be positive and strictly increasing.')
    initial = _initial_concentration(initial_concentration, concentration)

    mass = counts*particle_mass
    _finite(mass, 'crystal mass per bin')
    total_mass = mass.sum(-1, keepdim=True)
    _finite(total_mass, 'total crystal mass')
    if bool((total_mass <= 0).any()):
        raise ValueError('Quantiles are undefined for zero crystal mass.')
    cdf = mass.cumsum(-1)/total_mass
    probabilities = counts.new_tensor((.1, .5, .9)).view(1, 1, 3).expand(*counts.shape[:2], 3)
    # Locate each percentile before interpolation between logarithmic size edges.
    indices = torch.searchsorted(cdf.contiguous(), probabilities.contiguous(), right=False).clamp_max(bins-1)
    previous = torch.cat((torch.zeros_like(cdf[..., :1]), cdf[..., :-1]), -1)
    lower = previous.gather(-1, indices)
    width = cdf.gather(-1, indices)-lower
    if not bool(torch.isfinite(width).all()) or bool((width <= 0).any()):
        raise ValueError('Invalid mass width in the quantile bin.')
    fraction = (probabilities-lower)/width
    log_edges = edges.log()
    sizes = torch.exp(log_edges[:-1][indices]+fraction*(log_edges[1:][indices]-log_edges[:-1][indices]))*1e6
    d10, d50, d90 = sizes.unbind(-1)
    result = dict(d10_um=d10, d50_um=d50, d90_um=d90,
                  span=(d90-d10)/d50, concentration=concentration,
                  yield_fraction=(initial-concentration)/initial)
    for name, tensor in result.items():
        _finite(tensor, name)
    return result


def _positive_mapping(values, name):
    if not isinstance(values, Mapping) or set(values) != set(QUALITY_KEYS):
        raise ValueError(f"{name} requires exactly four quality-metric scales.")
    result = {}
    for key in QUALITY_KEYS:
        value = _real_tensor(values[key], f'{name}.{key}')
        if value.ndim != 0:
            raise ValueError(f"{name}.{key} must be a finite positive scalar.")
        scalar = float(value)
        if not math.isfinite(scalar) or scalar <= 0:
            raise ValueError(f"{name}.{key} must be a finite positive scalar.")
        result[key] = scalar
    return result


def _scale_bundle(scales):
    if not isinstance(scales, Mapping) or set(scales) != {'absolute', 'response'}:
        raise ValueError('Quality scales require absolute and response sections.')
    return {kind: _positive_mapping(scales[kind], kind) for kind in ('absolute', 'response')}


def _quality_pair(out, sample, particle_mass, edges):
    predicted_n = _real_tensor(out['counts'], 'predicted counts')
    predicted = quality_tensor(predicted_n, out['concentration'], particle_mass, edges,
                               sample['initial_concentration'])
    reference = predicted['concentration']
    target_n = _real_tensor(sample['target_n'], 'true counts', reference)
    target_c = _real_tensor(sample['target_c'], 'true concentration', reference)
    if target_n.shape != predicted_n.shape or target_c.shape != reference.shape:
        raise ValueError('Predicted quality and true-label shapes must match.')
    target = quality_tensor(target_n, target_c, particle_mass, edges, sample['initial_concentration'])
    return predicted, target


def quality_components(out, sample, particle_mass, edges, scales):
    """Absolute normalized MSE for four quality metrics; future labels are used only for scoring."""
    scale = _scale_bundle(scales)['absolute']
    predicted, target = _quality_pair(out, sample, particle_mass, edges)
    return {key: ((predicted[key]-target[key])/scale[key]).square().mean() for key in QUALITY_KEYS}


def quality_response_components(out, sample, particle_mass, edges, scales):
    """Quality-difference MSE for branches sharing history; branch 0 is baseline, and gradients flow to both predictions. Inputs contain one B-by-H-by-bin group, B at least 2. Independent base trajectories cannot be combined as one response group; provided n/c/t histories must match across branches."""
    scale = _scale_bundle(scales)['response']
    predicted, target = _quality_pair(out, sample, particle_mass, edges)
    reference = predicted['concentration']
    if reference.shape[0] < 2:
        raise ValueError('Quality response requires a baseline and at least one nonzero-action branch.')
    initial = _initial_concentration(sample['initial_concentration'], reference)
    if not bool((initial == initial[:1]).all()):
        raise ValueError('Same-state response branches must share the initial dissolved concentration C0.')
    for key in ('n', 'c', 't'):
        if key not in sample:
            continue
        history = _real_tensor(sample[key], f"{key} history", reference)
        if history.ndim < 1 or history.shape[0] != reference.shape[0]:
            raise ValueError('Response history must match the branch batch dimension.')
        _finite(history, f"{key} history")
        if not bool((history == history[:1]).all()):
            raise ValueError('Quality-response branches must share identical history.')
    return {key: (((predicted[key][1:]-predicted[key][:1])
                   -(target[key][1:]-target[key][:1]))/scale[key]).square().mean()
            for key in QUALITY_KEYS}


def fit_quality_scales(rows, absolute_floors, response_floors):
    """Fit scales only from already loaded train labels, without additional file or role reads. Absolute scales use population standard deviation (ddof=0) over all saved train times. Response scales use future RMS differences of nonzero-offset branches against the unique zero-offset baseline, excluding the shared origin but retaining genuinely zero responses. Missing groups use the floor. Each scale is max(estimate, explicit positive floor), returned as a serializable float."""
    rows = list(rows)
    if not rows:
        raise ValueError('Quality scales require nonempty train base trajectories.')
    try:
        roles = [str(row['role']) for row in rows]
    except (KeyError, TypeError) as exc:
        raise ValueError('Each base trajectory must provide its train role.') from exc
    
    if any(role != 'train' for role in roles):
        raise ValueError('Quality scales may only be fitted to train base trajectories.')
    floors = dict(absolute=_positive_mapping(absolute_floors, 'absolute_floors'),
                  response=_positive_mapping(response_floors, 'response_floors'))
    absolute_values = {key: [] for key in QUALITY_KEYS}
    response_squares = dict.fromkeys(QUALITY_KEYS, 0.)
    response_count = 0
    for row in rows:
        n = np.asarray(row['counts'], dtype=np.float64)
        c = np.asarray(row['concentration'], dtype=np.float64)
        if n.ndim != 2 or n.shape[0] < 1 or c.shape != n.shape[:1]:
            raise ValueError('Train trajectories require time-by-bin counts and time concentration arrays.')
        initial = float(c[0])
        pm, edges = row['particle_mass_kg'], row['size_edges_m']
        absolute = quality_tensor(torch.as_tensor(n)[None], torch.as_tensor(c)[None], pm, edges, initial)
        for key in QUALITY_KEYS:
            absolute_values[key].append(absolute[key][0].detach().cpu().numpy())
        for group in row.get('groups', []):
            if str(group.get('role', 'train')) != 'train':
                raise ValueError('Response scales may only read train branches.')
            offsets = np.asarray(group['offsets_c'], dtype=np.float64)
            if (offsets.ndim != 1 or not np.isfinite(offsets).all()
                    or len(np.unique(offsets)) != len(offsets)):
                raise ValueError('Train-response offsets must be finite, distinct and one-dimensional.')
            zero = np.flatnonzero(offsets == 0.)
            if len(zero) != 1:
                raise ValueError('Train responses require exactly one zero-offset baseline.')
            selected = np.flatnonzero(offsets != 0.)
            if not len(selected):
                continue
            branch_n = np.asarray(group['counts'], dtype=np.float64)
            branch_c = np.asarray(group['concentration'], dtype=np.float64)
            if (branch_n.ndim != 3 or branch_n.shape[0] != len(offsets)
                    or branch_n.shape[1] < 2 or branch_n.shape[-1] != n.shape[-1]
                    or branch_c.shape != branch_n.shape[:2]):
                raise ValueError('Train-response labels require branch-by-(H+1)-by-bin counts and corresponding concentrations.')
            if (not np.allclose(branch_n[:, 0], branch_n[:1, 0], rtol=1e-12, atol=1e-14)
                    or not np.allclose(branch_c[:, 0], branch_c[:1, 0], rtol=1e-12, atol=1e-14)):
                raise ValueError('Train-response branches must start from the same true state.')
            response = quality_tensor(torch.as_tensor(branch_n), torch.as_tensor(branch_c), pm, edges, initial)
            baseline = int(zero[0])
            for key in QUALITY_KEYS:
                values = response[key].detach().cpu().numpy()
                delta = values[selected, 1:]-values[baseline:baseline+1, 1:]
                try:
                    with np.errstate(over='raise', invalid='raise'):
                        response_squares[key] += float(np.square(delta).sum())
                except FloatingPointError as exc:
                    raise ValueError('Train quality-response scales overflowed or are nonfinite.') from exc
            response_count += len(selected)*(branch_n.shape[1]-1)
    result = dict(absolute={}, response={})
    for key in QUALITY_KEYS:
        try:
            with np.errstate(over='raise', invalid='raise'):
                std = float(np.std(np.concatenate(absolute_values[key]), ddof=0))
                rms = math.sqrt(response_squares[key]/response_count) if response_count else 0.
        except (FloatingPointError, OverflowError) as exc:
            raise ValueError('Train quality scales overflowed or are nonfinite.') from exc
        if not math.isfinite(std) or not math.isfinite(rms):
            raise ValueError('Train quality scales must be finite.')
        result['absolute'][key] = max(std, floors['absolute'][key])
        result['response'][key] = max(rms, floors['response'][key])
    return result
